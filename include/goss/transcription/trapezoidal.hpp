// include/goss/transcription/trapezoidal.hpp
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "goss/ad/cppadcg_backend.hpp"
#include "goss/nlp/nlp_problem.hpp"
#include "goss/transcription/errors.hpp"
#include "goss/transcription/mesh.hpp"
#include "goss/transcription/ocp_problem.hpp"
#include "goss/transcription/transcription.hpp"
#include "goss/transcription/variable_layout.hpp"

namespace goss::transcription {

struct Trapezoidal {
    // Primary overload: explicit non-uniform node times.
    // All implementation logic lives here; the uniform path delegates to this.
    template <typename DynamicsFn, typename CostFn,
              typename AlgResFn, typename PathConstraintFn>
    static CompiledOcp compile(const OcpProblem<DynamicsFn, CostFn, AlgResFn, PathConstraintFn>& ocp,
                               const NonUniformMesh& mesh,
                               const std::string& model_name = "goss_trap") {
        mesh.validate();

        // Fail loudly if the caller passes a DAE problem (num_algebraic > 0).
        // Trapezoidal does not support algebraic variables in v1; silently dropping
        // them would produce a wrong-but-plausible ODE-only NLP.
        if (ocp.num_algebraic > 0) {
            throw TranscriptionError(
                "Trapezoidal::compile: algebraic variables (num_algebraic > 0) are not "
                "supported by Trapezoidal in v1; use HermiteSimpson for DAE problems.");
        }
        // Fail loudly if the caller passes a path-constrained problem.
        // Trapezoidal does not support path constraints in v1; use HermiteSimpson instead.
        if (ocp.num_path_constraints > 0) {
            throw TranscriptionError(
                "Trapezoidal::compile: path constraints (num_path_constraints > 0) are not "
                "supported by Trapezoidal in v1; use HermiteSimpson for path-constrained problems.");
        }

        const std::size_t ns = ocp.num_states;
        const std::size_t nc = ocp.num_controls;
        const std::size_t nn = mesh.num_nodes();
        const std::size_t ni = mesh.num_intervals();

        // Validate bound-vector sizes before touching any element.
        if (ocp.state_lower.size() != ns || ocp.state_upper.size() != ns)
            throw TranscriptionError("compile: state bound vectors must have size == num_states");
        if (ocp.control_lower.size() != nc || ocp.control_upper.size() != nc)
            throw TranscriptionError("compile: control bound vectors must have size == num_controls");
        if (ocp.initial_state.size() != ns || ocp.final_state.size() != ns)
            throw TranscriptionError("compile: initial_state/final_state must have size == num_states");

        VariableLayout layout(ns, nc, nn);

        // Capture node_times by value so the packed functor owns the data.
        // Per-interval hk = node_times[k+1] - node_times[k]; per-node tk = node_times[k].
        const std::vector<double> node_times = mesh.node_times;

        // The packed functor: captures ocp, layout, and node_times by value.
        // Uses generic lambda so it can be instantiated with the AD type during recording.
        auto packed = [ocp, layout, ns, nc, nn, ni, node_times](const auto& z) {
            using T = typename std::decay_t<decltype(z)>::value_type;
            std::vector<T> outputs;
            outputs.reserve(1 + ni * ns);

            // Helper lambdas to slice x_k, u_k from z.
            auto state_at = [&](std::size_t node) {
                std::vector<T> x(ns);
                for (std::size_t i = 0; i < ns; ++i) x[i] = z[layout.state_index(node, i)];
                return x;
            };
            auto control_at = [&](std::size_t node) {
                std::vector<T> u(nc);
                for (std::size_t j = 0; j < nc; ++j) u[j] = z[layout.control_index(node, j)];
                return u;
            };

            // Output 0: trapezoidal cost quadrature summed per-interval.
            // Each interval k contributes (hk/2)*(L_k + L_{k+1}).
            T cost = T(0);
            for (std::size_t k = 0; k < ni; ++k) {
                T tk  = T(node_times[k]);
                T tk1 = T(node_times[k + 1]);
                T hk  = tk1 - tk;
                T Lk  = ocp.cost(state_at(k),     control_at(k),     tk);
                T Lk1 = ocp.cost(state_at(k + 1), control_at(k + 1), tk1);
                cost += T(0.5) * hk * (Lk + Lk1);
            }
            outputs.push_back(cost);

            // Outputs 1..: defects per interval per state.
            // For interval k: x_{k+1}[i] - x_k[i] - (hk/2)*(f_k[i] + f_{k+1}[i]) == 0
            for (std::size_t k = 0; k < ni; ++k) {
                T tk  = T(node_times[k]);
                T tk1 = T(node_times[k + 1]);
                T hk  = tk1 - tk;
                auto xk  = state_at(k);
                auto xk1 = state_at(k + 1);
                auto fk  = ocp.dynamics(xk,  control_at(k),     tk);
                auto fk1 = ocp.dynamics(xk1, control_at(k + 1), tk1);
                for (std::size_t i = 0; i < ns; ++i) {
                    outputs.push_back(xk1[i] - xk[i] - T(0.5) * hk * (fk[i] + fk1[i]));
                }
            }
            return outputs;
        };

        auto backend = std::make_unique<goss::ad::CppADCGBackend>(
            packed, layout.total_variables(), model_name);

        // Bounds.
        const std::size_t nv = layout.total_variables();
        std::vector<double> zl(nv, -kInf), zu(nv, kInf);
        for (std::size_t k = 0; k < nn; ++k) {
            for (std::size_t i = 0; i < ns; ++i) {
                std::size_t idx = layout.state_index(k, i);
                zl[idx] = ocp.state_lower[i];
                zu[idx] = ocp.state_upper[i];
            }
            for (std::size_t j = 0; j < nc; ++j) {
                std::size_t idx = layout.control_index(k, j);
                zl[idx] = ocp.control_lower[j];
                zu[idx] = ocp.control_upper[j];
            }
        }
        // Pin fixed boundary states via equal bounds.
        // Guard with i < size() so a caller passing a shorter-than-ns fixed vector cannot
        // trigger out-of-bounds access (UB). The contract is that when a pin fires the
        // corresponding initial_state / final_state entry is also valid (same size).
        for (std::size_t i = 0; i < ns; ++i) {
            if (i < ocp.initial_state_fixed.size() && ocp.initial_state_fixed[i] != 0.0) {
                std::size_t idx = layout.state_index(0, i);
                zl[idx] = zu[idx] = ocp.initial_state[i];
            }
            if (i < ocp.final_state_fixed.size() && ocp.final_state_fixed[i] != 0.0) {
                std::size_t idx = layout.state_index(nn - 1, i);
                zl[idx] = zu[idx] = ocp.final_state[i];
            }
        }

        // Constraint bounds: all defect constraints are equalities [0,0].
        const std::size_t num_defects = ni * ns;
        std::vector<double> gl(num_defects, 0.0), gu(num_defects, 0.0);

        auto problem = std::make_unique<nlp::NLPProblem>(
            std::move(backend), std::move(zl), std::move(zu), std::move(gl), std::move(gu));
        return CompiledOcp{std::move(problem), layout};
    }

    // Backward-compatible uniform overload: delegates to the non-uniform path.
    // This is a one-line wrapper; all implementation is in the overload above.
    template <typename DynamicsFn, typename CostFn,
              typename AlgResFn, typename PathConstraintFn>
    static CompiledOcp compile(const OcpProblem<DynamicsFn, CostFn, AlgResFn, PathConstraintFn>& ocp,
                               const std::string& model_name = "goss_trap") {
        return compile(ocp, to_nonuniform(ocp.mesh), model_name);
    }
};

}  // namespace goss::transcription
