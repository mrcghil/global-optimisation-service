// include/goss/transcription/hermite_simpson.hpp
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

struct HermiteSimpson {
    // Primary overload: explicit non-uniform node times.
    // All implementation logic lives here; the uniform path delegates to this.
    template <typename DynamicsFn, typename CostFn, typename AlgResFn>
    static CompiledOcp compile(const OcpProblem<DynamicsFn, CostFn, AlgResFn>& ocp,
                               const NonUniformMesh& mesh,
                               const std::string& model_name = "goss_hs") {
        mesh.validate();
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

        const std::size_t na = ocp.num_algebraic;
        // Validate algebraic bound vectors before touching any element.
        if (na > 0) {
            if (ocp.algebraic_lower_bounds.size() != na || ocp.algebraic_upper_bounds.size() != na) {
                throw TranscriptionError(
                    "compile: algebraic_lower_bounds and algebraic_upper_bounds "
                    "must each have size == num_algebraic");
            }
        }
        VariableLayout layout(ns, nc, nn, na);

        // Capture node_times by value so the packed functor owns the data.
        // Per-interval hk = node_times[k+1] - node_times[k]; per-node tk = node_times[k].
        const std::vector<double> node_times = mesh.node_times;

        // Packed functor: captures ocp, layout, and node_times by value (cheap functors).
        // Generic lambda so it can be instantiated with the AD scalar type during recording.
        // na is captured explicitly so the algebraic residual loop is zero-overhead when na==0.
        auto packed = [ocp, layout, ns, nc, ni, na, node_times](const auto& z) {
            using T = typename std::decay_t<decltype(z)>::value_type;
            std::vector<T> outputs;
            // Reserve: 1 cost + ni*ns defects + nn*na algebraic residuals.
            // nn = ni + 1 (one more node than intervals).
            const std::size_t nn_local = ni + 1;
            outputs.reserve(1 + ni * ns + nn_local * na);

            // Helper lambdas to extract x_k, u_k from z.
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
            // Extract the algebraic variable vector at a node. Returns empty vector when na==0.
            auto algebraic_at = [&](std::size_t node) {
                std::vector<T> alg_vars(na);
                for (std::size_t k = 0; k < na; ++k)
                    alg_vars[k] = z[layout.algebraic_index(node, k)];
                return alg_vars;
            };
            auto midpoint_control = [&](const std::vector<T>& uk, const std::vector<T>& uk1) {
                std::vector<T> um(nc);
                for (std::size_t j = 0; j < nc; ++j) um[j] = T(0.5) * (uk[j] + uk1[j]);
                return um;
            };

            // Output 0: Simpson quadrature cost.
            // Per interval k: (hk/6)*(L_k + 4*L_mid + L_{k+1})
            // where x_mid is the Hermite interpolated midpoint (not a decision variable)
            // and hk = node_times[k+1] - node_times[k] is the per-interval step size.
            T cost = T(0);
            std::vector<T> defects;
            defects.reserve(ni * ns);

            for (std::size_t k = 0; k < ni; ++k) {
                // Per-interval step size and times (each interval may differ).
                T tk   = T(node_times[k]);
                T tk1  = T(node_times[k + 1]);
                T hk   = tk1 - tk;
                // Midpoint time is the arithmetic mean of the two endpoint times.
                T tmid = T(0.5) * (tk + tk1);

                auto xk  = state_at(k);
                auto xk1 = state_at(k + 1);
                auto uk  = control_at(k);
                auto uk1 = control_at(k + 1);

                // Dynamics at the left and right endpoints.
                auto fk  = ocp.dynamics(xk, uk, tk);
                auto fk1 = ocp.dynamics(xk1, uk1, tk1);

                // Hermite interpolated midpoint state (compressed form — no decision variable):
                //   x_mid[i] = 0.5*(x_k[i] + x_{k+1}[i]) + (hk/8)*(f_k[i] - f_{k+1}[i])
                std::vector<T> xmid(ns);
                for (std::size_t i = 0; i < ns; ++i)
                    xmid[i] = T(0.5) * (xk[i] + xk1[i]) + (hk / T(8)) * (fk[i] - fk1[i]);

                // Midpoint control and dynamics.
                auto umid = midpoint_control(uk, uk1);
                auto fmid = ocp.dynamics(xmid, umid, tmid);

                // Hermite-Simpson defect per state i:
                //   x_{k+1}[i] - x_k[i] - (hk/6)*(f_k[i] + 4*f_mid[i] + f_{k+1}[i]) = 0
                for (std::size_t i = 0; i < ns; ++i)
                    defects.push_back(xk1[i] - xk[i] - (hk / T(6)) * (fk[i] + T(4) * fmid[i] + fk1[i]));

                // Simpson cost contribution for this interval.
                T Lk   = ocp.cost(xk, uk, tk);
                T Lmid = ocp.cost(xmid, umid, tmid);
                T Lk1  = ocp.cost(xk1, uk1, tk1);
                cost += (hk / T(6)) * (Lk + T(4) * Lmid + Lk1);
            }

            // Pack outputs: cost first, then defects (same order as Trapezoidal), then
            // algebraic residuals at each node (one per algebraic variable per node).
            outputs.push_back(cost);
            for (auto& d : defects) outputs.push_back(d);

            // Algebraic residual constraints: g(x_k, u_k, alg_k, t_k) == 0 at every node k.
            // One equality constraint per algebraic variable per node.
            // These are added AFTER the defect rows so existing constraint indexing is unchanged.
            if (na > 0) {
                for (std::size_t k = 0; k < nn_local; ++k) {
                    T tk = T(node_times[k]);
                    auto residuals_at_k = ocp.algebraic_residuals_functor(
                        state_at(k), control_at(k), algebraic_at(k), tk);
                    // residuals_at_k has size na; push each residual as a separate constraint row.
                    for (std::size_t j = 0; j < na; ++j) {
                        outputs.push_back(residuals_at_k[j]);
                    }
                }
            }
            return outputs;
        };

        auto backend = std::make_unique<goss::ad::CppADCGBackend>(
            packed, layout.total_variables(), model_name);

        // Variable bounds: per-node state and control bounds.
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
        // Guards use i < size() to prevent out-of-bounds access when the caller
        // passes a shorter-than-ns fixed vector (Task 4 corrected form).
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

        // Algebraic variable bounds: per-node, same bound for every node.
        // (Algebraic variables are not pinned at boundary nodes — they are free to vary
        // as long as the residual constraint is satisfied.)
        if (na > 0) {
            for (std::size_t k = 0; k < nn; ++k) {
                for (std::size_t j = 0; j < na; ++j) {
                    const std::size_t alg_idx = layout.algebraic_index(k, j);
                    zl[alg_idx] = ocp.algebraic_lower_bounds[j];
                    zu[alg_idx] = ocp.algebraic_upper_bounds[j];
                }
            }
        }

        // Constraint bounds: defect rows are equalities [0,0]; algebraic residual rows
        // are also equalities [0,0] (the solver enforces g == 0 at every collocation node).
        const std::size_t num_defects = ni * ns;
        const std::size_t num_alg_constraints = nn * na;   // one per algebraic per node
        const std::size_t total_constraints = num_defects + num_alg_constraints;
        std::vector<double> gl(total_constraints, 0.0), gu(total_constraints, 0.0);

        auto problem = std::make_unique<nlp::NLPProblem>(
            std::move(backend), std::move(zl), std::move(zu), std::move(gl), std::move(gu));
        return CompiledOcp{std::move(problem), layout};
    }

    // Backward-compatible uniform overload: delegates to the non-uniform path.
    // Three-param template so it accepts OcpProblem<Dyn,Cost,AlgResFn> (including the
    // default AlgResFn=NoAlgebraicResiduals used by all existing ODE callers).
    template <typename DynamicsFn, typename CostFn, typename AlgResFn>
    static CompiledOcp compile(const OcpProblem<DynamicsFn, CostFn, AlgResFn>& ocp,
                               const std::string& model_name = "goss_hs") {
        return compile(ocp, to_nonuniform(ocp.mesh), model_name);
    }
};

}  // namespace goss::transcription
