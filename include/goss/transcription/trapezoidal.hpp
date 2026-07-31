// include/goss/transcription/trapezoidal.hpp
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "goss/ad/cppadcg_backend.hpp"
#include "goss/nlp/nlp_problem.hpp"
#include "goss/transcription/ocp_problem.hpp"
#include "goss/transcription/transcription.hpp"
#include "goss/transcription/variable_layout.hpp"

namespace goss::transcription {

struct Trapezoidal {
    template <typename DynamicsFn, typename CostFn>
    static CompiledOcp compile(const OcpProblem<DynamicsFn, CostFn>& ocp,
                               const std::string& model_name = "goss_trap") {
        ocp.mesh.validate();
        const std::size_t ns = ocp.num_states;
        const std::size_t nc = ocp.num_controls;
        const std::size_t nn = ocp.mesh.num_nodes();
        const std::size_t ni = ocp.mesh.num_intervals;
        const double t0 = ocp.mesh.t_initial;
        const double h = ocp.mesh.interval_width();
        VariableLayout layout(ns, nc, nn);

        // The packed functor: captures ocp by value (functors are cheap), layout by value.
        // Uses generic lambda so it can be instantiated with the AD type during recording.
        auto packed = [ocp, layout, ns, nc, nn, ni, t0, h](const auto& z) {
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

            // Output 0: trapezoidal quadrature of running cost.
            // Endpoints weight h/2, interior nodes weight h.
            T cost = T(0);
            for (std::size_t k = 0; k < nn; ++k) {
                T tk = T(t0 + static_cast<double>(k) * h);
                T Lk = ocp.cost(state_at(k), control_at(k), tk);
                // trapezoid weights: endpoints h/2, interior h
                T weight = (k == 0 || k == nn - 1) ? T(h / 2.0) : T(h);
                cost += weight * Lk;
            }
            outputs.push_back(cost);

            // Outputs 1..: defects per interval per state.
            // For interval k: x_{k+1}[i] - x_k[i] - (h/2)*(f_k[i] + f_{k+1}[i]) == 0
            for (std::size_t k = 0; k < ni; ++k) {
                T tk = T(t0 + static_cast<double>(k) * h);
                T tk1 = T(t0 + static_cast<double>(k + 1) * h);
                auto xk = state_at(k);
                auto xk1 = state_at(k + 1);
                auto fk = ocp.dynamics(xk, control_at(k), tk);
                auto fk1 = ocp.dynamics(xk1, control_at(k + 1), tk1);
                for (std::size_t i = 0; i < ns; ++i) {
                    outputs.push_back(xk1[i] - xk[i] - T(h / 2.0) * (fk[i] + fk1[i]));
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
        // Pin fixed boundary states via equal bounds (simplest approach: fixed variable).
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
};

}  // namespace goss::transcription
