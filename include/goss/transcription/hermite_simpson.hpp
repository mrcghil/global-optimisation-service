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
#include "goss/transcription/invoke.hpp"
#include "goss/transcription/variable_layout.hpp"

namespace goss::transcription {

struct HermiteSimpson {
    // Primary overload: explicit non-uniform node times.
    // All implementation logic lives here; the uniform path delegates to this.
    // Four template params so the overload accepts OcpProblem<Dyn,Cost,AlgResFn,PathConstraintFn>.
    // Existing 2- and 3-param OcpProblem callers still deduce correctly because
    // AlgResFn and PathConstraintFn both have defaults.
    template <typename DynamicsFn, typename CostFn, typename AlgResFn, typename PathConstraintFn>
    static CompiledOcp compile(const OcpProblem<DynamicsFn, CostFn, AlgResFn, PathConstraintFn>& ocp,
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

        const std::size_t na  = ocp.num_algebraic;
        const std::size_t npc = ocp.num_path_constraints;
        // Validate algebraic bound vectors before touching any element.
        if (na > 0) {
            if (ocp.algebraic_lower_bounds.size() != na || ocp.algebraic_upper_bounds.size() != na) {
                throw TranscriptionError(
                    "compile: algebraic_lower_bounds and algebraic_upper_bounds "
                    "must each have size == num_algebraic");
            }
        }
        // Validate path-constraint bound vectors when path constraints are present.
        if (npc > 0) {
            if (ocp.path_constraint_lower.size() != npc) {
                throw TranscriptionError(
                    "compile: path_constraint_lower must have size == num_path_constraints");
            }
            if (ocp.path_constraint_upper.size() != npc) {
                throw TranscriptionError(
                    "compile: path_constraint_upper must have size == num_path_constraints");
            }
        }
        VariableLayout layout(ns, nc, nn, na);

        // Capture node_times by value so the packed functor owns the data.
        // Per-interval hk = node_times[k+1] - node_times[k]; per-node tk = node_times[k].
        const std::vector<double> node_times = mesh.node_times;

        const std::size_t np = ocp.num_parameters;

        // Packed functor: captures ocp, layout, node_times, and np by value (cheap functors).
        // Generic lambda so it can be instantiated with the AD scalar type during recording.
        // Accepts (z, p) where p is the parameter vector (empty when np == 0).
        // na is captured explicitly so the algebraic residual loop is zero-overhead when na==0.
        // npc captured alongside na so the path-constraint loop is zero-overhead when npc==0.
        // IMPORTANT: algebraic_residuals_functor and path_constraints are NOT threaded through
        // call_dynamics/call_cost — they have their own fixed arities and are out of scope.
        auto packed = [ocp, layout, ns, nc, ni, na, npc, node_times, np](const auto& z, const auto& p) {
            using T = typename std::decay_t<decltype(z)>::value_type;
            std::vector<T> outputs;
            // Reserve: 1 cost + ni*ns defects + nn*na algebraic residuals + nn*npc path rows.
            // nn = ni + 1 (one more node than intervals).
            const std::size_t nn_local = ni + 1;
            outputs.reserve(1 + ni * ns + nn_local * na + nn_local * npc);

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
                auto fk  = detail::call_dynamics(ocp.dynamics, xk,  uk,  p, tk);
                auto fk1 = detail::call_dynamics(ocp.dynamics, xk1, uk1, p, tk1);

                // Hermite interpolated midpoint state (compressed form — no decision variable):
                //   x_mid[i] = 0.5*(x_k[i] + x_{k+1}[i]) + (hk/8)*(f_k[i] - f_{k+1}[i])
                std::vector<T> xmid(ns);
                for (std::size_t i = 0; i < ns; ++i)
                    xmid[i] = T(0.5) * (xk[i] + xk1[i]) + (hk / T(8)) * (fk[i] - fk1[i]);

                // Midpoint control and dynamics.
                auto umid = midpoint_control(uk, uk1);
                auto fmid = detail::call_dynamics(ocp.dynamics, xmid, umid, p, tmid);

                // Hermite-Simpson defect per state i:
                //   x_{k+1}[i] - x_k[i] - (hk/6)*(f_k[i] + 4*f_mid[i] + f_{k+1}[i]) = 0
                for (std::size_t i = 0; i < ns; ++i)
                    defects.push_back(xk1[i] - xk[i] - (hk / T(6)) * (fk[i] + T(4) * fmid[i] + fk1[i]));

                // Simpson cost contribution for this interval.
                T Lk   = detail::call_cost(ocp.cost, xk,   uk,   p, tk);
                T Lmid = detail::call_cost(ocp.cost, xmid, umid, p, tmid);
                T Lk1  = detail::call_cost(ocp.cost, xk1,  uk1,  p, tk1);
                cost += (hk / T(6)) * (Lk + T(4) * Lmid + Lk1);
            }

            // Pack outputs: cost first, then defects (same order as Trapezoidal), then
            // algebraic residuals at each node (one per algebraic variable per node).
            outputs.push_back(cost);
            for (auto& d : defects) outputs.push_back(d);

            // Algebraic residual constraints: g(x_k, u_k, alg_k, t_k) == 0 at every node k.
            // One equality constraint per algebraic variable per node.
            // These are added AFTER the defect rows so existing constraint indexing is unchanged.
            // NOTE: algebraic_residuals_functor takes (x, u, alg, t) — parameters are NOT
            // threaded here (out of scope; alg_vars slot is occupied by algebraic vars, not p).
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

            // Path-constraint rows: g(x_k, u_k, t_k) at every collocation node k.
            // Evaluated AFTER algebraic rows so the index base is num_defects + num_alg_constraints.
            // Output order: node-major — all npc constraints for node 0, then node 1, etc.
            // path_constraints takes (x, u, t) — 3 args, no alg_vars (unlike algebraic_residuals_functor).
            // NOTE: parameters are NOT threaded into path_constraints (out of scope for Task 5).
            if (npc > 0) {
                for (std::size_t k = 0; k < nn_local; ++k) {
                    T tk = T(node_times[k]);
                    auto gk = ocp.path_constraints(state_at(k), control_at(k), tk);
                    for (std::size_t j = 0; j < npc; ++j) {
                        outputs.push_back(gk[j]);
                    }
                }
            }
            return outputs;
        };

        std::unique_ptr<goss::ad::CppADCGBackend> backend;
        if (np > 0) {
            backend = std::make_unique<goss::ad::CppADCGBackend>(
                packed, layout.total_variables(), np, ocp.parameter_defaults, model_name);
        } else {
            // Wrap the two-arg packed functor as one-arg (empty p) to reuse the
            // existing single-argument constructor path unchanged.
            auto packed_no_params = [packed](const auto& z) {
                using T = typename std::decay_t<decltype(z)>::value_type;
                return packed(z, std::vector<T>{});
            };
            backend = std::make_unique<goss::ad::CppADCGBackend>(
                packed_no_params, layout.total_variables(), model_name);
        }

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

        // Constraint bounds — layout MUST match the packed functor output order:
        //   [0 .. num_defects)                                     : defect equalities [0,0]
        //   [num_defects .. num_defects+num_alg_constraints)       : algebraic equalities [0,0]
        //   [num_defects+num_alg_constraints .. total_constraints) : path rows [lower[j], upper[j]]
        const std::size_t num_defects         = ni * ns;
        const std::size_t num_alg_constraints = nn * na;   // one per algebraic per node
        const std::size_t num_path_rows       = nn * npc;  // one per path constraint per node
        const std::size_t total_constraints   = num_defects + num_alg_constraints + num_path_rows;
        // Initialise all to [0,0]; defect and algebraic rows stay as equalities.
        std::vector<double> gl(total_constraints, 0.0), gu(total_constraints, 0.0);
        // Fill path-constraint bounds after the algebraic block (arbitrary [lower, upper]).
        for (std::size_t k = 0; k < nn; ++k) {
            for (std::size_t j = 0; j < npc; ++j) {
                const std::size_t row = num_defects + num_alg_constraints + k * npc + j;
                gl[row] = ocp.path_constraint_lower[j];
                gu[row] = ocp.path_constraint_upper[j];
            }
        }

        auto problem = std::make_unique<nlp::NLPProblem>(
            std::move(backend), std::move(zl), std::move(zu), std::move(gl), std::move(gu));
        return CompiledOcp{std::move(problem), layout};
    }

    // Backward-compatible uniform overload: delegates to the non-uniform path.
    // Four-param template so it accepts OcpProblem<Dyn,Cost,AlgResFn,PathConstraintFn>
    // (including 2- and 3-param callers where AlgResFn and PathConstraintFn default).
    template <typename DynamicsFn, typename CostFn, typename AlgResFn, typename PathConstraintFn>
    static CompiledOcp compile(const OcpProblem<DynamicsFn, CostFn, AlgResFn, PathConstraintFn>& ocp,
                               const std::string& model_name = "goss_hs") {
        return compile(ocp, to_nonuniform(ocp.mesh), model_name);
    }
};

}  // namespace goss::transcription
