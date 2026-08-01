// include/goss/transcription/legendre_gauss_lobatto.hpp
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "goss/ad/cppadcg_backend.hpp"
#include "goss/nlp/nlp_problem.hpp"
#include "goss/transcription/errors.hpp"
#include "goss/transcription/lgl_nodes.hpp"
#include "goss/transcription/mesh.hpp"
#include "goss/transcription/ocp_problem.hpp"
#include "goss/transcription/transcription.hpp"
#include "goss/transcription/variable_layout.hpp"

namespace goss::transcription {

/// Pseudospectral collocation using Legendre-Gauss-Lobatto (LGL) nodes.
/// The entire time horizon [t0, tf] is collocated at n LGL nodes mapped from [-1,1].
/// Convergence is spectral (super-algebraic) for smooth problems — exponential
/// in the number of nodes, unlike the algebraic O(h^p) of local schemes.
///
/// The ODE is enforced at nodes k = 1..n-1 via the global differentiation matrix D:
///   (D @ X)[k, s] = (tf-t0)/2 * f_s(x_k, u_k, t_k)   for k = 1..n-1, all s
/// Node 0's ODE defect is omitted because the initial state is pinned (fixed by
/// variable bounds); x(0) still enters all other defects through the dense D matrix,
/// so the initial condition propagates globally into every remaining equation.
/// This produces dense coupling (every node in every defect), unlike the banded
/// local schemes. For moderate n (up to ~40) this is still efficient; for large n
/// consider multiple LGL sub-intervals (hp-pseudospectral, out of scope here).
///
/// REQUIREMENT: all initial state components MUST be pinned via set_initial_state /
/// initial_state_fixed before calling compile(). Free initial states are not supported
/// (omitting the node-0 defect only avoids overdetermination when x(0) is fixed).
struct LegendreGaussLobatto {
    template <typename DynamicsFn, typename CostFn>
    static CompiledOcp compile(const OcpProblem<DynamicsFn, CostFn>& ocp,
                               const std::string& model_name = "goss_lgl") {
        ocp.mesh.validate();
        const std::size_t nn = ocp.mesh.num_nodes();  // number of LGL nodes
        const std::size_t ns = ocp.num_states;
        const std::size_t nc = ocp.num_controls;
        const double t0 = ocp.mesh.t_initial;
        const double tf = ocp.mesh.t_final;
        const double half_duration = 0.5 * (tf - t0);

        if (nn < 2)
            throw TranscriptionError(
                "LegendreGaussLobatto: need at least 2 nodes (num_intervals >= 1)");
        if (ocp.state_lower.size() != ns || ocp.state_upper.size() != ns)
            throw TranscriptionError("compile: state bound vectors must have size == num_states");
        if (ocp.control_lower.size() != nc || ocp.control_upper.size() != nc)
            throw TranscriptionError("compile: control bound vectors must have size == num_controls");
        if (ocp.initial_state.size() != ns || ocp.final_state.size() != ns)
            throw TranscriptionError("compile: initial_state/final_state must have size == num_states");

        // Guard: all initial state components must be pinned.
        // The node-0 collocation defect is omitted (to avoid overdetermination when
        // x(0) is fixed). If any x_i(0) is free, the system becomes underdetermined
        // for that component — IPOPT would find an arbitrary x_i(0) silently.
        for (std::size_t i = 0; i < ns; ++i) {
            // OcpProblem convention: initial_state_fixed[i] != 0.0 means component i is pinned.
            const bool pinned =
                (i < ocp.initial_state_fixed.size()) && (ocp.initial_state_fixed[i] != 0.0);
            if (!pinned) {
                throw TranscriptionError(
                    "LegendreGaussLobatto: all initial states must be pinned "
                    "(set_initial_state) — free initial states are not supported "
                    "(the node-0 collocation defect is omitted for the pinned case).");
            }
        }

        // Pre-compute LGL nodes on [-1,1] and map to [t0,tf].
        std::vector<double> lgl_xi, lgl_weights_ref;
        lgl_nodes_and_weights(nn, lgl_xi, lgl_weights_ref);
        // t_k = t0 + half_duration * (xi_k + 1)
        std::vector<double> t_lgl(nn);
        for (std::size_t k = 0; k < nn; ++k)
            t_lgl[k] = t0 + half_duration * (lgl_xi[k] + 1.0);
        // Gauss-Lobatto quadrature weights scaled to [t0, tf]:
        // integral ~ sum_k w_k * f(t_k) where w_k = half_duration * lgl_weights_ref[k].
        std::vector<double> lgl_weights_physical(nn);
        for (std::size_t k = 0; k < nn; ++k)
            lgl_weights_physical[k] = half_duration * lgl_weights_ref[k];

        // Differentiation matrix D on [-1,1]: D_{kj} = d phi_j / d xi at xi_k.
        const std::vector<double> D = lgl_differentiation_matrix(lgl_xi);

        VariableLayout layout(ns, nc, nn);

        // Packed functor — captures pre-computed D, t_lgl, weights by value.
        auto packed = [ocp, layout, ns, nc, nn, D, t_lgl,
                       lgl_weights_physical, half_duration](const auto& z) {
            using T = typename std::decay_t<decltype(z)>::value_type;
            // Outputs: 1 cost + (nn-1)*ns defects (nodes 1..nn-1)
            std::vector<T> outputs;
            outputs.reserve(1 + (nn - 1) * ns);

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

            // Output 0: Gauss-Lobatto quadrature of running cost.
            T cost = T(0);
            for (std::size_t k = 0; k < nn; ++k) {
                T tk = T(t_lgl[k]);
                cost += T(lgl_weights_physical[k]) * ocp.cost(state_at(k), control_at(k), tk);
            }
            outputs.push_back(cost);

            // Outputs 1..(nn-1)*ns: LGL collocation defects at nodes 1..nn-1.
            // Node 0 is excluded: its state value is fixed by variable bounds
            // (initial condition pinning), so including a defect there would
            // create a structurally overdetermined system for IPOPT. Nodes 1..nn-1
            // still reference x(0) through the dense differentiation matrix D, so
            // the initial condition propagates into all (nn-1)*ns defects.
            //
            // Defect at node k, state s:
            //   sum_j D[k,j] * x_s(j) - (tf-t0)/2 * f_s(x_k, u_k, t_k) = 0
            // Pre-compute dynamics at each node.
            std::vector<std::vector<T>> F(nn);
            for (std::size_t k = 0; k < nn; ++k)
                F[k] = ocp.dynamics(state_at(k), control_at(k), T(t_lgl[k]));

            for (std::size_t k = 1; k < nn; ++k) {
                for (std::size_t s = 0; s < ns; ++s) {
                    // Differentiation: (D @ x_s)[k] = sum_j D[k,j] * x_s(j)
                    T Dx_ks = T(0);
                    for (std::size_t j = 0; j < nn; ++j)
                        Dx_ks += T(D[k * nn + j]) * z[layout.state_index(j, s)];
                    outputs.push_back(Dx_ks - T(half_duration) * F[k][s]);
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
                const std::size_t idx = layout.state_index(k, i);
                zl[idx] = ocp.state_lower[i];
                zu[idx] = ocp.state_upper[i];
            }
            for (std::size_t j = 0; j < nc; ++j) {
                const std::size_t idx = layout.control_index(k, j);
                zl[idx] = ocp.control_lower[j];
                zu[idx] = ocp.control_upper[j];
            }
        }
        // Pin fixed boundary states.  Node 0 is t=t0 (first LGL node = -1 mapped to t0).
        // Node nn-1 is t=tf (last LGL node = +1 mapped to tf).
        for (std::size_t i = 0; i < ns; ++i) {
            if (i < ocp.initial_state_fixed.size() && ocp.initial_state_fixed[i] != 0.0) {
                const std::size_t idx = layout.state_index(0, i);
                zl[idx] = zu[idx] = ocp.initial_state[i];
            }
            if (i < ocp.final_state_fixed.size() && ocp.final_state_fixed[i] != 0.0) {
                const std::size_t idx = layout.state_index(nn - 1, i);
                zl[idx] = zu[idx] = ocp.final_state[i];
            }
        }

        // Constraint bounds: (nn-1)*ns defects (nodes 1..nn-1), all equalities [0,0].
        const std::size_t num_defects = (nn - 1) * ns;
        std::vector<double> gl(num_defects, 0.0), gu(num_defects, 0.0);

        auto problem = std::make_unique<nlp::NLPProblem>(
            std::move(backend), std::move(zl), std::move(zu),
            std::move(gl), std::move(gu));
        return CompiledOcp{std::move(problem), layout};
    }

    /// NonUniformMesh overload — always throws.
    ///
    /// LGL uses global single-interval collocation over [t0, tf] at LGL nodes.
    /// Multi-interval adaptive mesh refinement (refine_and_solve) is meaningless
    /// for this scheme — the mesh is the collocation grid, not an AMR partition.
    /// Use Trapezoidal or HermiteSimpson for adaptive mesh refinement.
    template <typename DynamicsFn, typename CostFn>
    static CompiledOcp compile(const OcpProblem<DynamicsFn, CostFn>&,
                               const NonUniformMesh&,
                               const std::string&) {
        throw TranscriptionError(
            "LegendreGaussLobatto does not support NonUniformMesh / refine_and_solve: "
            "LGL uses global single-interval collocation; multi-interval refinement "
            "requires hp-pseudospectral (out of scope). Use Trapezoidal or "
            "HermiteSimpson for adaptive mesh refinement.");
    }
};

}  // namespace goss::transcription
