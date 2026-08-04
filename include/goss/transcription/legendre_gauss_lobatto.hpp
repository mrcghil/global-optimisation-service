// include/goss/transcription/legendre_gauss_lobatto.hpp
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "goss/ad/cppadcg_backend.hpp"
#include "goss/nlp/nlp_problem.hpp"
#include "goss/transcription/errors.hpp"
#include "goss/transcription/hp_mesh.hpp"
#include "goss/transcription/lgl_nodes.hpp"
#include "goss/transcription/mesh.hpp"
#include "goss/transcription/ocp_problem.hpp"
#include "goss/transcription/transcription.hpp"
#include "goss/transcription/invoke.hpp"
#include "goss/transcription/variable_layout.hpp"

namespace goss::transcription {

/// Pseudospectral collocation using Legendre-Gauss-Lobatto (LGL) nodes.
/// The entire time horizon [t0, tf] is collocated at n LGL nodes mapped from [-1,1].
/// Convergence is spectral (super-algebraic) for smooth problems — exponential
/// in the number of nodes, unlike the algebraic O(h^p) of local schemes.
///
/// Collocation strategy (node-0 handling):
///   When controls are present (nc > 0):
///     The ODE is enforced at ALL n nodes k = 0..n-1:
///       (D @ X)[k, s] = (tf-t0)/2 * f_s(x_k, u_k, t_k)   for k = 0..n-1, all s
///     Including node 0 is required because u_0 enters the cost quadrature but
///     has no other constraint — omitting the node-0 defect lets the optimizer
///     drive u_0 to zero to reduce the cost, degrading convergence to O(1/n).
///     DOF balance (nc>0, after pinning x(0)): nn*ns defect equations vs
///     (nn-1)*ns free states + nn*nc controls — the system is underdetermined,
///     giving the optimizer freedom; the cost functional properly constrains u_0.
///
///   When no controls are present (nc == 0):
///     The ODE is enforced at nodes k = 1..n-1 only:
///       (D @ X)[k, s] = (tf-t0)/2 * f_s(x_k, t_k)   for k = 1..n-1, all s
///     Including node 0 would overdetermine the system: nn*ns equations but only
///     (nn-1)*ns free state unknowns (x(0) is pinned). Dropping node 0 keeps
///     the system square; x(0) still propagates globally via the dense D matrix.
///
/// This produces dense coupling (every node in every defect), unlike the banded
/// local schemes. For moderate n (up to ~40) this is still efficient; for large n
/// or solutions with sharp features, use compile_hp() for hp-pseudospectral
/// multi-segment collocation.
///
/// REQUIREMENT: all initial state components MUST be pinned via set_initial_state /
/// initial_state_fixed before calling compile(). Free initial states are not supported
/// (for nc==0 the node-0 defect is dropped to avoid overdetermination, so a free
/// x(0) would be unconstrained; for nc>0 the pinned x(0) anchors the trajectory).
struct LegendreGaussLobatto {
    template <typename DynamicsFn, typename CostFn,
              typename AlgResFn, typename PathConstraintFn>
    static CompiledOcp compile(const OcpProblem<DynamicsFn, CostFn, AlgResFn, PathConstraintFn>& ocp,
                               const std::string& model_name = "goss_lgl") {
        ocp.mesh.validate();

        // Fail loudly if the caller passes a DAE problem (num_algebraic > 0).
        // LegendreGaussLobatto does not support algebraic variables in v1; silently
        // dropping them would produce a wrong-but-plausible ODE-only NLP.
        if (ocp.num_algebraic > 0) {
            throw TranscriptionError(
                "LegendreGaussLobatto::compile: algebraic variables (num_algebraic > 0) are not "
                "supported by LegendreGaussLobatto in v1; use HermiteSimpson for DAE problems.");
        }
        // Fail loudly if the caller passes a path-constrained problem.
        // LegendreGaussLobatto does not support path constraints in v1; use HermiteSimpson instead.
        if (ocp.num_path_constraints > 0) {
            throw TranscriptionError(
                "LegendreGaussLobatto::compile: path constraints (num_path_constraints > 0) are not "
                "supported by LegendreGaussLobatto in v1; use HermiteSimpson for path-constrained problems.");
        }

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
        // For nc==0: the node-0 defect is dropped to avoid overdetermination; a free
        // x(0) would be unconstrained and IPOPT would find an arbitrary value silently.
        // For nc>0: node 0 is collocated, but x(0) is still pinned by variable bounds
        // so the initial trajectory anchor is correct.
        for (std::size_t i = 0; i < ns; ++i) {
            // OcpProblem convention: initial_state_fixed[i] != 0.0 means component i is pinned.
            const bool pinned =
                (i < ocp.initial_state_fixed.size()) && (ocp.initial_state_fixed[i] != 0.0);
            if (!pinned) {
                throw TranscriptionError(
                    "LegendreGaussLobatto: all initial states must be pinned "
                    "(set_initial_state) — free initial states are not supported "
                    "(x(0) must be fixed to anchor the trajectory correctly).");
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

        const std::size_t np = ocp.num_parameters;

        // Determine the first collocation node.
        // When controls are present (nc > 0): collocate node 0 so that u_0 is
        // constrained by the ODE — omitting it lets the optimizer drive u_0 to zero
        // (u_0 enters the cost but no defect), degrading convergence to O(1/n).
        // When nc == 0: drop node 0 to avoid overdetermination — with x(0) pinned
        // there are only (nn-1)*ns free state unknowns but nn*ns equations would be
        // generated, making the NLP structurally infeasible.
        const std::size_t first_collocation_node = (nc > 0) ? 0 : 1;
        const std::size_t num_collocation_nodes  = nn - first_collocation_node;

        // Packed functor — captures pre-computed D, t_lgl, weights, and np by value.
        // Accepts (z, p) where p is the parameter vector (empty when np == 0).
        auto packed = [ocp, layout, ns, nc, nn, D, t_lgl,
                       lgl_weights_physical, half_duration,
                       first_collocation_node, num_collocation_nodes, np](const auto& z, const auto& p) {
            using T = typename std::decay_t<decltype(z)>::value_type;
            // Outputs: 1 cost + num_collocation_nodes*ns defects
            std::vector<T> outputs;
            outputs.reserve(1 + num_collocation_nodes * ns);

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
                cost += T(lgl_weights_physical[k]) *
                        detail::call_cost(ocp.cost, state_at(k), control_at(k), p, tk);
            }
            outputs.push_back(cost);

            // Collocation defects: LGL ODE residuals at nodes first_collocation_node..nn-1.
            //
            // nc > 0 case: defects at k = 0..nn-1 (all nodes).
            //   Node 0's defect constrains u_0, which also enters the cost quadrature.
            //   Without this constraint the optimizer would freely minimize cost w.r.t.
            //   u_0, driving it to zero and biasing the objective. The system is
            //   underdetermined (nn*ns eqs, (nn-1)*ns + nn*nc unknowns after pinning x(0));
            //   the cost functional provides the additional regularization.
            //
            // nc == 0 case: defects at k = 1..nn-1 only (node 0 excluded).
            //   With x(0) pinned, including node 0 would create nn*ns equations for only
            //   (nn-1)*ns unknowns — structurally overdetermined. x(0) still propagates
            //   into all (nn-1)*ns defects through the dense differentiation matrix D.
            //
            // Defect at node k, state s:
            //   sum_j D[k,j] * x_s(j) - (tf-t0)/2 * f_s(x_k, u_k, t_k) = 0
            // Pre-compute dynamics at each node.
            std::vector<std::vector<T>> F(nn);
            for (std::size_t k = 0; k < nn; ++k)
                F[k] = detail::call_dynamics(ocp.dynamics, state_at(k), control_at(k), p, T(t_lgl[k]));

            for (std::size_t k = first_collocation_node; k < nn; ++k) {
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

        // Constraint bounds: num_collocation_nodes*ns defects, all equalities [0,0].
        // For nc > 0: num_collocation_nodes == nn   (nodes 0..nn-1 collocated).
        // For nc == 0: num_collocation_nodes == nn-1 (node 0 excluded to avoid overdetermination).
        const std::size_t num_defects = num_collocation_nodes * ns;
        std::vector<double> gl(num_defects, 0.0), gu(num_defects, 0.0);

        auto problem = std::make_unique<nlp::NLPProblem>(
            std::move(backend), std::move(zl), std::move(zu),
            std::move(gl), std::move(gu));
        std::vector<model::ParameterSpec> param_specs;
        param_specs.reserve(ocp.num_parameters);
        for (std::size_t i = 0; i < ocp.num_parameters; ++i)
            param_specs.push_back(model::ParameterSpec{
                ocp.parameter_names[i], ocp.parameter_defaults[i],
                ocp.parameter_lower[i], ocp.parameter_upper[i]});
        return CompiledOcp{std::move(problem), layout, model::ParameterValidator(std::move(param_specs))};
    }

    /// NonUniformMesh overload — always throws.
    ///
    /// LGL uses global single-interval collocation over [t0, tf] at LGL nodes.
    /// Multi-interval adaptive mesh refinement (refine_and_solve) is meaningless
    /// for this scheme — the mesh is the collocation grid, not an AMR partition.
    /// Use Trapezoidal or HermiteSimpson for adaptive mesh refinement.
    template <typename DynamicsFn, typename CostFn,
              typename AlgResFn, typename PathConstraintFn>
    static CompiledOcp compile(const OcpProblem<DynamicsFn, CostFn, AlgResFn, PathConstraintFn>&,
                               const NonUniformMesh&,
                               const std::string&) {
        throw TranscriptionError(
            "LegendreGaussLobatto does not support NonUniformMesh / refine_and_solve: "
            "LGL uses global single-interval collocation. For multi-segment collocation "
            "use compile_hp() (hp-pseudospectral); for adaptive mesh refinement use "
            "Trapezoidal or HermiteSimpson.");
    }

    /// hp-Pseudospectral collocation: partition [t0,tf] into S segments, each with
    /// its own set of LGL nodes and differentiation matrix. State continuity across
    /// segment boundaries is enforced via explicit equality constraints in the NLP.
    ///
    /// hp_mesh specifies: segment_boundary_times (size S+1) and per_segment_node_count
    /// (size S). The ocp.mesh.t_initial / ocp.mesh.t_final must match hp_mesh.
    ///
    /// Boundary nodes are DUPLICATED (not shared): segment s owns global nodes
    /// [offset_s, offset_s + n_s) where offset_s = sum_{r<s} n_r.
    /// Continuity constraints tie the last state of segment s to the first state
    /// of segment s+1. Controls are DISCONTINUOUS across segment boundaries.
    ///
    /// Node-0 collocation strategy (Reconciliation 2 — matches single-interval compile()):
    ///   nc > 0: collocate node 0 for ALL segments (first_collocation_node = 0).
    ///     WHY: each segment's u_0 enters the cost quadrature but is not tied by
    ///     continuity (continuity constrains states only). Without a node-0 defect,
    ///     the optimizer drives u_0 → 0 per segment, biasing the objective (O(1/n) bug
    ///     re-introduced per segment). Collocating node 0 constrains u_0 properly;
    ///     the system is underdetermined (cost provides additional regularisation).
    ///   nc == 0: skip node 0 for ALL segments (first_collocation_node = 1).
    ///     WHY: with x(0) pinned (segment 0) or continuity-tied (segments 1..S-1),
    ///     collocating node 0 would overdetermine the system — same reasoning as
    ///     single-interval compile() for nc==0. Matches S=1 single-interval exactly.
    ///
    /// REQUIREMENT: all initial state components must be pinned (same guard as compile()).
    template <typename DynamicsFn, typename CostFn,
              typename AlgResFn, typename PathConstraintFn>
    static CompiledOcp compile_hp(
            const OcpProblem<DynamicsFn, CostFn, AlgResFn, PathConstraintFn>& ocp,
            const HpMesh& hp_mesh,
            const std::string& model_name = "goss_lgl_hp") {
        hp_mesh.validate();
        ocp.mesh.validate();

        // Guard: algebraic variables not supported in hp-pseudospectral v1.
        if (ocp.num_algebraic > 0) {
            throw TranscriptionError(
                "LegendreGaussLobatto::compile_hp: algebraic variables "
                "(num_algebraic > 0) are not supported by hp-pseudospectral in v1; "
                "use HermiteSimpson for DAE problems.");
        }
        // Guard: path constraints not supported in hp-pseudospectral v1.
        if (ocp.num_path_constraints > 0) {
            throw TranscriptionError(
                "LegendreGaussLobatto::compile_hp: path constraints "
                "(num_path_constraints > 0) are not supported by hp-pseudospectral "
                "in v1; use HermiteSimpson for path-constrained problems.");
        }
        // Guard: parameters not supported in hp-pseudospectral v1.
        // compile_hp uses the non-parametric backend (single-argument constructor),
        // so a problem with num_parameters > 0 would silently ignore parameters,
        // producing a wrong NLP. Fail loudly here rather than at parameter-injection time.
        if (ocp.num_parameters > 0) {
            throw TranscriptionError(
                "LegendreGaussLobatto::compile_hp: parameters (num_parameters > 0) are not "
                "supported by hp-pseudospectral in v1; use HermiteSimpson::compile or "
                "Trapezoidal::compile for parametric problems.");
        }

        // Verify that the hp_mesh time horizon matches the OcpProblem mesh.
        if (std::abs(hp_mesh.t_initial() - ocp.mesh.t_initial) > 1e-12 ||
            std::abs(hp_mesh.t_final()   - ocp.mesh.t_final  ) > 1e-12) {
            throw TranscriptionError(
                "compile_hp: hp_mesh time horizon [" +
                std::to_string(hp_mesh.t_initial()) + ", " +
                std::to_string(hp_mesh.t_final()) +
                "] does not match ocp.mesh [" +
                std::to_string(ocp.mesh.t_initial) + ", " +
                std::to_string(ocp.mesh.t_final) + "]");
        }

        const std::size_t num_states   = ocp.num_states;
        const std::size_t num_controls = ocp.num_controls;
        const std::size_t num_seg      = hp_mesh.num_segments();
        const std::size_t total_nodes  = hp_mesh.total_nodes();

        if (total_nodes < 2)
            throw TranscriptionError("compile_hp: total_nodes must be >= 2");

        if (ocp.state_lower.size() != num_states || ocp.state_upper.size() != num_states)
            throw TranscriptionError(
                "compile_hp: state bound vectors must have size == num_states");
        if (ocp.control_lower.size() != num_controls ||
                ocp.control_upper.size() != num_controls)
            throw TranscriptionError(
                "compile_hp: control bound vectors must have size == num_controls");
        if (ocp.initial_state.size() != num_states ||
                ocp.final_state.size() != num_states)
            throw TranscriptionError(
                "compile_hp: initial_state/final_state must have size == num_states");

        // Guard: all initial state components must be pinned (same as compile()).
        // WHY: segment 0 node-0 is the trajectory anchor; for nc==0 its defect
        // is dropped (would overdetermine), so a free x(0) is unconstrained;
        // for nc>0 the pinned x(0) correctly anchors the trajectory.
        for (std::size_t state_idx = 0; state_idx < num_states; ++state_idx) {
            const bool pinned =
                (state_idx < ocp.initial_state_fixed.size()) &&
                (ocp.initial_state_fixed[state_idx] != 0.0);
            if (!pinned) {
                throw TranscriptionError(
                    "LegendreGaussLobatto::compile_hp: all initial states must be pinned "
                    "(same requirement as compile()): free initial states are not "
                    "supported.");
            }
        }

        // --- Per-segment node-0 collocation decision (Reconciliation 2) ---
        // Applies uniformly to every segment.
        //   nc > 0: first_collocation_node = 0 (collocate all nodes including node 0).
        //   nc == 0: first_collocation_node = 1 (skip node 0 to avoid overdetermination).
        // This mirrors single-interval compile() exactly and fixes the per-segment
        // u_0 unconstrained bias that would otherwise be introduced by hp segmentation.
        const std::size_t first_collocation_node = (num_controls > 0) ? 0 : 1;

        // --- Pre-compute per-segment LGL data ---
        // For segment s: LGL nodes on [-1,1], physical times, physical quadrature weights,
        // and the differentiation matrix D_s (on reference [-1,1]).
        // All captured by value into the packed functor.
        std::vector<std::vector<double>> all_D(num_seg);
        std::vector<std::vector<double>> all_t_nodes(num_seg);
        std::vector<std::vector<double>> all_weights_phys(num_seg);
        std::vector<double>              all_half_dur(num_seg);

        for (std::size_t seg = 0; seg < num_seg; ++seg) {
            const std::size_t seg_n_nodes = hp_mesh.per_segment_node_count[seg];
            const double t_a_s  = hp_mesh.segment_boundary_times[seg];
            const double t_b_s  = hp_mesh.segment_boundary_times[seg + 1];
            const double half_dur_s = 0.5 * (t_b_s - t_a_s);
            all_half_dur[seg] = half_dur_s;

            std::vector<double> lgl_xi_seg, lgl_weights_ref_seg;
            lgl_nodes_and_weights(seg_n_nodes, lgl_xi_seg, lgl_weights_ref_seg);

            // Physical node times: t_k = t_a_s + half_dur_s * (xi_k + 1).
            all_t_nodes[seg].resize(seg_n_nodes);
            for (std::size_t local_k = 0; local_k < seg_n_nodes; ++local_k)
                all_t_nodes[seg][local_k] =
                    t_a_s + half_dur_s * (lgl_xi_seg[local_k] + 1.0);

            // Physical quadrature weights: w_phys_k = half_dur_s * w_ref_k.
            all_weights_phys[seg].resize(seg_n_nodes);
            for (std::size_t local_k = 0; local_k < seg_n_nodes; ++local_k)
                all_weights_phys[seg][local_k] =
                    half_dur_s * lgl_weights_ref_seg[local_k];

            // Local differentiation matrix D_s on [-1,1].
            // Collocation defect at local node k, state i:
            //   sum_j D_s[k,j] * x_i(offset+j) - half_dur_s * f_i(x_k, u_k, t_k) = 0
            // (The half_dur factor appears in the defect; D_s is on the reference domain.)
            all_D[seg] = lgl_differentiation_matrix(lgl_xi_seg);
        }

        // Global node offsets: offset_s = sum_{r<s} n_r.
        std::vector<std::size_t> global_node_offsets(num_seg, 0);
        for (std::size_t seg = 1; seg < num_seg; ++seg)
            global_node_offsets[seg] =
                global_node_offsets[seg - 1] + hp_mesh.per_segment_node_count[seg - 1];

        // --- Compute constraint counts for reserve / gl-gu sizing ---
        // Defects: per segment, nodes first_collocation_node..n_s-1, all states.
        //   nc == 0: (n_s - 1) * ns per segment (skip node 0)
        //   nc > 0:  n_s       * ns per segment (include node 0)
        // Continuity: (S-1) * ns equality constraints.
        std::size_t num_defects = 0;
        for (std::size_t seg = 0; seg < num_seg; ++seg)
            num_defects +=
                (hp_mesh.per_segment_node_count[seg] - first_collocation_node) * num_states;
        const std::size_t num_continuity =
            (num_seg > 1 ? num_seg - 1 : 0) * num_states;
        const std::size_t num_constraints_total = num_defects + num_continuity;

        VariableLayout layout(num_states, num_controls, total_nodes);

        // --- Packed functor (AD-safe: no std::function, captures by value) ---
        // Captures: ocp (dynamics/cost functors by value), layout, all pre-computed
        // per-segment data, and per_segment_node_count. No virtual dispatch.
        auto packed = [ocp,
                       layout,
                       num_states,
                       num_controls,
                       num_seg,
                       num_defects,
                       num_continuity,
                       first_collocation_node,
                       all_D,
                       all_t_nodes,
                       all_weights_phys,
                       all_half_dur,
                       global_node_offsets,
                       per_segment_node_count = hp_mesh.per_segment_node_count]
                      (const auto& z) {
            using T = typename std::decay_t<decltype(z)>::value_type;

            auto state_at = [&](std::size_t global_node_idx) {
                std::vector<T> x(num_states);
                for (std::size_t state_i = 0; state_i < num_states; ++state_i)
                    x[state_i] = z[layout.state_index(global_node_idx, state_i)];
                return x;
            };
            auto control_at = [&](std::size_t global_node_idx) {
                std::vector<T> u(num_controls);
                for (std::size_t ctrl_j = 0; ctrl_j < num_controls; ++ctrl_j)
                    u[ctrl_j] = z[layout.control_index(global_node_idx, ctrl_j)];
                return u;
            };

            std::vector<T> outputs;
            outputs.reserve(1 + num_defects + num_continuity);

            // --- Output 0: total running cost via LGL quadrature over ALL nodes ---
            // Cost sums all nodes regardless of first_collocation_node — the quadrature
            // weights are purely a cost concern, independent of the collocation skip.
            T total_cost = T(0);
            for (std::size_t seg = 0; seg < num_seg; ++seg) {
                const std::size_t seg_n_nodes   = per_segment_node_count[seg];
                const std::size_t global_offset = global_node_offsets[seg];
                for (std::size_t local_k = 0; local_k < seg_n_nodes; ++local_k) {
                    const std::size_t global_k = global_offset + local_k;
                    total_cost +=
                        T(all_weights_phys[seg][local_k]) *
                        ocp.cost(state_at(global_k), control_at(global_k),
                                 T(all_t_nodes[seg][local_k]));
                }
            }
            outputs.push_back(total_cost);

            // --- Outputs 1..num_defects: per-segment collocation defects ---
            // Node-0 handling (Reconciliation 2):
            //   nc > 0 (first_collocation_node == 0): defect loop k = 0..n_s-1.
            //     Constrains u_0 of every segment via the ODE, preventing the optimizer
            //     from driving it to zero (which would bias the objective O(1/n)).
            //   nc == 0 (first_collocation_node == 1): defect loop k = 1..n_s-1.
            //     Avoids overdetermination: with x(0) pinned or continuity-tied, adding
            //     a node-0 defect would create more equations than free state unknowns.
            for (std::size_t seg = 0; seg < num_seg; ++seg) {
                const std::size_t seg_n_nodes   = per_segment_node_count[seg];
                const std::size_t global_offset = global_node_offsets[seg];
                const double half_dur_s         = all_half_dur[seg];

                // Pre-compute dynamics at each node of this segment.
                std::vector<std::vector<T>> F_seg(seg_n_nodes);
                for (std::size_t local_k = 0; local_k < seg_n_nodes; ++local_k) {
                    const std::size_t global_k = global_offset + local_k;
                    F_seg[local_k] = ocp.dynamics(
                        state_at(global_k), control_at(global_k),
                        T(all_t_nodes[seg][local_k]));
                }

                // Collocation defect at local node k, state i:
                //   sum_j D_s[k,j] * x_i(global_offset+j) - half_dur_s * F_seg[k][i] = 0
                for (std::size_t local_k = first_collocation_node;
                         local_k < seg_n_nodes; ++local_k) {
                    for (std::size_t state_i = 0; state_i < num_states; ++state_i) {
                        T Dx_k_i = T(0);
                        for (std::size_t local_j = 0; local_j < seg_n_nodes; ++local_j) {
                            const std::size_t global_j = global_offset + local_j;
                            Dx_k_i +=
                                T(all_D[seg][local_k * seg_n_nodes + local_j]) *
                                z[layout.state_index(global_j, state_i)];
                        }
                        outputs.push_back(
                            Dx_k_i - T(half_dur_s) * F_seg[local_k][state_i]);
                    }
                }
            }

            // --- Outputs (1+num_defects)..(num_defects+num_continuity): continuity ---
            // For boundary between segment s and segment s+1, state i:
            //   x_i(last_node_of_s) - x_i(first_node_of_(s+1)) = 0
            // Controls are NOT continuity-constrained (standard hp-OC convention).
            for (std::size_t seg = 0; seg + 1 < num_seg; ++seg) {
                const std::size_t last_node_of_seg =
                    global_node_offsets[seg] + per_segment_node_count[seg] - 1;
                const std::size_t first_node_of_next = global_node_offsets[seg + 1];
                for (std::size_t state_i = 0; state_i < num_states; ++state_i) {
                    outputs.push_back(
                        z[layout.state_index(last_node_of_seg,   state_i)] -
                        z[layout.state_index(first_node_of_next, state_i)]);
                }
            }

            return outputs;
        };

        auto backend = std::make_unique<goss::ad::CppADCGBackend>(
            packed, layout.total_variables(), model_name);

        // --- Variable bounds: per-node state and control box constraints ---
        const std::size_t nv = layout.total_variables();
        std::vector<double> zl(nv, -kInf), zu(nv, kInf);
        for (std::size_t global_k = 0; global_k < total_nodes; ++global_k) {
            for (std::size_t state_i = 0; state_i < num_states; ++state_i) {
                const std::size_t idx = layout.state_index(global_k, state_i);
                zl[idx] = ocp.state_lower[state_i];
                zu[idx] = ocp.state_upper[state_i];
            }
            for (std::size_t ctrl_j = 0; ctrl_j < num_controls; ++ctrl_j) {
                const std::size_t idx = layout.control_index(global_k, ctrl_j);
                zl[idx] = ocp.control_lower[ctrl_j];
                zu[idx] = ocp.control_upper[ctrl_j];
            }
        }
        // Pin initial state at global node 0 (first node of segment 0).
        for (std::size_t state_i = 0; state_i < num_states; ++state_i) {
            if (state_i < ocp.initial_state_fixed.size() &&
                    ocp.initial_state_fixed[state_i] != 0.0) {
                const std::size_t idx = layout.state_index(0, state_i);
                zl[idx] = zu[idx] = ocp.initial_state[state_i];
            }
        }
        // Pin final state at global node total_nodes-1 (last node of last segment).
        for (std::size_t state_i = 0; state_i < num_states; ++state_i) {
            if (state_i < ocp.final_state_fixed.size() &&
                    ocp.final_state_fixed[state_i] != 0.0) {
                const std::size_t idx = layout.state_index(total_nodes - 1, state_i);
                zl[idx] = zu[idx] = ocp.final_state[state_i];
            }
        }

        // --- Constraint bounds: all equality (collocation defects + continuity) ---
        std::vector<double> gl(num_constraints_total, 0.0);
        std::vector<double> gu(num_constraints_total, 0.0);

        auto problem = std::make_unique<nlp::NLPProblem>(
            std::move(backend), std::move(zl), std::move(zu),
            std::move(gl), std::move(gu));
        std::vector<model::ParameterSpec> param_specs_hp;
        param_specs_hp.reserve(ocp.num_parameters);
        for (std::size_t i = 0; i < ocp.num_parameters; ++i)
            param_specs_hp.push_back(model::ParameterSpec{
                ocp.parameter_names[i], ocp.parameter_defaults[i],
                ocp.parameter_lower[i], ocp.parameter_upper[i]});
        return CompiledOcp{std::move(problem), layout, model::ParameterValidator(std::move(param_specs_hp))};
    }
};

}  // namespace goss::transcription
