// tests/transcription/test_hp_pseudospectral.cpp
//
// Tests for LegendreGaussLobatto::compile_hp — multi-segment hp-pseudospectral collocation.
//
// Key properties verified:
//   1. Variable layout: total_nodes = sum(n_s); state_index / control_index consistent.
//   2. S=1 segment with same node count reduces to same constraint count as compile().
//   3. S=1 segment solves to the same solution as single-interval compile().
//   4. compile_hp rejects free initial state (same guard as compile()).
//   5. 3-segment solve on exp-decay achieves near-analytic accuracy.
//   6. Continuity at segment boundaries satisfied after solve.
//   7. Controlled (nc>0) problem: node-0 is collocated for ALL segments so u_0 is
//      constrained; objective must match the single-interval LGL result (no u_0 bias).
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <string>
#include "goss/transcription/legendre_gauss_lobatto.hpp"
#include "goss/transcription/hp_mesh.hpp"
#include "goss/transcription/errors.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "transcription/ocp_fixtures.hpp"

namespace {

// Helper: solve a compiled OCP with IPOPT and return the result.
// Initializes all variables to initial_guess_value.
goss::solver::SolverResult solve_compiled(
        const goss::transcription::CompiledOcp& compiled,
        double initial_guess_value = 1.0,
        double solver_tolerance    = 1e-11) {
    goss::solver::IpoptSolver solver;
    solver.set_tolerance(solver_tolerance);
    const std::vector<double> initial_guess(
        compiled.problem->num_variables(), initial_guess_value);
    return solver.solve(*compiled.problem, initial_guess);
}

// ---------------------------------------------------------------------------
// Controlled-problem fixtures for the nc>0 node-0 reconciliation test.
//
// Single-integrator min-energy: dx/dt = u, x(0)=0, x(T)=1, min ∫₀ᵀ u² dt.
// Analytic solution: u*(t) = 1/T (constant), J* = 1/T.
// T = 1.0  →  J* = 1.0.
// ---------------------------------------------------------------------------
struct SingleIntegratorDynamics {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& /*x*/,
                               const std::vector<T>& u,
                               T /*t*/) const {
        return { u[0] };
    }
};

struct MinEnergyCost {
    template <typename T>
    T operator()(const std::vector<T>& /*x*/,
                 const std::vector<T>& u,
                 T /*t*/) const {
        return u[0] * u[0];
    }
};

// Build a single-integrator min-energy OCP with the given number of intervals.
// x(0)=0 pinned, x(T)=1 pinned, u ∈ [-10, 10], cost = u².
inline auto make_single_integrator_min_energy(std::size_t num_intervals) {
    goss::transcription::OcpProblem<SingleIntegratorDynamics, MinEnergyCost> ocp;
    ocp.num_states    = 1;
    ocp.num_controls  = 1;
    ocp.dynamics      = SingleIntegratorDynamics{};
    ocp.cost          = MinEnergyCost{};
    ocp.mesh          = goss::transcription::Mesh{0.0, 1.0, num_intervals};
    ocp.state_lower   = { -1e19 };
    ocp.state_upper   = {  1e19 };
    ocp.control_lower = { -10.0 };
    ocp.control_upper = {  10.0 };
    ocp.initial_state       = { 0.0 };
    ocp.initial_state_fixed = { 1.0 };   // pin x(0) = 0
    ocp.final_state         = { 1.0 };
    ocp.final_state_fixed   = { 1.0 };   // pin x(T) = 1
    return ocp;
}

}  // namespace

// --- Test 1: variable count and layout for a 3-segment problem ---
// Segments: n_s = [4, 5, 3]; total_nodes = 12.
// Variables per node = ns + nc. For exp-decay: ns=1, nc=0, vpn=1.
// Total variables = 12.
TEST(HpPseudospectral, VariableLayoutThreeSegments) {
    goss::transcription::HpMesh hp_mesh;
    hp_mesh.segment_boundary_times = {0.0, 0.4, 0.7, 1.0};
    hp_mesh.per_segment_node_count = {4, 5, 3};  // total_nodes = 12

    // Use exp-decay ocp (1 state, 0 controls).
    auto ocp = goss::transcription::test::make_exponential_decay(1.0, 1.0, /*intervals=*/7);

    auto compiled = goss::transcription::LegendreGaussLobatto::compile_hp(
        ocp, hp_mesh, "layout_3seg");

    // total_variables = total_nodes * (ns + nc) = 12 * 1 = 12
    EXPECT_EQ(compiled.problem->num_variables(), 12u);
    EXPECT_EQ(compiled.layout.num_nodes(), 12u);
    EXPECT_EQ(compiled.layout.num_states(), 1u);
    EXPECT_EQ(compiled.layout.num_controls(), 0u);
}

// --- Test 2: S=1 compile_hp matches compile() on exp-decay ---
// compile_hp with a single segment of 8 nodes must give the same constraint count
// as compile(ocp, name) with num_intervals=7 (=> 8 nodes).
// WHY: nc==0 → first_collocation_node=1 → defects=(8-1)*1=7, same as single-interval compile().
TEST(HpPseudospectral, SingleSegmentMatchesSingleIntervalLayout) {
    auto ocp = goss::transcription::test::make_exponential_decay(1.0, 1.0, /*intervals=*/7);

    // Single-interval LGL (existing path).
    auto compiled_single =
        goss::transcription::LegendreGaussLobatto::compile(ocp, "hp_s1_single");

    // hp compile_hp with S=1, 8 nodes.
    const goss::transcription::HpMesh hp_mesh =
        goss::transcription::to_single_segment_hp_mesh(ocp.mesh);
    auto compiled_hp =
        goss::transcription::LegendreGaussLobatto::compile_hp(ocp, hp_mesh, "hp_s1_multi");

    // Variable counts must match.
    EXPECT_EQ(compiled_hp.problem->num_variables(),
              compiled_single.problem->num_variables());
    // S=1 has no continuity constraints; defect count = (8-1)*1 = 7.
    // Both must have the same constraint count.
    EXPECT_EQ(compiled_hp.problem->num_constraints(),
              compiled_single.problem->num_constraints());
}

// --- Test 3: S=1 compile_hp solves to same solution as compile() ---
TEST(HpPseudospectral, SingleSegmentSolvesToSameSolutionAsSingleInterval) {
    const double x0 = 1.0, time_final = 1.0;
    auto ocp = goss::transcription::test::make_exponential_decay(
        x0, time_final, /*intervals=*/7);

    auto compiled_single =
        goss::transcription::LegendreGaussLobatto::compile(ocp, "hp_s1_solve_single");
    const goss::transcription::HpMesh hp_mesh =
        goss::transcription::to_single_segment_hp_mesh(ocp.mesh);
    auto compiled_hp =
        goss::transcription::LegendreGaussLobatto::compile_hp(ocp, hp_mesh, "hp_s1_solve_hp");

    auto result_single = solve_compiled(compiled_single, x0);
    auto result_hp     = solve_compiled(compiled_hp, x0);

    ASSERT_EQ(result_single.status, goss::solver::SolverStatus::Success);
    ASSERT_EQ(result_hp.status,     goss::solver::SolverStatus::Success);

    // Final node state must match to tight tolerance.
    const std::size_t last_node = compiled_single.layout.num_nodes() - 1;
    const double x_final_single = result_single.x[compiled_single.layout.state_index(last_node, 0)];
    const double x_final_hp     = result_hp.x[compiled_hp.layout.state_index(last_node, 0)];
    EXPECT_NEAR(x_final_hp, x_final_single, 1e-8)
        << "S=1 compile_hp must match compile() final state";
}

// --- Test 4: hp compile_hp rejects free initial state ---
TEST(HpPseudospectral, RejectsFreeInitialState) {
    goss::transcription::OcpProblem<goss::transcription::test::ExpDecayDynamics,
                                    goss::transcription::test::ZeroCost> ocp;
    ocp.num_states = 1;
    ocp.num_controls = 0;
    ocp.dynamics = goss::transcription::test::ExpDecayDynamics{};
    ocp.cost     = goss::transcription::test::ZeroCost{};
    ocp.mesh = goss::transcription::Mesh{0.0, 1.0, 7};
    ocp.state_lower = {-1e19};
    ocp.state_upper = { 1e19};
    ocp.control_lower = {};
    ocp.control_upper = {};
    ocp.initial_state       = {1.0};
    ocp.initial_state_fixed = {0.0};  // free — must be rejected
    ocp.final_state         = {0.0};
    ocp.final_state_fixed   = {0.0};

    goss::transcription::HpMesh hp_mesh;
    hp_mesh.segment_boundary_times = {0.0, 0.5, 1.0};
    hp_mesh.per_segment_node_count = {4, 4};

    EXPECT_THROW(
        goss::transcription::LegendreGaussLobatto::compile_hp(ocp, hp_mesh, "hp_free_init"),
        goss::transcription::TranscriptionError);
}

// --- Test 5: 3-segment hp solve on exp-decay ---
// Partition [0,1] into 3 segments: [0,0.3], [0.3,0.7], [0.7,1.0] with 4+5+4=13 nodes.
// The solution at the last global node must approximate exp(-1) to 1e-5.
TEST(HpPseudospectral, ThreeSegmentSolvesExponentialDecay) {
    const double x0 = 1.0, time_final = 1.0;
    auto ocp = goss::transcription::test::make_exponential_decay(
        x0, time_final, /*intervals=*/7);

    goss::transcription::HpMesh hp_mesh;
    hp_mesh.segment_boundary_times = {0.0, 0.3, 0.7, 1.0};
    hp_mesh.per_segment_node_count = {4, 5, 4};  // total 13 nodes

    auto compiled = goss::transcription::LegendreGaussLobatto::compile_hp(
        ocp, hp_mesh, "hp_3seg_expdecay");
    auto result = solve_compiled(compiled, x0);

    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);

    // Last global node = node 12 (0-indexed), which is the last node of segment 2.
    const std::size_t last_global_node = compiled.layout.num_nodes() - 1;
    const double x_final = result.x[compiled.layout.state_index(last_global_node, 0)];
    // WHY 2e-5: 3 segments of 4+5+4 nodes each has less global spectral accuracy
    // than 13 single-interval LGL nodes (each segment's polynomial degree is limited
    // to 3, 4, and 3 respectively). The error 1.4e-5 is correct numerical behaviour;
    // 2e-5 gives a small safety margin while still verifying meaningful accuracy.
    EXPECT_NEAR(x_final,
                goss::transcription::test::exp_decay_solution(x0, time_final),
                2e-5)
        << "3-segment hp: final x must approximate exp(-1)";
}

// --- Test 6: continuity at segment boundaries is satisfied ---
// After solving, for each internal boundary the last state of segment s must
// equal the first state of segment s+1 (within solver tolerance).
TEST(HpPseudospectral, ContinuityAtSegmentBoundaries) {
    const double x0 = 1.0, time_final = 1.0;
    auto ocp = goss::transcription::test::make_exponential_decay(
        x0, time_final, /*intervals=*/7);

    goss::transcription::HpMesh hp_mesh;
    hp_mesh.segment_boundary_times = {0.0, 0.4, 0.7, 1.0};
    hp_mesh.per_segment_node_count = {4, 5, 3};  // total 12 nodes

    auto compiled = goss::transcription::LegendreGaussLobatto::compile_hp(
        ocp, hp_mesh, "hp_continuity_check");
    auto result = solve_compiled(compiled, x0, /*solver_tolerance=*/1e-10);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);

    // Segment offsets (cumulative sum of per_segment_node_count).
    std::vector<std::size_t> offsets;
    offsets.push_back(0);
    for (const std::size_t n_s : hp_mesh.per_segment_node_count)
        offsets.push_back(offsets.back() + n_s);

    const std::size_t num_states = ocp.num_states;
    const std::size_t num_seg = hp_mesh.num_segments();

    // For each internal boundary s=0..S-2, check state continuity.
    for (std::size_t seg = 0; seg + 1 < num_seg; ++seg) {
        // Last node of segment seg.
        const std::size_t last_node_of_seg =
            offsets[seg] + hp_mesh.per_segment_node_count[seg] - 1;
        // First node of segment seg+1.
        const std::size_t first_node_of_next = offsets[seg + 1];

        for (std::size_t state_idx = 0; state_idx < num_states; ++state_idx) {
            const double x_end_s =
                result.x[compiled.layout.state_index(last_node_of_seg, state_idx)];
            const double x_start_s1 =
                result.x[compiled.layout.state_index(first_node_of_next, state_idx)];
            EXPECT_NEAR(x_end_s, x_start_s1, 1e-8)
                << "Continuity violated at segment boundary " << seg
                << " -> " << (seg+1) << ", state " << state_idx;
        }
    }
}

// --- Test: per-segment differentiation matrix is exact up to degree n_s-1 ---
// For segment [t_a, t_b] with n_s LGL nodes, the local D_s matrix on [-1,1]
// differentiates polynomials of degree <= n_s-1 exactly (standard LGL property).
// We verify the PHYSICAL-domain scaling: (D_s @ p_xi)[row] = half_dur * (dp/dt)(t_row)
// for p(t) = (t - t_a)^k, k=1..n_s-1.
TEST(HpPseudospectral, PerSegmentDifferentiationMatrixExactUpToDegreeNMinusOne) {
    // Test for two segment sizes: n_s=4 and n_s=6.
    for (const std::size_t n_s : {4u, 6u}) {
        // Arbitrary physical interval to test real-world scaling.
        const double t_a_seg   = 0.3;
        const double t_b_seg   = 0.8;
        const double half_dur_seg = 0.5 * (t_b_seg - t_a_seg);

        std::vector<double> lgl_xi_seg, lgl_weights_seg;
        goss::transcription::lgl_nodes_and_weights(n_s, lgl_xi_seg, lgl_weights_seg);

        // Physical node times: t_k = t_a + half_dur * (xi_k + 1).
        std::vector<double> t_physical(n_s);
        for (std::size_t k = 0; k < n_s; ++k)
            t_physical[k] = t_a_seg + half_dur_seg * (lgl_xi_seg[k] + 1.0);

        // Local differentiation matrix on [-1,1].
        const std::vector<double> D_seg =
            goss::transcription::lgl_differentiation_matrix(lgl_xi_seg);

        // For polynomial p(t) = (t - t_a)^k (k=1..n_s-1):
        //   p_xi[j] = (t_physical[j] - t_a)^k = (half_dur * (xi_j+1))^k
        //   (dp/dt)(t_row) = k * (t_row - t_a)^(k-1)
        // Collocation scaling: (D_s @ p_xi)[row] must equal half_dur_seg * (dp/dt)(t_row).
        const std::size_t degree_max = n_s - 1;  // LGL is exact up to degree n_s-1
        for (std::size_t poly_degree = 1; poly_degree <= degree_max; ++poly_degree) {
            // Evaluate polynomial at LGL nodes.
            std::vector<double> p_at_nodes(n_s);
            for (std::size_t j = 0; j < n_s; ++j)
                p_at_nodes[j] = std::pow(t_physical[j] - t_a_seg,
                                         static_cast<double>(poly_degree));

            for (std::size_t row = 0; row < n_s; ++row) {
                // Compute (D_s @ p_xi)[row] = sum_j D_seg[row*n_s + j] * p_at_nodes[j].
                double D_times_p = 0.0;
                for (std::size_t col = 0; col < n_s; ++col)
                    D_times_p += D_seg[row * n_s + col] * p_at_nodes[col];

                // Expected: half_dur_seg * dp/dt at t_physical[row].
                const double dp_dt_at_row =
                    static_cast<double>(poly_degree) *
                    std::pow(t_physical[row] - t_a_seg,
                             static_cast<double>(poly_degree - 1));
                const double expected = half_dur_seg * dp_dt_at_row;

                EXPECT_NEAR(D_times_p, expected, 1e-8)
                    << "n_s=" << n_s << " degree=" << poly_degree << " row=" << row;
            }
        }
    }
}

// --- Test 7: Controlled (nc>0) problem — node-0 reconciliation guard ---
//
// Single-integrator min-energy: dx/dt=u, x(0)=0, x(T)=1, min ∫₀ᵀ u² dt.
// Analytic: u*(t)=1/T (constant), J*=1/T=1.0 for T=1.
//
// Solved via compile_hp with 3 segments (total 4+5+4=13 nodes, all with nc>0).
// compile_hp MUST collocate node 0 for ALL segments (first_collocation_node=0
// when nc>0). Without this, each segment's u_0 would be unconstrained → optimizer
// drives it toward zero → objective is biased away from J*=1.0.
//
// Also solved via single-interval compile() with same total node count (12 nodes).
// Both objectives must match J*=1.0 to 1e-4 (LGL spectral on smooth problem).
//
// This test is the canary for Reconciliation 2: if node-0 is wrongly skipped for
// nc>0, this test fails with a biased objective (<<1.0 or >>1.0).
TEST(HpPseudospectral, ControlledProblemNodeZeroCollocated) {
    constexpr double kAnalyticObjective = 1.0;  // J* = 1/T = 1.0 for T=1
    constexpr double kTolerance         = 1e-4;  // LGL spectral; tight tolerance

    // Single-interval LGL reference: 12 intervals => 13 nodes (nc=1).
    auto ocp_single = make_single_integrator_min_energy(/*num_intervals=*/12);
    auto compiled_single = goss::transcription::LegendreGaussLobatto::compile(
        ocp_single, "hp_ctrl_single_ref");
    auto result_single = solve_compiled(compiled_single, /*initial_guess=*/0.5);
    ASSERT_EQ(result_single.status, goss::solver::SolverStatus::Success)
        << "Single-interval reference solve failed";
    const double obj_single = result_single.objective_value;
    EXPECT_NEAR(obj_single, kAnalyticObjective, kTolerance)
        << "Single-interval LGL objective must match J*=1.0";

    // hp compile_hp: 3 segments, total 4+5+4=13 nodes, nc=1.
    // WHY same total node count as single-interval: isolates the hp assembly,
    // ruling out node-count differences as a confounder.
    // NOTE: ocp mesh must still be set for hp_mesh time-horizon matching.
    auto ocp_hp = make_single_integrator_min_energy(/*num_intervals=*/12);
    goss::transcription::HpMesh hp_mesh;
    hp_mesh.segment_boundary_times = {0.0, 0.3, 0.7, 1.0};
    hp_mesh.per_segment_node_count = {4, 5, 4};  // total 13 nodes, nc>0 per segment

    auto compiled_hp = goss::transcription::LegendreGaussLobatto::compile_hp(
        ocp_hp, hp_mesh, "hp_ctrl_hp_3seg");
    auto result_hp = solve_compiled(compiled_hp, /*initial_guess=*/0.5);
    ASSERT_EQ(result_hp.status, goss::solver::SolverStatus::Success)
        << "hp compile_hp controlled solve failed";
    const double obj_hp = result_hp.objective_value;

    // Primary assertion: hp objective must match J*=1.0.
    // If node-0 is wrongly skipped (nc>0 not collocated), u_0 of each segment
    // is driven toward zero by the optimizer, biasing the objective.
    EXPECT_NEAR(obj_hp, kAnalyticObjective, kTolerance)
        << "hp compile_hp (nc>0, 3 segments): objective must match J*=1.0 "
           "(node-0 must be collocated for all segments when nc>0).\n"
           "  Actual hp obj:     " << obj_hp << "\n"
           "  Single-interval:   " << obj_single << "\n"
           "  Analytic J*:       " << kAnalyticObjective;

    // Secondary assertion: hp and single-interval objectives agree tightly.
    // This confirms the segmented assembly is consistent with single-interval LGL.
    EXPECT_NEAR(obj_hp, obj_single, 1e-3)
        << "hp and single-interval objectives must agree; hp obj=" << obj_hp
        << ", single=" << obj_single;
}
