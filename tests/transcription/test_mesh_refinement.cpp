// tests/transcription/test_mesh_refinement.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "goss/transcription/mesh_refinement.hpp"
#include "goss/transcription/trapezoidal.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "transcription/ocp_fixtures.hpp"

TEST(MeshRefinement, ErrorEstimatorReturnsOneEntryPerInterval) {
    const double x0 = 1.0;
    auto ocp = goss::transcription::test::make_exponential_decay(x0, 1.0, 10);
    auto nu_mesh = goss::transcription::to_nonuniform(ocp.mesh);
    auto compiled = goss::transcription::Trapezoidal::compile(ocp, nu_mesh, "refine_err_sz");
    goss::solver::IpoptSolver solver;
    std::vector<double> z0(compiled.problem->num_variables(), x0);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    auto errors = goss::transcription::estimate_interval_errors(ocp, nu_mesh, result, compiled.layout);
    ASSERT_EQ(errors.size(), nu_mesh.num_intervals());
    for (double err : errors) EXPECT_GE(err, 0.0);
}

TEST(MeshRefinement, ErrorEstimatorIsLargerOnCoarseFastDecay) {
    // Fast decay (dx/dt = -10x): early intervals have large truncation error on a coarse mesh.
    const double x0 = 1.0;
    auto ocp = goss::transcription::test::make_fast_decay(x0, 1.0, 4);
    goss::transcription::NonUniformMesh uniform_coarse = goss::transcription::to_nonuniform(ocp.mesh);
    auto compiled = goss::transcription::Trapezoidal::compile(
        ocp, uniform_coarse, "refine_err_fast");
    goss::solver::IpoptSolver solver;
    std::vector<double> z0(compiled.problem->num_variables(), x0);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    auto errors = goss::transcription::estimate_interval_errors(
        ocp, uniform_coarse, result, compiled.layout);
    ASSERT_EQ(errors.size(), 4u);
    // The first interval [0, 0.25] encompasses the fast decay; it should carry
    // more error than the last interval [0.75, 1.0] where the solution is ~flat.
    EXPECT_GT(errors[0], errors[3])
        << "Early interval should have larger error on fast-decay problem";
}

// Localization invariant: corrupting a single interior node m inflates ONLY the
// two adjacent intervals (m-1 and m) and leaves distant intervals unaffected.
// This guarantees that Task 5's refinement loop targets the right intervals.
//
// Setup: 8-interval exp-decay problem (9 nodes). Corrupt node m=4 by +5.0.
// Expected: errors[3] and errors[4] >> 1.0 ; errors[0] << 1e-2.
TEST(MeshRefinement, ErrorEstimatorLocalizesCorruptedNode) {
    const double x0 = 1.0;
    // Use exp-decay: well-conditioned, analytic solution, 8 intervals so node 4
    // is strictly interior with non-adjacent interval 0 far away.
    auto ocp = goss::transcription::test::make_exponential_decay(x0, /*tf=*/1.0, /*intervals=*/8);
    goss::transcription::NonUniformMesh nu_mesh = goss::transcription::to_nonuniform(ocp.mesh);
    auto compiled = goss::transcription::Trapezoidal::compile(ocp, nu_mesh, "refine_err_loc");

    goss::solver::IpoptSolver solver;
    std::vector<double> z0(compiled.problem->num_variables(), x0);
    auto clean_result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(clean_result.status, goss::solver::SolverStatus::Success);

    // Verify baseline: all intervals have small error on the clean solution.
    auto clean_errors = goss::transcription::estimate_interval_errors(
        ocp, nu_mesh, clean_result, compiled.layout);
    ASSERT_EQ(clean_errors.size(), 8u);
    for (std::size_t k = 0; k < clean_errors.size(); ++k)
        EXPECT_LT(clean_errors[k], 1e-2)
            << "Baseline: interval " << k << " should have small error before corruption";

    // Corrupt node m=4: add a large offset (+5.0) to its state.
    const std::size_t m = 4;
    const double corruption = 5.0;
    auto corrupted_result = clean_result;  // copy
    corrupted_result.x[compiled.layout.state_index(m, 0)] += corruption;

    // Re-estimate errors on the corrupted result.
    auto errors = goss::transcription::estimate_interval_errors(
        ocp, nu_mesh, corrupted_result, compiled.layout);
    ASSERT_EQ(errors.size(), 8u);

    // Interval m-1 = 3: RK4 starts at clean node 3, but its "solved end" is the corrupted
    // node 4 — so the comparison blows up.
    EXPECT_GT(errors[m - 1], 1.0)
        << "Interval m-1=" << (m - 1) << " should be large (corrupted end node)";

    // Interval m = 4: RK4 starts from the corrupted node 4 and integrates forward
    // to clean node 5 — the corrupted IC propagates the error to the RK4 result.
    EXPECT_GT(errors[m], 1.0)
        << "Interval m=" << m << " should be large (corrupted start node)";

    // Interval 0 is non-adjacent to node 4 (separated by 4 hops).
    // Because each interval is integrated from its OWN start node, the corruption
    // of node 4 cannot affect intervals that neither start nor end at node 4.
    EXPECT_LT(errors[0], 1e-2)
        << "Non-adjacent interval 0 should remain small (localization guarantee)";
}
