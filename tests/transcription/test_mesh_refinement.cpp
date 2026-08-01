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
