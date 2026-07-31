// tests/transcription/test_trapezoidal.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "goss/transcription/trapezoidal.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "transcription/ocp_fixtures.hpp"

TEST(Trapezoidal, SolvesExponentialDecay) {
    const double x0 = 1.0, tf = 1.0;
    const std::size_t intervals = 40;
    auto ocp = goss::transcription::test::make_exponential_decay(x0, tf, intervals);
    auto compiled = goss::transcription::Trapezoidal::compile(ocp, "trap_expdecay");

    goss::solver::IpoptSolver solver;
    // initial guess: all nodes = x0 (a crude but feasible-ish start)
    std::vector<double> z0(compiled.problem->num_variables(), x0);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);

    // final node state should match x0*exp(-tf)
    const auto& layout = compiled.layout;
    std::size_t last = layout.num_nodes() - 1;
    double x_final = result.x[layout.state_index(last, 0)];
    EXPECT_NEAR(x_final, goss::transcription::test::exp_decay_solution(x0, tf), 1e-3);
}

TEST(Trapezoidal, PinsInitialState) {
    auto ocp = goss::transcription::test::make_exponential_decay(2.0, 1.0, 20);
    auto compiled = goss::transcription::Trapezoidal::compile(ocp, "trap_pin");
    // node 0 state 0 must be pinned to 2.0 via equal var bounds
    std::size_t idx = compiled.layout.state_index(0, 0);
    EXPECT_DOUBLE_EQ(compiled.problem->variable_lower_bounds()[idx], 2.0);
    EXPECT_DOUBLE_EQ(compiled.problem->variable_upper_bounds()[idx], 2.0);
}

// Regression: the boundary-pin guard must be per-index (i < size()), not just
// !empty(). When final_state_fixed[0] == 0.0, the final state must NOT be pinned
// (bounds stay wide), even though the vector is non-empty.
TEST(Trapezoidal, FinalStateFreeWhenNotFixed) {
    auto ocp = goss::transcription::test::make_exponential_decay(1.0, 1.0, 20);
    auto compiled = goss::transcription::Trapezoidal::compile(ocp, "trap_freefinal");
    std::size_t last = compiled.layout.num_nodes() - 1;
    std::size_t idx = compiled.layout.state_index(last, 0);
    // final_state_fixed[0] == 0.0 → not pinned → lower < upper
    EXPECT_LT(compiled.problem->variable_lower_bounds()[idx],
              compiled.problem->variable_upper_bounds()[idx]);
}
