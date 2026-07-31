// tests/transcription/test_trapezoidal.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
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

namespace {
// Solve exp-decay at a given interval count, return max nodal error vs analytic.
double trap_max_error(std::size_t intervals) {
    const double x0 = 1.0, tf = 1.0;
    auto ocp = goss::transcription::test::make_exponential_decay(x0, tf, intervals);
    auto compiled = goss::transcription::Trapezoidal::compile(
        ocp, "trap_conv_" + std::to_string(intervals));
    goss::solver::IpoptSolver solver;
    solver.set_tolerance(1e-10);
    std::vector<double> z0(compiled.problem->num_variables(), x0);
    auto result = solver.solve(*compiled.problem, z0);
    if (result.status != goss::solver::SolverStatus::Success) return 1e9;
    const auto& layout = compiled.layout;
    const double h = tf / static_cast<double>(intervals);
    double max_err = 0.0;
    for (std::size_t k = 0; k < layout.num_nodes(); ++k) {
        double xk = result.x[layout.state_index(k, 0)];
        double exact = goss::transcription::test::exp_decay_solution(x0, k * h);
        max_err = std::max(max_err, std::abs(xk - exact));
    }
    return max_err;
}
}  // namespace

TEST(Trapezoidal, ConvergesAtSecondOrder) {
    double e_coarse = trap_max_error(10);
    double e_fine = trap_max_error(20);
    double e_finer = trap_max_error(40);
    ASSERT_LT(e_finer, e_fine);
    ASSERT_LT(e_fine, e_coarse);
    // order p ≈ log2(e(h)/e(h/2)); trapezoidal is O(h^2) ⇒ p≈2.
    double order1 = std::log(e_coarse / e_fine) / std::log(2.0);
    double order2 = std::log(e_fine / e_finer) / std::log(2.0);
    EXPECT_GE(order1, 1.8) << "trapezoidal should be ~2nd order";
    EXPECT_GE(order2, 1.8) << "trapezoidal should be ~2nd order";
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
