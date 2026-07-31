// tests/transcription/test_hermite_simpson.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "transcription/ocp_fixtures.hpp"

TEST(HermiteSimpson, SolvesExponentialDecay) {
    const double x0 = 1.0, tf = 1.0;
    auto ocp = goss::transcription::test::make_exponential_decay(x0, tf, 20);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "hs_expdecay");
    goss::solver::IpoptSolver solver;
    std::vector<double> z0(compiled.problem->num_variables(), x0);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    std::size_t last = compiled.layout.num_nodes() - 1;
    double x_final = result.x[compiled.layout.state_index(last, 0)];
    EXPECT_NEAR(x_final, goss::transcription::test::exp_decay_solution(x0, tf), 1e-5);
}

TEST(HermiteSimpson, SolvesHarmonicOscillator) {
    // x0(0)=1, x1(0)=0 -> x0(t)=cos t. Check x0(tf).
    const double tf = 1.0;
    auto ocp = goss::transcription::test::make_harmonic(1.0, 0.0, tf, 20);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "hs_harmonic");
    goss::solver::IpoptSolver solver;
    std::vector<double> z0(compiled.problem->num_variables(), 0.5);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    std::size_t last = compiled.layout.num_nodes() - 1;
    double x0_final = result.x[compiled.layout.state_index(last, 0)];
    EXPECT_NEAR(x0_final, goss::transcription::test::harmonic_x0_solution(1.0, 0.0, tf), 1e-4);
}
