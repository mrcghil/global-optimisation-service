// tests/transcription/test_legendre_gauss_lobatto.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "goss/transcription/errors.hpp"
#include "goss/transcription/legendre_gauss_lobatto.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "transcription/ocp_fixtures.hpp"

TEST(LegendreGaussLobatto, SolvesExponentialDecayWithFewNodes) {
    // LGL is spectrally accurate — 8 nodes should give excellent accuracy.
    const double x0 = 1.0, tf = 1.0;
    // num_intervals = 7 => num_nodes = 8 LGL nodes
    auto ocp = goss::transcription::test::make_exponential_decay(x0, tf, /*intervals=*/7);
    auto compiled = goss::transcription::LegendreGaussLobatto::compile(ocp, "lgl_expdecay");

    goss::solver::IpoptSolver solver;
    solver.set_tolerance(1e-11);
    std::vector<double> z0(compiled.problem->num_variables(), x0);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);

    std::size_t last = compiled.layout.num_nodes() - 1;
    double x_final = result.x[compiled.layout.state_index(last, 0)];
    // 8 LGL nodes should give much better than 1e-8 accuracy on smooth exp(-t).
    EXPECT_NEAR(x_final, goss::transcription::test::exp_decay_solution(x0, tf), 1e-8);
}

TEST(LegendreGaussLobatto, PinsInitialState) {
    auto ocp = goss::transcription::test::make_exponential_decay(2.0, 1.0, 7);
    auto compiled = goss::transcription::LegendreGaussLobatto::compile(ocp, "lgl_pin");
    std::size_t idx = compiled.layout.state_index(0, 0);
    EXPECT_DOUBLE_EQ(compiled.problem->variable_lower_bounds()[idx], 2.0);
    EXPECT_DOUBLE_EQ(compiled.problem->variable_upper_bounds()[idx], 2.0);
}

// Regression test: compile() must throw TranscriptionError when any initial state
// component is not pinned (initial_state_fixed[i] == 0). The omitted node-0 defect
// makes the system underdetermined for free initial states — reject early to prevent
// IPOPT from silently returning an arbitrary x(0).
TEST(LegendreGaussLobatto, RejectsFreeInitialState) {
    // Build an OCP identical to make_exponential_decay but with initial_state_fixed[0] = 0
    // (free initial state) — compile() must throw before building the NLP.
    goss::transcription::OcpProblem<goss::transcription::test::ExpDecayDynamics,
                                    goss::transcription::test::ZeroCost>
        ocp;
    ocp.num_states = 1;
    ocp.num_controls = 0;
    ocp.dynamics = goss::transcription::test::ExpDecayDynamics{};
    ocp.cost = goss::transcription::test::ZeroCost{};
    ocp.mesh = goss::transcription::Mesh{0.0, 1.0, /*intervals=*/7};
    ocp.state_lower = { -1e19 };
    ocp.state_upper = { 1e19 };
    ocp.control_lower = {};
    ocp.control_upper = {};
    ocp.initial_state = { 1.0 };
    ocp.initial_state_fixed = { 0.0 };   // FREE initial state — must be rejected
    ocp.final_state = { 0.0 };
    ocp.final_state_fixed = { 0.0 };

    EXPECT_THROW(
        goss::transcription::LegendreGaussLobatto::compile(ocp, "lgl_free_init"),
        goss::transcription::TranscriptionError);
}

TEST(LegendreGaussLobatto, SolvesHarmonicOscillator) {
    const double tf = 1.0;
    // 10 LGL nodes over [0,1] for x' = [x1, -x0].
    auto ocp = goss::transcription::test::make_harmonic(1.0, 0.0, tf, /*intervals=*/9);
    auto compiled = goss::transcription::LegendreGaussLobatto::compile(ocp, "lgl_harmonic");
    goss::solver::IpoptSolver solver;
    solver.set_tolerance(1e-11);
    std::vector<double> z0(compiled.problem->num_variables(), 0.5);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    std::size_t last = compiled.layout.num_nodes() - 1;
    double x0_final = result.x[compiled.layout.state_index(last, 0)];
    EXPECT_NEAR(x0_final, goss::transcription::test::harmonic_x0_solution(1.0, 0.0, tf), 1e-7);
}
