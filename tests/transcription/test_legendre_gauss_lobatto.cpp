// tests/transcription/test_legendre_gauss_lobatto.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "goss/transcription/errors.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/transcription/legendre_gauss_lobatto.hpp"
#include "goss/transcription/mesh.hpp"
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

namespace {
// Solve exp-decay with n LGL nodes; return max nodal error vs analytic solution.
double lgl_max_error(std::size_t num_nodes) {
    const double x0 = 1.0, tf = 1.0;
    // num_intervals = num_nodes - 1 so that mesh.num_nodes() == num_nodes
    auto ocp = goss::transcription::test::make_exponential_decay(
        x0, tf, /*intervals=*/num_nodes - 1);
    auto compiled = goss::transcription::LegendreGaussLobatto::compile(
        ocp, "lgl_conv_" + std::to_string(num_nodes));
    goss::solver::IpoptSolver solver;
    solver.set_tolerance(1e-13);  // solver tolerance well below spectral accuracy
    std::vector<double> z0(compiled.problem->num_variables(), x0);
    auto result = solver.solve(*compiled.problem, z0);
    if (result.status != goss::solver::SolverStatus::Success) return 1e9;

    const auto& layout = compiled.layout;
    // Pre-compute LGL node times to compare at each node.
    std::vector<double> lgl_xi, lgl_weights;
    goss::transcription::lgl_nodes_and_weights(num_nodes, lgl_xi, lgl_weights);
    const double half_dur = 0.5 * tf;
    double max_err = 0.0;
    for (std::size_t k = 0; k < num_nodes; ++k) {
        const double t_k = 0.0 + half_dur * (lgl_xi[k] + 1.0);
        const double xk  = result.x[layout.state_index(k, 0)];
        const double exact = goss::transcription::test::exp_decay_solution(x0, t_k);
        max_err = std::max(max_err, std::abs(xk - exact));
    }
    return max_err;
}
}  // namespace

TEST(LegendreGaussLobatto, ConvergesSpectrally) {
    // Errors at n=3,5,7,9,11 LGL nodes on smooth exp(-t).
    const std::vector<std::size_t> node_counts = {3, 5, 7, 9, 11};
    std::vector<double> errors;
    for (std::size_t n : node_counts) errors.push_back(lgl_max_error(n));

    // Errors must decrease monotonically.
    for (std::size_t i = 0; i + 1 < errors.size(); ++i)
        ASSERT_LT(errors[i + 1], errors[i])
            << "LGL error must decrease as n increases";

    // Spectral convergence check: the convergence rate (ratio of log errors)
    // must be much steeper than O(h^4) = O(n^{-4}).
    // For O(h^4): halving h (doubling nodes) reduces error by 16x -> ratio ~4 per doubling.
    // For spectral: ratio should be >>4, e.g. >6 for n=3->5->7.
    // Check the ratio log(e[i])/log(e[i+1]) > 4 for consecutive pairs.
    // Use 3->7 (skip by 2) for a cleaner ratio.
    const double log_ratio_3_to_7 =
        std::log(errors[0] / errors[2]) / std::log(static_cast<double>(node_counts[2]) /
                                                   static_cast<double>(node_counts[0]));
    EXPECT_GT(log_ratio_3_to_7, 4.0)
        << "LGL spectral convergence should exceed O(h^4); "
           "observed log-ratio: " << log_ratio_3_to_7;

    // Hard accuracy check: 11 LGL nodes must achieve < 1e-10 on smooth exp(-t).
    EXPECT_LT(errors.back(), 1e-10)
        << "11 LGL nodes should achieve near-machine precision on smooth exp(-t)";
}

// Regression test (I-1): LegendreGaussLobatto::compile(ocp, NonUniformMesh, name)
// must throw TranscriptionError with a clear message, not produce a cryptic
// template compile error. LGL uses global single-interval collocation; calling it
// with a NonUniformMesh (the refine_and_solve interface) is a misuse.
TEST(LegendreGaussLobatto, RejectsNonUniformMesh) {
    auto ocp = goss::transcription::test::make_exponential_decay(1.0, 1.0, 4);
    goss::transcription::NonUniformMesh nu_mesh =
        goss::transcription::to_nonuniform(ocp.mesh);
    EXPECT_THROW(
        goss::transcription::LegendreGaussLobatto::compile(ocp, nu_mesh, "lgl_nu_reject"),
        goss::transcription::TranscriptionError);
}

TEST(LegendreGaussLobatto, SameNodeCountOutperformsHermiteSimpson) {
    // With n=9 LGL nodes vs 9 HS nodes (8 intervals), LGL should win on smooth problem.
    const std::size_t n_nodes = 9;
    const double x0 = 1.0, tf = 1.0;

    // LGL: 9 nodes
    double lgl_err = lgl_max_error(n_nodes);

    // Hermite-Simpson: 9 nodes = 8 intervals, error measured at node times.
    auto hs_ocp = goss::transcription::test::make_exponential_decay(x0, tf, n_nodes - 1);
    auto hs_compiled = goss::transcription::HermiteSimpson::compile(
        hs_ocp, "lgl_vs_hs_compare");
    goss::solver::IpoptSolver solver;
    solver.set_tolerance(1e-13);
    std::vector<double> z0(hs_compiled.problem->num_variables(), x0);
    auto hs_result = solver.solve(*hs_compiled.problem, z0);
    ASSERT_EQ(hs_result.status, goss::solver::SolverStatus::Success);
    const auto& hs_layout = hs_compiled.layout;
    const double h = tf / static_cast<double>(n_nodes - 1);
    double hs_err = 0.0;
    for (std::size_t k = 0; k < n_nodes; ++k) {
        double xk = hs_result.x[hs_layout.state_index(k, 0)];
        double exact = goss::transcription::test::exp_decay_solution(x0, k * h);
        hs_err = std::max(hs_err, std::abs(xk - exact));
    }

    EXPECT_LT(lgl_err, hs_err)
        << "LGL with " << n_nodes << " nodes should outperform HS with "
        << n_nodes << " nodes on smooth exp(-t)";
}
