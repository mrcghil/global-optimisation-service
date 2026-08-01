// tests/transcription/test_hermite_simpson.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/transcription/mesh.hpp"
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

TEST(HermiteSimpson, PinsInitialState) {
    auto ocp = goss::transcription::test::make_exponential_decay(2.0, 1.0, 20);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "hs_pin");
    std::size_t idx = compiled.layout.state_index(0, 0);
    EXPECT_DOUBLE_EQ(compiled.problem->variable_lower_bounds()[idx], 2.0);
    EXPECT_DOUBLE_EQ(compiled.problem->variable_upper_bounds()[idx], 2.0);
}

TEST(HermiteSimpson, FinalStateFreeWhenNotFixed) {
    auto ocp = goss::transcription::test::make_exponential_decay(1.0, 1.0, 20);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "hs_freefinal");
    std::size_t last = compiled.layout.num_nodes() - 1;
    std::size_t idx = compiled.layout.state_index(last, 0);
    EXPECT_LT(compiled.problem->variable_lower_bounds()[idx],
              compiled.problem->variable_upper_bounds()[idx]);
}

namespace {
double hs_max_error(std::size_t intervals) {
    const double x0 = 1.0, tf = 1.0;
    auto ocp = goss::transcription::test::make_exponential_decay(x0, tf, intervals);
    auto compiled = goss::transcription::HermiteSimpson::compile(
        ocp, "hs_conv_" + std::to_string(intervals));
    goss::solver::IpoptSolver solver;
    solver.set_tolerance(1e-11);  // discretization error must dominate, not solver tol
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

TEST(HermiteSimpson, ConvergesAtFourthOrder) {
    // Use coarse meshes so O(h^4) error stays above solver tolerance floor.
    double e1 = hs_max_error(5);
    double e2 = hs_max_error(10);
    double e3 = hs_max_error(20);
    ASSERT_LT(e2, e1);
    ASSERT_LT(e3, e2);
    double order1 = std::log(e1 / e2) / std::log(2.0);
    double order2 = std::log(e2 / e3) / std::log(2.0);
    // HS is O(h^4). Allow slack (3.5) for solver-tolerance contamination at fine meshes.
    EXPECT_GE(order1, 3.5) << "Hermite-Simpson should be ~4th order";
    EXPECT_GE(order2, 3.5) << "Hermite-Simpson should be ~4th order";
}

TEST(HermiteSimpson, NonUniformMeshSolvesExponentialDecay) {
    const double x0 = 1.0;
    auto ocp = goss::transcription::test::make_exponential_decay(x0, 1.0, 1);
    goss::transcription::NonUniformMesh nu_mesh;
    nu_mesh.node_times = {0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0};
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, nu_mesh, "hs_nu_expdecay");
    goss::solver::IpoptSolver solver;
    solver.set_tolerance(1e-10);
    std::vector<double> z0(compiled.problem->num_variables(), x0);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    std::size_t last = compiled.layout.num_nodes() - 1;
    double x_final = result.x[compiled.layout.state_index(last, 0)];
    EXPECT_NEAR(x_final, goss::transcription::test::exp_decay_solution(x0, 1.0), 1e-4);
}

TEST(HermiteSimpson, UniformOverloadStillPassesAfterRefactor) {
    const double x0 = 1.0;
    auto ocp = goss::transcription::test::make_exponential_decay(x0, 1.0, 20);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "hs_uniform_regression");
    goss::solver::IpoptSolver solver;
    solver.set_tolerance(1e-11);
    std::vector<double> z0(compiled.problem->num_variables(), x0);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    std::size_t last = compiled.layout.num_nodes() - 1;
    double x_final = result.x[compiled.layout.state_index(last, 0)];
    EXPECT_NEAR(x_final, goss::transcription::test::exp_decay_solution(x0, 1.0), 1e-5);
}
