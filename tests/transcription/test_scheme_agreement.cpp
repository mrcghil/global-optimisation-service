// tests/transcription/test_scheme_agreement.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "goss/transcription/trapezoidal.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/transcription/legendre_gauss_lobatto.hpp"
#include "goss/transcription/mesh.hpp"
#include "goss/transcription/mesh_refinement.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "transcription/ocp_fixtures.hpp"

namespace {
// Thin helper: asserts solve succeeded (ASSERT_EQ aborts on failure) then
// returns the terminal state value.  The assert must live in a void function
// so that gtest's fatal-failure mechanism (which does a bare `return;`) is
// legal.  Callers use ASSERT_NO_FATAL_FAILURE to propagate the abort upward.
void solve_and_check(goss::transcription::CompiledOcp& compiled, double guess,
                     double& out_final) {
    goss::solver::IpoptSolver solver;
    std::vector<double> z0(compiled.problem->num_variables(), guess);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    std::size_t last = compiled.layout.num_nodes() - 1;
    out_final = result.x[compiled.layout.state_index(last, 0)];
}
}  // namespace

TEST(SchemeAgreement, TrapezoidalAndHermiteSimpsonAgreeOnExpDecay) {
    const double x0 = 1.0, tf = 1.0;
    const std::size_t intervals = 80;  // fine enough that both are accurate
    auto ocp_t = goss::transcription::test::make_exponential_decay(x0, tf, intervals);
    auto ocp_h = goss::transcription::test::make_exponential_decay(x0, tf, intervals);
    auto ct = goss::transcription::Trapezoidal::compile(ocp_t, "agree_trap");
    auto ch = goss::transcription::HermiteSimpson::compile(ocp_h, "agree_hs");
    double xt = 0.0, xh = 0.0;
    ASSERT_NO_FATAL_FAILURE(solve_and_check(ct, x0, xt));
    ASSERT_NO_FATAL_FAILURE(solve_and_check(ch, x0, xh));
    double exact = goss::transcription::test::exp_decay_solution(x0, tf);
    EXPECT_NEAR(xt, exact, 1e-3);
    EXPECT_NEAR(xh, exact, 1e-3);
    EXPECT_NEAR(xt, xh, 2e-3);  // both converge to the same analytic solution
}

TEST(SchemeAgreement, TrapAndHSOnNonUniformMeshAgreeOnExpDecay) {
    const double x0 = 1.0;
    goss::transcription::NonUniformMesh nu_mesh;
    nu_mesh.node_times = {0.0, 0.1, 0.25, 0.5, 0.75, 1.0};
    auto ocp = goss::transcription::test::make_exponential_decay(x0, 1.0, 1);

    auto ct = goss::transcription::Trapezoidal::compile(ocp, nu_mesh, "agree_nu_trap");
    auto ch = goss::transcription::HermiteSimpson::compile(ocp, nu_mesh, "agree_nu_hs");

    goss::solver::IpoptSolver solver;
    auto solve_final = [&](goss::transcription::CompiledOcp& compiled) {
        std::vector<double> z0(compiled.problem->num_variables(), x0);
        auto result = solver.solve(*compiled.problem, z0);
        EXPECT_EQ(result.status, goss::solver::SolverStatus::Success);
        std::size_t last = compiled.layout.num_nodes() - 1;
        return result.x[compiled.layout.state_index(last, 0)];
    };

    double xt = solve_final(ct);
    double xh = solve_final(ch);
    double exact = goss::transcription::test::exp_decay_solution(x0, 1.0);
    EXPECT_NEAR(xt, exact, 1e-2);
    EXPECT_NEAR(xh, exact, 1e-4);   // HS is more accurate on the same mesh
    EXPECT_NEAR(xt, xh, 1e-2);
}

TEST(SchemeAgreement, AllThreeSchemesAgreeOnExpDecay) {
    const double x0 = 1.0, tf = 1.0;
    // Use 20 nodes so both HS (19 intervals) and LGL (20 nodes) are well-resolved.
    const std::size_t n_nodes_minus_1 = 19;
    auto ocp_t = goss::transcription::test::make_exponential_decay(x0, tf, n_nodes_minus_1);
    auto ocp_h = goss::transcription::test::make_exponential_decay(x0, tf, n_nodes_minus_1);
    auto ocp_l = goss::transcription::test::make_exponential_decay(x0, tf, n_nodes_minus_1);

    auto ct  = goss::transcription::Trapezoidal::compile(ocp_t, "agree3_trap");
    auto ch  = goss::transcription::HermiteSimpson::compile(ocp_h, "agree3_hs");
    auto cl  = goss::transcription::LegendreGaussLobatto::compile(ocp_l, "agree3_lgl");

    goss::solver::IpoptSolver solver;
    solver.set_tolerance(1e-11);
    auto solve_final_state = [&](goss::transcription::CompiledOcp& compiled) {
        std::vector<double> z0(compiled.problem->num_variables(), x0);
        auto result = solver.solve(*compiled.problem, z0);
        EXPECT_EQ(result.status, goss::solver::SolverStatus::Success);
        std::size_t last = compiled.layout.num_nodes() - 1;
        return result.x[compiled.layout.state_index(last, 0)];
    };

    double xt = solve_final_state(ct);
    double xh = solve_final_state(ch);
    double xl = solve_final_state(cl);
    double exact = goss::transcription::test::exp_decay_solution(x0, tf);

    EXPECT_NEAR(xt, exact, 1e-3);
    EXPECT_NEAR(xh, exact, 1e-6);
    EXPECT_NEAR(xl, exact, 1e-10);  // LGL at 20 nodes beats both on smooth problem
    EXPECT_NEAR(xt, xh, 1e-3);
    EXPECT_NEAR(xh, xl, 1e-5);
}
