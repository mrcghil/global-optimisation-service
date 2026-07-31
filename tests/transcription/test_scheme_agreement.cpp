// tests/transcription/test_scheme_agreement.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "goss/transcription/trapezoidal.hpp"
#include "goss/transcription/hermite_simpson.hpp"
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
