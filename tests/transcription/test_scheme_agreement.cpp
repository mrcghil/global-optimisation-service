// tests/transcription/test_scheme_agreement.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "goss/transcription/trapezoidal.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "transcription/ocp_fixtures.hpp"

namespace {
double final_state(goss::transcription::CompiledOcp& compiled, double guess) {
    goss::solver::IpoptSolver solver;
    std::vector<double> z0(compiled.problem->num_variables(), guess);
    auto result = solver.solve(*compiled.problem, z0);
    EXPECT_EQ(result.status, goss::solver::SolverStatus::Success);
    std::size_t last = compiled.layout.num_nodes() - 1;
    return result.x[compiled.layout.state_index(last, 0)];
}
}  // namespace

TEST(SchemeAgreement, TrapezoidalAndHermiteSimpsonAgreeOnExpDecay) {
    const double x0 = 1.0, tf = 1.0;
    const std::size_t intervals = 80;  // fine enough that both are accurate
    auto ocp_t = goss::transcription::test::make_exponential_decay(x0, tf, intervals);
    auto ocp_h = goss::transcription::test::make_exponential_decay(x0, tf, intervals);
    auto ct = goss::transcription::Trapezoidal::compile(ocp_t, "agree_trap");
    auto ch = goss::transcription::HermiteSimpson::compile(ocp_h, "agree_hs");
    double xt = final_state(ct, x0);
    double xh = final_state(ch, x0);
    double exact = goss::transcription::test::exp_decay_solution(x0, tf);
    EXPECT_NEAR(xt, exact, 1e-3);
    EXPECT_NEAR(xh, exact, 1e-3);
    EXPECT_NEAR(xt, xh, 2e-3);  // both converge to the same analytic solution
}
