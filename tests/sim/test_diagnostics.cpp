// tests/sim/test_diagnostics.cpp
#include <gtest/gtest.h>
#include "goss/sim/diagnostics.hpp"
#include "goss/solver/solver_result.hpp"

namespace {
goss::solver::SolverResult with_status(goss::solver::SolverStatus s, const std::string& msg = "") {
    goss::solver::SolverResult r;
    r.status = s;
    r.message = msg;
    return r;
}
}  // namespace

TEST(Diagnostics, SuccessIsOk) {
    auto d = goss::sim::diagnose(with_status(goss::solver::SolverStatus::Success));
    EXPECT_TRUE(d.ok);
}

TEST(Diagnostics, InfeasibleGivesAdvice) {
    auto d = goss::sim::diagnose(with_status(goss::solver::SolverStatus::InfeasibleProblem));
    EXPECT_FALSE(d.ok);
    EXPECT_NE(d.summary.find("infeasible"), std::string::npos);
    EXPECT_FALSE(d.advice.empty());
}

TEST(Diagnostics, IterationLimitSuggestsGuessOrMesh) {
    auto d = goss::sim::diagnose(with_status(goss::solver::SolverStatus::IterationLimit));
    EXPECT_FALSE(d.ok);
    EXPECT_NE(d.advice.find("initial guess"), std::string::npos);
}

TEST(Diagnostics, FailureAppendsSolverMessage) {
    auto d = goss::sim::diagnose(with_status(goss::solver::SolverStatus::Failure, "code 42"));
    EXPECT_FALSE(d.ok);
    EXPECT_NE(d.summary.find("code 42"), std::string::npos);
}

TEST(Diagnostics, SuccessButBadIntegrationIsDowngraded) {
    auto d = goss::sim::diagnose(with_status(goss::solver::SolverStatus::Success), /*err=*/0.5, /*tol=*/1e-3);
    EXPECT_FALSE(d.ok);
    EXPECT_NE(d.summary.find("re-integration"), std::string::npos);
}

TEST(Diagnostics, SuccessAndGoodIntegrationStaysOk) {
    auto d = goss::sim::diagnose(with_status(goss::solver::SolverStatus::Success), /*err=*/1e-6, /*tol=*/1e-3);
    EXPECT_TRUE(d.ok);
}
