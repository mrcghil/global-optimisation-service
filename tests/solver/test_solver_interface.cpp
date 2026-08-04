// tests/solver/test_solver_interface.cpp
#include <gtest/gtest.h>
#include <memory>
#include "goss/solver/solver.hpp"
#include "goss/solver/solver_result.hpp"
#include "goss/solver/errors.hpp"

namespace {
// Minimal stub proving the interface is implementable + polymorphic.
class StubSolver : public goss::solver::Solver {
 public:
    goss::solver::SolverResult solve(const goss::nlp::NLPProblem&,
                                     const std::vector<double>& initial_guess,
                                     const std::vector<double>& /*parameters*/ = {}) override {
        goss::solver::SolverResult result;
        result.status = goss::solver::SolverStatus::Success;
        result.x = initial_guess;
        result.objective_value = 0.0;
        return result;
    }
};
}  // namespace

TEST(SolverInterface, StatusEnumHasExpectedValues) {
    EXPECT_NE(goss::solver::SolverStatus::Success, goss::solver::SolverStatus::Failure);
}

TEST(SolverInterface, SolverErrorIsThrowable) {
    EXPECT_THROW(throw goss::solver::SolverError("boom"), goss::solver::SolverError);
}

TEST(SolverInterface, IsPolymorphic) {
    std::unique_ptr<goss::solver::Solver> solver = std::make_unique<StubSolver>();
    auto result = solver->solve(*static_cast<goss::nlp::NLPProblem*>(nullptr), {1.0, 2.0});
    // Note: StubSolver never dereferences the problem, so the null cast is safe HERE only.
    EXPECT_EQ(result.status, goss::solver::SolverStatus::Success);
    ASSERT_EQ(result.x.size(), 2u);
}
