// tests/solver/test_nlopt_solver.cpp
// Integration tests: verify that NloptSolver (COBYLA) correctly solves
// bound-constrained and general-constrained NLP problems.
//
// HS1: min 100*(x2 - x1^2)^2 + (1 - x1)^2
//      x2 >= -1.5  (lower bound on x2)
//      x1 free
// x* = (1, 1),  f* = 0
//
// HS71: canonical IPOPT example with inequality + equality constraints
// HS35: single inequality-only QP
#include <gtest/gtest.h>
#include <memory>
#include "goss/solver/nlopt_solver.hpp"
#include "goss/nlp/nlp_problem.hpp"
#include "goss/ad/cppadcg_backend.hpp"
#include "solver/hs_fixtures.hpp"

namespace {

// HS1: min 100(x2-x1^2)^2 + (1-x1)^2 ; x2 >= -1.5 ; x* = (1,1), f* = 0. m=0.
struct HS1 {
    std::size_t input_size() const { return 2; }
    std::size_t output_size() const { return 1; }  // objective only, no constraints
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x) const {
        T a = x[1] - x[0]*x[0];
        T b = T(1) - x[0];
        return {T(100)*a*a + b*b};
    }
};

}  // namespace

TEST(NloptSolver, SolvesBoundConstrainedRosenbrockHS1) {
    HS1 f;
    auto backend = std::make_unique<goss::ad::CppADCGBackend>(f, f.input_size(), "nlopt_hs1");
    const double inf = 2e19;
    // x1 free, x2 >= -1.5 ; no general constraints (m=0).
    goss::nlp::NLPProblem problem(std::move(backend),
        {-inf, -1.5}, {inf, inf}, {}, {});
    goss::solver::NloptSolver solver;
    solver.set_max_evaluations(50000);
    auto result = solver.solve(problem, {-1.2, 1.0});
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    EXPECT_NEAR(result.objective_value, 0.0, 1e-4);
    EXPECT_NEAR(result.x[0], 1.0, 1e-2);
    EXPECT_NEAR(result.x[1], 1.0, 1e-2);
}

TEST(NloptSolver, SolvesHS71) {
    goss::solver::hs::HS71 f;
    auto backend = std::make_unique<goss::ad::CppADCGBackend>(f, f.input_size(), "nlopt_hs71");
    goss::nlp::NLPProblem problem(std::move(backend),
        {1.0, 1.0, 1.0, 1.0}, {5.0, 5.0, 5.0, 5.0},
        {25.0, 40.0}, {goss::solver::hs::kInf, 40.0});
    goss::solver::NloptSolver solver;
    solver.set_max_evaluations(100000);
    auto result = solver.solve(problem, {1.0, 5.0, 5.0, 1.0});
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    EXPECT_NEAR(result.objective_value, 17.0140173, 1e-2);  // COBYLA: looser tol
}

TEST(NloptSolver, SolvesHS35) {
    goss::solver::hs::HS35 f;
    auto backend = std::make_unique<goss::ad::CppADCGBackend>(f, f.input_size(), "nlopt_hs35");
    const double inf = goss::solver::hs::kInf;
    goss::nlp::NLPProblem problem(std::move(backend),
        {0.0, 0.0, 0.0}, {inf, inf, inf}, {-inf}, {3.0});
    goss::solver::NloptSolver solver;
    solver.set_max_evaluations(100000);
    auto result = solver.solve(problem, {0.5, 0.5, 0.5});
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    EXPECT_NEAR(result.objective_value, 1.0 / 9.0, 1e-3);
}
