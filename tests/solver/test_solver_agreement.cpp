// tests/solver/test_solver_agreement.cpp
// Cross-solver agreement test: proves IPOPT and NLopt reach the same optimum
// on HS71, validating the model + IPOPT's derivative path independently via
// the derivative-free baseline (COBYLA uses only f/g evaluations, no gradients).
#include <gtest/gtest.h>
#include <memory>
#include "goss/solver/ipopt_solver.hpp"
#include "goss/solver/nlopt_solver.hpp"
#include "goss/nlp/nlp_problem.hpp"
#include "goss/ad/cppadcg_backend.hpp"
#include "solver/hs_fixtures.hpp"

namespace {
goss::nlp::NLPProblem make_hs71(const std::string& name) {
    goss::solver::hs::HS71 f;
    auto backend = std::make_unique<goss::ad::CppADCGBackend>(f, f.input_size(), name);
    return goss::nlp::NLPProblem(std::move(backend),
        {1.0, 1.0, 1.0, 1.0}, {5.0, 5.0, 5.0, 5.0},
        {25.0, 40.0}, {goss::solver::hs::kInf, 40.0});
}
}  // namespace

TEST(SolverAgreement, IpoptAndNloptReachSameOptimumOnHS71) {
    auto problem_ipopt = make_hs71("agree_hs71_ipopt");
    auto problem_nlopt = make_hs71("agree_hs71_nlopt");
    const std::vector<double> x0{1.0, 5.0, 5.0, 1.0};

    goss::solver::IpoptSolver ipopt;
    auto r_ipopt = ipopt.solve(problem_ipopt, x0);

    goss::solver::NloptSolver nlopt;
    nlopt.set_max_evaluations(100000);
    auto r_nlopt = nlopt.solve(problem_nlopt, x0);

    ASSERT_EQ(r_ipopt.status, goss::solver::SolverStatus::Success);
    ASSERT_EQ(r_nlopt.status, goss::solver::SolverStatus::Success);
    // Independent solvers, independent derivative paths -> same objective.
    EXPECT_NEAR(r_ipopt.objective_value, r_nlopt.objective_value, 1e-2);
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(r_ipopt.x[i], r_nlopt.x[i], 5e-2) << "x[" << i << "] disagrees";
    }
}
