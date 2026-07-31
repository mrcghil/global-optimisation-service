#include <gtest/gtest.h>
#include <map>
#include <memory>
#include "goss/nlp/nlp_problem.hpp"
#include "goss/ad/cppadcg_backend.hpp"
#include "nlp/packed_fixtures.hpp"

TEST(NLPIntegration, LagrangianHessianIncludesConstraintCurvature) {
    goss::nlp::test::NonlinearConstraintProblem f;
    auto backend = std::make_unique<goss::ad::CppADCGBackend>(f, f.input_size(), "nlp_integ");
    goss::nlp::NLPProblem problem(std::move(backend),
        {-10.0, -10.0}, {10.0, 10.0}, {-100.0}, {100.0});

    const auto& pattern = problem.lagrangian_hessian_sparsity();
    auto values = problem.eval_lagrangian_hessian({3.0, 4.0}, 1.0, {5.0});
    ASSERT_EQ(pattern.size(), values.size());

    // Expected lower-triangle Lagrangian Hessian at x=(3,4), sigma=1, lambda=5.
    // H = sigma*nabla^2_f + lambda*nabla^2_g
    // nabla^2_f = [[2,0],[0,2]], nabla^2_g = [[2*x1,2*x0],[2*x0,0]] = [[8,6],[6,0]]
    // H = [[2+5*8, 5*6],[5*6, 2+0]] = [[42,30],[30,2]]
    std::map<std::pair<std::size_t, std::size_t>, double> expected{
        {{0, 0}, 42.0}, {{1, 0}, 30.0}, {{1, 1}, 2.0}};
    for (std::size_t k = 0; k < pattern.size(); ++k) {
        const auto key = pattern[k];
        ASSERT_GE(key.first, key.second);  // lower triangle
        auto it = expected.find(key);
        ASSERT_NE(it, expected.end()) << "unexpected entry (" << key.first << "," << key.second << ")";
        EXPECT_DOUBLE_EQ(values[k], it->second);
    }
    // Every expected structural nonzero must appear in the pattern.
    EXPECT_EQ(pattern.size(), expected.size());
}

TEST(NLPIntegration, ConstraintEvalMatchesNonlinearValue) {
    goss::nlp::test::NonlinearConstraintProblem f;
    auto backend = std::make_unique<goss::ad::CppADCGBackend>(f, f.input_size(), "nlp_integ2");
    goss::nlp::NLPProblem problem(std::move(backend),
        {-10.0, -10.0}, {10.0, 10.0}, {-100.0}, {100.0});
    EXPECT_DOUBLE_EQ(problem.eval_constraints({3.0, 4.0})[0], 36.0);  // 9*4
}
