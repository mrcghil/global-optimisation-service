#include <algorithm>
#include <gtest/gtest.h>
#include <memory>
#include "goss/nlp/nlp_problem.hpp"
#include "goss/ad/cppadcg_backend.hpp"
#include "nlp/packed_fixtures.hpp"

namespace {
goss::nlp::NLPProblem make_quad_problem(const std::string& name) {
    goss::nlp::test::QuadraticWithLinearConstraint f;
    auto backend = std::make_unique<goss::ad::CppADCGBackend>(f, f.input_size(), name);
    return goss::nlp::NLPProblem(std::move(backend),
        {-10.0, -10.0}, {10.0, 10.0}, {0.0}, {0.0});
}
}  // namespace

TEST(NLPEval, ObjectiveMatchesPackedOutputZero) {
    auto problem = make_quad_problem("nlp_obj");
    EXPECT_DOUBLE_EQ(problem.eval_objective({1.0, 2.0}), 5.0);  // 1 + 4
}

TEST(NLPEval, ConstraintsMatchPackedOutputsRest) {
    auto problem = make_quad_problem("nlp_con");
    auto g = problem.eval_constraints({1.0, 2.0});
    ASSERT_EQ(g.size(), 1u);
    EXPECT_DOUBLE_EQ(g[0], 2.0);  // 1 + 2 - 1
}

TEST(NLPEval, EvalRejectsWrongXSize) {
    auto problem = make_quad_problem("nlp_badx");
    EXPECT_THROW(problem.eval_objective({1.0}), goss::nlp::NLPError);
    EXPECT_THROW(problem.eval_constraints({1.0, 2.0, 3.0}), goss::nlp::NLPError);
}

TEST(NLPEval, ObjectiveGradientIsDenseGradOfF) {
    auto problem = make_quad_problem("nlp_grad");
    auto grad = problem.eval_objective_gradient({3.0, 4.0});
    ASSERT_EQ(grad.size(), 2u);            // dense, size num_variables
    EXPECT_DOUBLE_EQ(grad[0], 6.0);        // d(x0^2+x1^2)/dx0 = 2*x0
    EXPECT_DOUBLE_EQ(grad[1], 8.0);        // = 2*x1
}

TEST(NLPEval, ConstraintJacobianValuesAndPatternAlign) {
    auto problem = make_quad_problem("nlp_cjac");
    const auto& pattern = problem.constraint_jacobian_sparsity();
    auto values = problem.eval_constraint_jacobian({3.0, 4.0});
    ASSERT_EQ(pattern.size(), values.size());
    // constraint 0 = x0 + x1 - 1, gradient (1, 1); both entries are constraint row 0
    for (std::size_t k = 0; k < pattern.size(); ++k) {
        EXPECT_EQ(pattern[k].first, 0u) << "constraint index must be 0";
        EXPECT_DOUBLE_EQ(values[k], 1.0);
    }
    EXPECT_EQ(pattern.size(), 2u);  // d/dx0 and d/dx1
}

TEST(NLPEval, ConstraintJacobianNeverReferencesObjectiveRow) {
    auto problem = make_quad_problem("nlp_cjac2");
    for (const auto& [constraint_index, col] : problem.constraint_jacobian_sparsity()) {
        EXPECT_LT(constraint_index, problem.num_constraints());
    }
}

TEST(NLPEval, LagrangianHessianObjectiveOnlyIsTwiceIdentity) {
    // objective x0^2 + x1^2 -> Hessian 2*I; constraint is linear -> zero Hessian.
    auto problem = make_quad_problem("nlp_laghess");
    const auto& pattern = problem.lagrangian_hessian_sparsity();
    auto values = problem.eval_lagrangian_hessian({1.0, 2.0}, 1.0, {0.0});
    ASSERT_EQ(pattern.size(), values.size());
    for (std::size_t k = 0; k < pattern.size(); ++k) {
        const auto [row, col] = pattern[k];
        EXPECT_GE(row, col);                      // lower triangle
        EXPECT_DOUBLE_EQ(values[k], row == col ? 2.0 : 0.0);
    }
}

TEST(NLPEval, LagrangianHessianRejectsWrongMultiplierCount) {
    auto problem = make_quad_problem("nlp_lagbad");
    EXPECT_THROW(problem.eval_lagrangian_hessian({1.0, 2.0}, 1.0, {0.0, 0.0}),
                 goss::nlp::NLPError);
}
