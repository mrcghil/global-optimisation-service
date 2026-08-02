// tests/model/test_path_constraint_lowering.cpp
// Tests for PathConstraintExpr construction, PathConstraintFunctor evaluation,
// and expression-typed operator>=/<=/== overloads.
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "goss/model/expr/expr.hpp"
#include "goss/transcription/transcription.hpp"

using namespace goss::model::expr;

// Test: (StateLeaf + ConstantExpr) >= 0.0 produces a PathConstraintExpr,
// not a BoundConstraint. The returned type should NOT be BoundConstraint.
TEST(PathConstraintLowering, ExpressionPlusConstantGeqProducesPathConstraintExpr) {
    // This must compile: (StateLeaf{0} + 1.0) >= 0.0
    auto path_expr = (StateLeaf{0} + 1.0) >= 0.0;
    // The result must have path_constraint_lower == 0.0 and path_constraint_upper == kInf.
    EXPECT_DOUBLE_EQ(path_expr.path_constraint_lower, 0.0);
    EXPECT_DOUBLE_EQ(path_expr.path_constraint_upper, goss::transcription::kInf);

    // Evaluate under double: x=[2.0], u=[], t=0 => (2.0 + 1.0) - 0.0 = 3.0
    // The expression is: StateLeaf{0} + 1.0 - 0.0 (the RHS is subtracted so g=expr-rhs)
    // Convention: (expr) >= rhs  =>  g = expr - rhs >= 0  =>  g(x,u,t) = expr.eval - rhs
    std::vector<double> x = {2.0};
    std::vector<double> u = {};
    double g_val = path_expr.constraint_expression.eval(x, u, 0.0);
    EXPECT_NEAR(g_val, 3.0, 1e-12);
}

// Test: (StateLeaf - ConstantExpr) <= 1.0 produces PathConstraintExpr with
// gl=-kInf, gu=0.0. Expression g = (expr - rhs).
// (x - 1.0) <= 1.0 => g = (x - 1.0) - 1.0 = x - 2.0 <= 0 => gu=0.
TEST(PathConstraintLowering, ExprLeqProducesNegativeUpperBound) {
    auto path_expr = (StateLeaf{0} - 1.0) <= 1.0;
    EXPECT_DOUBLE_EQ(path_expr.path_constraint_lower, -goss::transcription::kInf);
    EXPECT_DOUBLE_EQ(path_expr.path_constraint_upper, 0.0);

    std::vector<double> x = {3.0};
    std::vector<double> u = {};
    // g = (x - 1.0) - 1.0 = 3.0 - 2.0 = 1.0
    double g_val = path_expr.constraint_expression.eval(x, u, 0.0);
    EXPECT_NEAR(g_val, 1.0, 1e-12);
}

// Test: (ControlLeaf * ControlLeaf) == 0.0 produces equality constraint [0,0].
TEST(PathConstraintLowering, ExprEqZeroProducesEqualityConstraint) {
    auto path_expr = (ControlLeaf{0} * ControlLeaf{0}) == 0.0;
    EXPECT_DOUBLE_EQ(path_expr.path_constraint_lower, 0.0);
    EXPECT_DOUBLE_EQ(path_expr.path_constraint_upper, 0.0);
}

// Test: PathConstraintFunctor with a single entry evaluates correctly.
TEST(PathConstraintLowering, PathConstraintFunctorEvaluatesSingleEntry) {
    // Manually construct PathConstraintEntry and PathConstraintFunctor.
    // Entry: g(x,u,t) = x[0] + 1.0 - 0.0 (from (StateLeaf{0}+1.0) >= 0.0)
    using ExprType = BinaryExpr<SubTag,
                        BinaryExpr<AddTag, StateLeaf, ConstantExpr>,
                        ConstantExpr>;
    // g = (x[0] + 1.0) - 0.0
    ExprType g_expr{
        BinaryExpr<AddTag, StateLeaf, ConstantExpr>{ StateLeaf{0}, ConstantExpr{1.0} },
        ConstantExpr{0.0}
    };
    PathConstraintEntry<ExprType> entry{ std::move(g_expr), 0.0, goss::transcription::kInf };

    auto entries_tuple = std::make_tuple(std::move(entry));
    PathConstraintFunctor<decltype(entries_tuple)> functor{ std::move(entries_tuple), 1 };

    std::vector<double> x = {4.0};
    std::vector<double> u = {};
    auto result = functor(x, u, 0.0);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0], 5.0, 1e-12);  // 4.0 + 1.0 - 0.0 = 5.0
}

// Test: bare StateHandle >= double STILL produces BoundConstraint (box bound, not path).
// This test verifies no overload-resolution regression.
TEST(PathConstraintLowering, BareStateHandleGeqStillProducesBoundConstraint) {
    goss::model::StateHandle q_handle{0};
    // Must compile as BoundConstraint (the existing box-bound path).
    // If this accidentally resolved to PathConstraintExpr, the test would
    // fail to compile (BoundConstraint has no path_constraint_lower field).
    goss::model::expr::BoundConstraint bc = q_handle >= 0.0;
    EXPECT_DOUBLE_EQ(bc.lower_bound, 0.0);
    EXPECT_DOUBLE_EQ(bc.upper_bound, goss::transcription::kInf);
}
