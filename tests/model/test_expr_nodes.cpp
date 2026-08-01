// tests/model/test_expr_nodes.cpp
#include <gtest/gtest.h>
#include "goss/model/expr/errors.hpp"
#include "goss/model/expr/nodes.hpp"
#include "goss/model/expr/operators.hpp"
// CppAD is available via goss_model -> goss_transcription -> goss_nlp -> goss_ad -> cppadcg
#include <cppad/cppad.hpp>
#include <vector>

TEST(ExprError, IsThrowableAndCarriesMessage) {
    try {
        throw goss::model::expr::ExprError("test error");
    } catch (const goss::model::expr::ExprError& caught_error) {
        EXPECT_STREQ(caught_error.what(), "test error");
    }
}

TEST(ExprNodes, ConstantEvalReturnsWrappedValue) {
    const goss::model::expr::ConstantExpr constant_node{3.14};
    const std::vector<double> x_empty{};
    const std::vector<double> u_empty{};
    EXPECT_DOUBLE_EQ(constant_node.eval<double>(x_empty, u_empty, 0.0), 3.14);
}

TEST(ExprNodes, StateLeafIndexesXVector) {
    const goss::model::expr::StateLeaf state_node{1};  // second state
    const std::vector<double> x_vec{10.0, 20.0, 30.0};
    const std::vector<double> u_empty{};
    EXPECT_DOUBLE_EQ(state_node.eval<double>(x_vec, u_empty, 0.0), 20.0);
}

TEST(ExprNodes, ControlLeafIndexesUVector) {
    const goss::model::expr::ControlLeaf control_node{0};
    const std::vector<double> x_empty{};
    const std::vector<double> u_vec{7.5};
    EXPECT_DOUBLE_EQ(control_node.eval<double>(x_empty, u_vec, 0.0), 7.5);
}

TEST(ExprNodes, TimeLeafReturnsT) {
    const goss::model::expr::TimeLeaf time_node{};
    const std::vector<double> x_empty{};
    const std::vector<double> u_empty{};
    EXPECT_DOUBLE_EQ(time_node.eval<double>(x_empty, u_empty, 2.5), 2.5);
}

TEST(ExprNodes, BinaryAddExprSumsLeaves) {
    using namespace goss::model::expr;
    const BinaryExpr<AddTag, StateLeaf, ConstantExpr> add_node{StateLeaf{0}, ConstantExpr{5.0}};
    const std::vector<double> x_vec{3.0};
    const std::vector<double> u_empty{};
    // 3.0 + 5.0 = 8.0
    EXPECT_DOUBLE_EQ(add_node.eval<double>(x_vec, u_empty, 0.0), 8.0);
}

TEST(ExprNodes, BinaryMulExprMultipliesLeaves) {
    using namespace goss::model::expr;
    const BinaryExpr<MulTag, ConstantExpr, ControlLeaf> mul_node{ConstantExpr{2.0}, ControlLeaf{0}};
    const std::vector<double> x_empty{};
    const std::vector<double> u_vec{4.0};
    // 2.0 * 4.0 = 8.0
    EXPECT_DOUBLE_EQ(mul_node.eval<double>(x_empty, u_vec, 0.0), 8.0);
}

TEST(ExprNodes, UnaryNegExprNegatesOperand) {
    using namespace goss::model::expr;
    const UnaryNegExpr<ConstantExpr> neg_node{ConstantExpr{3.0}};
    const std::vector<double> x_empty{};
    const std::vector<double> u_empty{};
    EXPECT_DOUBLE_EQ(neg_node.eval<double>(x_empty, u_empty, 0.0), -3.0);
}

TEST(ExprNodes, NestedExprComposesCorrectly) {
    // Represent: state[0] + 2.0 * control[0]
    using namespace goss::model::expr;
    const BinaryExpr<AddTag,
        StateLeaf,
        BinaryExpr<MulTag, ConstantExpr, ControlLeaf>>
        nested_add{
            StateLeaf{0},
            BinaryExpr<MulTag, ConstantExpr, ControlLeaf>{ConstantExpr{2.0}, ControlLeaf{0}}
        };
    const std::vector<double> x_vec{10.0};
    const std::vector<double> u_vec{3.0};
    // 10.0 + 2.0 * 3.0 = 16.0
    EXPECT_DOUBLE_EQ(nested_add.eval<double>(x_vec, u_vec, 0.0), 16.0);
}

TEST(ExprOperators, PlusBuildsBinaryAddExpr) {
    using namespace goss::model::expr;
    const auto add_expr = StateLeaf{0} + ConstantExpr{1.0};
    const std::vector<double> x_vec{9.0};
    const std::vector<double> u_empty{};
    EXPECT_DOUBLE_EQ(add_expr.eval<double>(x_vec, u_empty, 0.0), 10.0);
}

TEST(ExprOperators, MinusBuildsSubExpr) {
    using namespace goss::model::expr;
    const auto sub_expr = ConstantExpr{5.0} - ControlLeaf{0};
    const std::vector<double> x_empty{};
    const std::vector<double> u_vec{2.0};
    EXPECT_DOUBLE_EQ(sub_expr.eval<double>(x_empty, u_vec, 0.0), 3.0);
}

TEST(ExprOperators, MulBuildsMulExpr) {
    using namespace goss::model::expr;
    const auto mul_expr = ConstantExpr{3.0} * ControlLeaf{0};
    const std::vector<double> x_empty{};
    const std::vector<double> u_vec{4.0};
    EXPECT_DOUBLE_EQ(mul_expr.eval<double>(x_empty, u_vec, 0.0), 12.0);
}

TEST(ExprOperators, UnaryMinusNegatesExpr) {
    using namespace goss::model::expr;
    const auto neg_expr = -StateLeaf{0};
    const std::vector<double> x_vec{7.0};
    const std::vector<double> u_empty{};
    EXPECT_DOUBLE_EQ(neg_expr.eval<double>(x_vec, u_empty, 0.0), -7.0);
}

TEST(ExprOperators, DoubleLiteralOnRhsIsWrappedImplicitly) {
    using namespace goss::model::expr;
    // q + 5.0 — the double overload wraps 5.0 in ConstantExpr.
    const auto add_double_expr = StateLeaf{0} + 5.0;
    const std::vector<double> x_vec{2.0};
    const std::vector<double> u_empty{};
    EXPECT_DOUBLE_EQ(add_double_expr.eval<double>(x_vec, u_empty, 0.0), 7.0);
}

TEST(ExprOperators, ComposedExprInstantiatesUnderCppADAD) {
    // Verify that q + w * rate * rate composes and eval<AD<double>> does not
    // fail to compile or produce NaN.  w = 0.1, q = 10.0, rate = 3.0 => 10 + 0.1*9 = 10.9
    using namespace goss::model::expr;
    using ADDouble = CppAD::AD<double>;
    const auto weight_mul_rate    = ConstantExpr{0.1} * ControlLeaf{0};
    const auto weight_rate_sq     = weight_mul_rate * ControlLeaf{0};
    const auto cost_expr          = StateLeaf{0} + weight_rate_sq;
    const std::vector<ADDouble> x_ad{ADDouble(10.0)};
    const std::vector<ADDouble> u_ad{ADDouble(3.0)};
    const ADDouble t_ad{0.0};
    const ADDouble result_ad = cost_expr.eval<ADDouble>(x_ad, u_ad, t_ad);
    EXPECT_DOUBLE_EQ(CppAD::Value(result_ad), 10.9);
}
