// tests/model/test_expr_lowering.cpp
// Tests for comparison-operator lowering: StateHandle >= double, <= double,
// ControlHandle >= double, <= double, and BoundaryPoint == double.
// Each test verifies that the operator returns the correct constraint struct
// and that apply_bound / apply_boundary forward correctly to Model setters.
// Also tests the integral() / CostFunctor DSL wrapper (Task 5).
#include <gtest/gtest.h>
#include "goss/model/expr/constraints.hpp"   // includes handles.hpp, model.hpp, transcription.hpp
#include "goss/model/expr/integral.hpp"
#include "goss/model/expr/operators.hpp"     // operator+/-/* on expression nodes
#include "goss/model/model.hpp"
#include "goss/transcription/transcription.hpp"
#include <cppad/cppad.hpp>

TEST(ExprLowering, StateGeqDoubleLowersToStateLowerBound) {
    goss::model::Model model;
    const auto q = model.add_state("q");
    const goss::model::expr::BoundConstraint bound = (q >= 0.0);
    EXPECT_EQ(bound.state_handle.index, q.index);
    EXPECT_DOUBLE_EQ(bound.lower_bound, 0.0);
    EXPECT_DOUBLE_EQ(bound.upper_bound, goss::transcription::kInf);
}

TEST(ExprLowering, StateLeqDoubleLowersToStateUpperBound) {
    goss::model::Model model;
    const auto q = model.add_state("q");
    const goss::model::expr::BoundConstraint bound = (q <= 100.0);
    EXPECT_DOUBLE_EQ(bound.lower_bound, -goss::transcription::kInf);
    EXPECT_DOUBLE_EQ(bound.upper_bound, 100.0);
}

TEST(ExprLowering, ControlGeqDoubleLowersToControlLowerBound) {
    goss::model::Model model;
    const auto rate = model.add_control("rate");
    const goss::model::expr::ControlBoundConstraint bound = (rate >= 0.0);
    EXPECT_EQ(bound.control_handle.index, rate.index);
    EXPECT_DOUBLE_EQ(bound.lower_bound, 0.0);
    EXPECT_DOUBLE_EQ(bound.upper_bound, goss::transcription::kInf);
}

TEST(ExprLowering, StateInitialEqDoubleLowersToBoundaryConstraint) {
    goss::model::Model model;
    const auto q = model.add_state("q");
    const goss::model::expr::BoundaryConstraint bc = (q.initial() == 10.0);
    EXPECT_EQ(bc.state_handle.index, q.index);
    EXPECT_DOUBLE_EQ(bc.fixed_value, 10.0);
    EXPECT_EQ(bc.kind, goss::model::expr::BoundaryConstraint::Kind::Initial);
}

TEST(ExprLowering, StateFinalEqDoubleLowersToFinalBoundaryConstraint) {
    goss::model::Model model;
    const auto q = model.add_state("q");
    const goss::model::expr::BoundaryConstraint bc = (q.final() == 1.0);
    EXPECT_EQ(bc.kind, goss::model::expr::BoundaryConstraint::Kind::Final);
    EXPECT_DOUBLE_EQ(bc.fixed_value, 1.0);
}

TEST(ExprLowering, ApplyBoundConstraintCallsModelSetter) {
    goss::model::Model model;
    const auto q    = model.add_state("q");
    const auto rate = model.add_control("rate");
    goss::model::expr::apply_bound(model, q >= 0.0);
    goss::model::expr::apply_bound(model, rate <= 5.0);
    EXPECT_DOUBLE_EQ(model.state_lower(0), 0.0);
    EXPECT_DOUBLE_EQ(model.state_upper(0), goss::transcription::kInf);
    EXPECT_DOUBLE_EQ(model.control_lower(0), -goss::transcription::kInf);
    EXPECT_DOUBLE_EQ(model.control_upper(0), 5.0);
}

TEST(ExprLowering, ApplyBoundaryConstraintCallsModelSetter) {
    goss::model::Model model;
    const auto q = model.add_state("q");
    goss::model::expr::apply_boundary(model, q.initial() == 10.0);
    EXPECT_TRUE(model.initial_fixed(0));
    EXPECT_DOUBLE_EQ(model.initial_value(0), 10.0);
    EXPECT_FALSE(model.final_fixed(0));
}

// ─── Task 5: integral() / CostFunctor tests ──────────────────────────────────

TEST(ExprIntegral, CostFunctorEvalMatchesManualCalculation) {
    using namespace goss::model::expr;
    // cost expression: q + 0.1 * rate^2 — at q=10, rate=3: 10 + 0.1*9 = 10.9
    const auto cost_expr    = StateLeaf{0} + ConstantExpr{0.1} * ControlLeaf{0} * ControlLeaf{0};
    const auto cost_functor = integral(cost_expr);
    const std::vector<double> x_vec{10.0};
    const std::vector<double> u_vec{3.0};
    // Verify the functor evaluates the expression tree correctly under double.
    EXPECT_DOUBLE_EQ(cost_functor(x_vec, u_vec, 0.0), 10.9);
}

TEST(ExprIntegral, CostFunctorInstantiatesUnderCppADAD) {
    using namespace goss::model::expr;
    // Verify that CostFunctor::operator() can be instantiated with CppAD::AD<double>
    // — this is required for the NLP solver to record gradients on the AD tape.
    using ADDouble = CppAD::AD<double>;
    const auto cost_expr    = StateLeaf{0} + ConstantExpr{0.1} * ControlLeaf{0} * ControlLeaf{0};
    const auto cost_functor = integral(cost_expr);
    const std::vector<ADDouble> x_ad{ADDouble(10.0)};
    const std::vector<ADDouble> u_ad{ADDouble(3.0)};
    const ADDouble result_ad = cost_functor(x_ad, u_ad, ADDouble(0.0));
    EXPECT_DOUBLE_EQ(CppAD::Value(result_ad), 10.9);
}

TEST(ExprIntegral, CostFunctorSatisfiesModelBuildCostFnContract) {
    // Verify that a CostFunctor can be passed to Model::build as the cost argument.
    // If CostFunctor does not satisfy the CostFn contract
    // (template<T> T operator()(const vector<T>&, const vector<T>&, T) const),
    // this test will fail to compile.
    using namespace goss::model::expr;
    goss::model::Model model;
    const auto q    = model.add_state("q");
    const auto rate = model.add_control("rate");
    model.set_mesh(0.0, 1.0, 2);
    const auto cost_functor = integral(
        StateLeaf{q.index} + ConstantExpr{0.1} * ControlLeaf{rate.index} * ControlLeaf{rate.index}
    );
    // Trivial dynamics: returns zero derivative regardless of state/control.
    // x_vec is used for value_type deduction — parameter named x_vec to avoid
    // shadowing the enclosing x variable (corrected form per plan self-correction).
    auto trivial_dynamics = [](const auto& x_vec, const auto& /*u*/, auto /*t*/) {
        using T2 = typename std::decay_t<decltype(x_vec)>::value_type;
        return std::vector<T2>{T2(0.0)};
    };
    // If CostFunctor does not satisfy CostFn, this will not compile.
    EXPECT_NO_THROW(model.build(trivial_dynamics, cost_functor));
}
