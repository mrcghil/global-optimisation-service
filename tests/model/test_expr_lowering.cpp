// tests/model/test_expr_lowering.cpp
// Tests for comparison-operator lowering: StateHandle >= double, <= double,
// ControlHandle >= double, <= double, and BoundaryPoint == double.
// Each test verifies that the operator returns the correct constraint struct
// and that apply_bound / apply_boundary forward correctly to Model setters.
#include <gtest/gtest.h>
#include "goss/model/expr/constraints.hpp"   // includes handles.hpp, model.hpp, transcription.hpp
#include "goss/model/model.hpp"
#include "goss/transcription/transcription.hpp"

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
