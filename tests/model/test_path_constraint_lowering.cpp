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

// Regression test: bare ControlHandle >= double MUST produce ControlBoundConstraint
// (box bound on the control, not a path constraint). This proves that operator>=
// on ControlHandle dispatches to the box-bound overload, not PathConstraintExpr.
TEST(PathConstraintLowering, BareControlHandleGeqStillProducesBoundConstraint) {
    goss::model::ControlHandle u{0};
    // Must compile as ControlBoundConstraint (the existing box-bound path).
    // If this accidentally resolved to PathConstraintExpr, the test would
    // fail to compile (ControlBoundConstraint has no path_constraint_lower field).
    goss::model::expr::ControlBoundConstraint bc = u >= 0.0;
    EXPECT_DOUBLE_EQ(bc.lower_bound, 0.0);
    EXPECT_DOUBLE_EQ(bc.upper_bound, goss::transcription::kInf);
}

// Tests for ExprModel::with_path_constraint() and build() path-constraint integration.

// Test: ExprModel::with_path_constraint() compiles and returns a new ExprModel whose
// PathTuple has one PathConstraintEntry appended. No build() call here — just verifies
// the type-accumulation pattern compiles.
TEST(PathConstraintLowering, ExprModelWithPathConstraintCompiles) {
    using namespace goss::model::expr;

    ExprModel<> model;
    const auto q_handle = model.add_state("state_x");
    model.set_mesh(0.0, 1.0, 4);

    // with_path_constraint() must accept a PathConstraintExpr and return a new ExprModel.
    // Constraint: x + 1.0 >= 0.0  (always satisfied for x >= -1.0)
    auto model_with_path = std::move(model)
        .with_path_constraint((StateLeaf{q_handle.index} + 1.0) >= 0.0);

    // The returned ExprModel should compile and allow further chaining.
    // We don't call build() here (no dynamics yet) — just verify the type compiles.
    SUCCEED();
}

// Test: ExprModel::build() with a path constraint produces an OcpProblem that has
// num_path_constraints == 1 and correctly populated bound vectors.
TEST(PathConstraintLowering, ExprModelBuildProducesOcpWithPathConstraints) {
    using namespace goss::model::expr;

    ExprModel<> model;
    const auto q_handle = model.add_state("state_x");
    const auto u_handle = model.add_control("control_u");
    model.set_mesh(0.0, 1.0, 4);
    model.apply(q_handle.initial() == 2.0);
    model.apply(q_handle >= -goss::transcription::kInf);
    model.apply(u_handle >= -1.0);
    model.apply(u_handle <= 1.0);

    // dx/dt = u,  cost = x^2,  path constraint: x + 1.0 >= 0.0
    auto ocp = std::move(model)
        .with_dynamics(q_handle, ControlLeaf{u_handle.index})
        .with_cost(integral(StateLeaf{q_handle.index} * StateLeaf{q_handle.index}))
        .with_path_constraint((StateLeaf{q_handle.index} + 1.0) >= 0.0)
        .build();

    // The OcpProblem should have 1 path constraint with bounds [0.0, +kInf].
    EXPECT_EQ(ocp.num_path_constraints, 1u);
    EXPECT_EQ(ocp.path_constraint_lower.size(), 1u);
    EXPECT_DOUBLE_EQ(ocp.path_constraint_lower[0], 0.0);
    EXPECT_DOUBLE_EQ(ocp.path_constraint_upper[0], goss::transcription::kInf);
}
