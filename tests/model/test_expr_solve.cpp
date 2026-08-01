// tests/model/test_expr_solve.cpp
//
// Flagship end-to-end tests for the operator-overload expression DSL.
// Test A: the queue optimal-control problem written entirely in the expr DSL,
//         solved with HermiteSimpson + IpoptSolver. Qualitative assertions only
//         (feasibility, bound compliance, q(0) pinned).
// Test B: numerical consistency — solves the same queue with both the plain
//         lambda path and the expr DSL path and asserts the objectives agree
//         within 1e-3. This is the spec's flagship correctness proof: if the
//         two paths diverge it signals a real bug in DynamicsFunctor or CostFunctor
//         (e.g. wrong placement by state_index, ConstantExpr wrapping 0 instead
//         of the actual constant, off-by-one in control indexing).

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "goss/model/expr/expr.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"

namespace {
// Shared constants for both tests — identical to test_model_solve.cpp queue problem
// so that Test B's consistency assertion is between truly equivalent problems.
constexpr double QUEUE_ARRIVAL   = 3.0;
constexpr double QUEUE_MAX_RATE  = 5.0;
constexpr double QUEUE_WEIGHT    = 0.1;
constexpr double QUEUE_T_FINAL   = 5.0;
constexpr double QUEUE_INITIAL   = 10.0;
constexpr std::size_t QUEUE_INTERVALS = 30;
}  // namespace

// Test A: the queue model written entirely with the operator-overload expression DSL.
// This is the flagship test — the spec's §7 sugar syntax.
// Problem: dq/dt = ARRIVAL - rate, q >= 0, 0 <= rate <= MAX_RATE, q(0) = QUEUE_INITIAL.
// Cost: integral(q + WEIGHT * rate^2).
TEST(ExprSolve, QueueModelWithOperatorOverloadDslSolvesSuccessfully) {
    using namespace goss::model::expr;

    ExprModel<> expr_model{};
    const auto q_handle    = expr_model.add_state("queue_length");
    const auto rate_handle = expr_model.add_control("service_rate");

    // Box bounds and boundary condition via comparison operators.
    // WHY apply() with comparison operators: the DSL merges successive bound
    // constraints so apply(q >= 0.0) then apply(rate <= MAX) do not overwrite
    // each other — they accumulate correctly into the underlying Model.
    expr_model.apply(q_handle >= 0.0);
    expr_model.apply(rate_handle >= 0.0);
    expr_model.apply(rate_handle <= QUEUE_MAX_RATE);
    expr_model.apply(q_handle.initial() == QUEUE_INITIAL);
    expr_model.set_mesh(0.0, QUEUE_T_FINAL, QUEUE_INTERVALS);

    // dq/dt = ARRIVAL - rate
    // WHY ConstantExpr{QUEUE_ARRIVAL}: raw double literals inside eval() would
    // be treated as constants that bypass AD recording; wrapping in ConstantExpr
    // guarantees the value is promoted to ScalarT on the CppAD tape.
    const auto dynamics_expression = ConstantExpr{QUEUE_ARRIVAL} - ControlLeaf{rate_handle.index};

    // cost = q + WEIGHT * rate^2
    const auto cost_expression =
        StateLeaf{q_handle.index} +
        ConstantExpr{QUEUE_WEIGHT} * ControlLeaf{rate_handle.index} * ControlLeaf{rate_handle.index};

    // with_dynamics and with_cost are rvalue-qualified; std::move is required to
    // transfer model_ so no deep copies of state vectors occur.
    auto solved_model = std::move(expr_model)
        .with_dynamics(q_handle, dynamics_expression)
        .with_cost(integral(cost_expression));

    auto ocp      = solved_model.build();
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "expr_queue");
    goss::solver::IpoptSolver solver;
    // Initial guess: all variables set to 5.0 (a feasible interior point for
    // this problem, consistent with what test_model_solve.cpp uses).
    std::vector<double> initial_guess(compiled.problem->num_variables(), 5.0);
    const auto result = solver.solve(*compiled.problem, initial_guess);

    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);

    const auto& layout = compiled.layout;
    // q(0) pinned to QUEUE_INITIAL by the boundary constraint.
    EXPECT_NEAR(result.x[layout.state_index(0, 0)], QUEUE_INITIAL, 1e-6);
    // q stays >= 0 at every node (allow tiny solver slack).
    for (std::size_t node_index = 0; node_index < layout.num_nodes(); ++node_index) {
        EXPECT_GE(result.x[layout.state_index(node_index, 0)], -1e-4)
            << "queue negative at node " << node_index;
    }
    // rate respected its box [0, QUEUE_MAX_RATE].
    for (std::size_t node_index = 0; node_index < layout.num_nodes(); ++node_index) {
        const double rate_value = result.x[layout.control_index(node_index, 0)];
        EXPECT_GE(rate_value, -1e-4);
        EXPECT_LE(rate_value, QUEUE_MAX_RATE + 1e-4);
    }
    EXPECT_GT(result.objective_value, 0.0);
}

// Test B: numerical consistency between the lambda DSL path and the expr DSL path.
// Both must solve to the same objective (within solver tolerance 1e-3).
// WHY this matters: if DynamicsFunctor places entries by tuple position instead of
// state_index, or if ConstantExpr wraps 0 instead of the actual value, the two
// paths diverge numerically — this test catches such bugs deterministically.
TEST(ExprSolve, QueueModelExprAndLambdaPathsProduceConsistentObjectives) {
    // --- Lambda path (same as test_model_solve.cpp QueueModelKeepsQueueNonNegative) ---
    double lambda_objective = 0.0;
    {
        goss::model::Model lambda_model;
        const auto q_lambda    = lambda_model.add_state("queue_length");
        const auto rate_lambda = lambda_model.add_control("service_rate");
        lambda_model.set_state_bounds(q_lambda, 0.0, goss::transcription::kInf);
        lambda_model.set_control_bounds(rate_lambda, 0.0, QUEUE_MAX_RATE);
        lambda_model.set_initial_state(q_lambda, QUEUE_INITIAL);
        lambda_model.set_mesh(0.0, QUEUE_T_FINAL, QUEUE_INTERVALS);

        // Generic AD-safe lambda: captures constants as ScalarT(...) to avoid
        // bare-double arithmetic on the CppAD tape, matching ConstantExpr semantics.
        auto lambda_dynamics = [](const auto& x_vec, const auto& u_vec, auto /*t*/) {
            using ScalarT = typename std::decay_t<decltype(x_vec)>::value_type;
            return std::vector<ScalarT>{ScalarT(QUEUE_ARRIVAL) - u_vec[0]};
        };
        auto lambda_cost = [](const auto& x_vec, const auto& u_vec, auto /*t*/) {
            using ScalarT = typename std::decay_t<decltype(x_vec)>::value_type;
            return x_vec[0] + ScalarT(QUEUE_WEIGHT) * u_vec[0] * u_vec[0];
        };

        auto ocp_lambda      = lambda_model.build(lambda_dynamics, lambda_cost);
        auto compiled_lambda = goss::transcription::HermiteSimpson::compile(ocp_lambda, "lambda_queue_consistency");
        goss::solver::IpoptSolver solver_lambda;
        std::vector<double> z0_lambda(compiled_lambda.problem->num_variables(), 5.0);
        const auto result_lambda = solver_lambda.solve(*compiled_lambda.problem, z0_lambda);
        ASSERT_EQ(result_lambda.status, goss::solver::SolverStatus::Success);
        lambda_objective = result_lambda.objective_value;
    }

    // --- Expr DSL path ---
    double expr_objective = 0.0;
    {
        using namespace goss::model::expr;
        ExprModel<> expr_model{};
        const auto q_expr    = expr_model.add_state("queue_length");
        const auto rate_expr = expr_model.add_control("service_rate");
        expr_model.apply(q_expr >= 0.0);
        expr_model.apply(rate_expr >= 0.0);
        expr_model.apply(rate_expr <= QUEUE_MAX_RATE);
        expr_model.apply(q_expr.initial() == QUEUE_INITIAL);
        expr_model.set_mesh(0.0, QUEUE_T_FINAL, QUEUE_INTERVALS);

        const auto dyn_expr  = ConstantExpr{QUEUE_ARRIVAL} - ControlLeaf{rate_expr.index};
        const auto cost_expr = StateLeaf{q_expr.index} +
            ConstantExpr{QUEUE_WEIGHT} * ControlLeaf{rate_expr.index} * ControlLeaf{rate_expr.index};

        auto ocp_expr      = std::move(expr_model)
            .with_dynamics(q_expr, dyn_expr)
            .with_cost(integral(cost_expr))
            .build();
        auto compiled_expr = goss::transcription::HermiteSimpson::compile(ocp_expr, "expr_queue_consistency");
        goss::solver::IpoptSolver solver_expr;
        std::vector<double> z0_expr(compiled_expr.problem->num_variables(), 5.0);
        const auto result_expr = solver_expr.solve(*compiled_expr.problem, z0_expr);
        ASSERT_EQ(result_expr.status, goss::solver::SolverStatus::Success);
        expr_objective = result_expr.objective_value;
    }

    // The two paths must produce the same objective within solver tolerance.
    // A larger gap signals a correctness bug (scaling, indexing, or wrong constant).
    EXPECT_NEAR(lambda_objective, expr_objective, 1e-3)
        << "Lambda objective: " << lambda_objective
        << ", Expr objective: "  << expr_objective;
}
