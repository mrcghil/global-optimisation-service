// tests/model/test_composition_solve.cpp
// Integration tests added in Task 7.
#include <gtest/gtest.h>
#include <cstddef>
#include <vector>
#include "goss/model/component.hpp"
#include "goss/model/composed_model.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"

TEST(CompositionSolve, QueueModelWithDerivedServiceRate) {
    constexpr double ARRIVAL = 3.0;
    constexpr double BASE_RATE = 1.0;
    constexpr double SLOPE = 0.8;
    constexpr double MAX_RATE = 5.0;
    constexpr double WEIGHT = 0.1;
    constexpr double T_HORIZON = 5.0;
    constexpr std::size_t NUM_INTERVALS = 30;

    goss::model::ComposedModel composed;
    // One control: a throttle input (0 to 1) modulating the service rate.
    auto throttle_handle = composed.add_control("throttle", 0.0, 1.0);

    // Service component: reads queue_length (input state), publishes service_rate (derived inline).
    goss::model::Component service_component("service");
    auto queue_in_handle = service_component.input_state("queue_length");
    auto service_rate_handle = service_component.add_derived(
        "service_rate",
        // Validation lambda (double only — not used in the AD path).
        [BASE_RATE, SLOPE](const std::vector<double>& x,
                           const std::vector<double>& u,
                           const std::vector<double>& /*prev_d*/,
                           double /*t*/) {
            // service_rate = BASE_RATE + SLOPE * queue_length * throttle
            return BASE_RATE + SLOPE * x[0] * u[0];
        });
    composed.add_component(std::move(service_component));

    // Queue component: owns queue_length, dynamics consume service_rate derived.
    goss::model::Component queue_component("queue");
    auto q_handle = queue_component.add_state("queue_length");
    queue_component.set_state_bounds(q_handle, 0.0, goss::transcription::kInf);
    queue_component.set_initial_state(q_handle, 10.0);
    queue_component.set_dynamics(
        // Validation lambda.
        [ARRIVAL](const std::vector<double>& /*x*/,
                  const std::vector<double>& /*u*/,
                  const std::vector<double>& d,
                  double /*t*/) {
            // dq/dt = ARRIVAL - service_rate
            return std::vector<double>{ ARRIVAL - d[0] };
        });
    composed.add_component(std::move(queue_component));
    composed.set_mesh(0.0, T_HORIZON, NUM_INTERVALS);

    // Generic AD-safe lambdas passed to build() — these are the hot path.
    // Ordering: derived-expression lambdas (topo order), then dynamics lambdas
    // (component registration order), then cost lambda.

    // service_rate inline expression: BASE_RATE + SLOPE * x[queue_global_idx] * u[throttle_global_idx]
    // After resolve_names(): queue_length is global state 0, throttle is global control 0.
    auto service_rate_expr = [BASE_RATE, SLOPE](
        const auto& x, const auto& u, const auto& /*prev_d*/, auto /*t*/) {
        using T = typename std::decay_t<decltype(x)>::value_type;
        return T(BASE_RATE) + T(SLOPE) * x[0] * u[0];
    };

    // queue dynamics: dq/dt = ARRIVAL - deriveds[service_rate_global_idx=0]
    auto queue_dynamics = [ARRIVAL](
        const auto& /*x*/, const auto& /*u*/, const auto& d, auto /*t*/) {
        using T = typename std::decay_t<decltype(d)>::value_type;
        return std::vector<T>{ T(ARRIVAL) - d[0] };
    };

    // Combined cost: q + WEIGHT * throttle^2
    auto combined_cost = [WEIGHT](
        const auto& x, const auto& u, const auto& /*d*/, auto /*t*/) {
        using T = typename std::decay_t<decltype(x)>::value_type;
        return x[0] + T(WEIGHT) * u[0] * u[0];
    };

    auto ocp = composed.build(
        goss::model::make_derived_exprs(service_rate_expr),
        goss::model::make_component_dyns(queue_dynamics),
        combined_cost);

    // Solve via HermiteSimpson + IpoptSolver — identical pipeline to monolithic test.
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "composed_queue");
    goss::solver::IpoptSolver solver;
    std::vector<double> initial_guess(compiled.problem->num_variables(), 5.0);
    auto result = solver.solve(*compiled.problem, initial_guess);

    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);

    const auto& layout = compiled.layout;
    // q(0) pinned to 10.
    EXPECT_NEAR(result.x[layout.state_index(0, 0)], 10.0, 1e-6);
    // q stays >= 0 at every node.
    for (std::size_t k = 0; k < layout.num_nodes(); ++k) {
        EXPECT_GE(result.x[layout.state_index(k, 0)], -1e-4)
            << "queue_length negative at node " << k;
    }
    // throttle respected its box [0, 1].
    for (std::size_t k = 0; k < layout.num_nodes(); ++k) {
        double throttle_val = result.x[layout.control_index(k, 0)];
        EXPECT_GE(throttle_val, -1e-4);
        EXPECT_LE(throttle_val, 1.0 + 1e-4);
    }
    EXPECT_GT(result.objective_value, 0.0);
}
