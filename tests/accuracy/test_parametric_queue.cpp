// tests/accuracy/test_parametric_queue.cpp
#include <gtest/gtest.h>
#include <vector>
#include "goss/model/model.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "goss/sim/initial_guess.hpp"
#include "goss/sim/parameters.hpp"

TEST(ParametricQueue, CompileOnceSolveManyAcrossArrivalRates) {
    constexpr double MAX_RATE = 5.0;
    goss::model::Model model;
    auto q    = model.add_state("queue_length");
    auto rate = model.add_control("service_rate");
    auto arrival = model.add_parameter("arrival_rate", /*default=*/2.0, 0.0, 10.0);
    (void)arrival;
    model.set_state_bounds(q, 0.0, 1e19);            // q >= 0
    model.set_control_bounds(rate, 0.0, MAX_RATE);
    model.set_initial_state(q, 10.0);
    model.set_mesh(0.0, 5.0, 30);

    // dq/dt = arrival_rate - service_rate ; cost = integral(q + 0.1*rate^2)
    auto dynamics = [](const auto& x, const auto& u, const auto& p, auto) {
        using T = std::decay_t<decltype(x[0])>;
        return std::vector<T>{ p[0] - u[0] };
    };
    auto cost = [](const auto& x, const auto& u, const auto&, auto) {
        using T = std::decay_t<decltype(x[0])>;
        return x[0] + T(0.1) * u[0] * u[0];
    };

    auto ocp = model.build(dynamics, cost);
    // COMPILE ONCE.
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "parametric_queue");
    const auto z0 = goss::sim::linear_guess(model, compiled.layout);

    goss::solver::IpoptSolver solver;

    goss::sim::apply_parameters(*compiled.problem, compiled.validator, {1.0});
    auto low = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(low.status, goss::solver::SolverStatus::Success);

    goss::sim::apply_parameters(*compiled.problem, compiled.validator, {4.0});
    auto high = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(high.status, goss::solver::SolverStatus::Success);

    // Higher arrival rate => costlier optimum (queue harder to drain).
    EXPECT_GT(high.objective_value, low.objective_value);
}
