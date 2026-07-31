// tests/model/test_model_solve.cpp
//
// End-to-end DSL solves: Model -> HermiteSimpson -> IpoptSolver.
// Covers the first num_controls > 0 end-to-end paths and the flagship queue example.

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "goss/model/model.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"

// --- Test 1: min-energy double integrator ---
// dx/dt = u, cost = integral(u^2), x(0)=0, x(T)=1 fixed.
// Analytic optimum: u = 1/T (constant), objective = 1/T.
TEST(ModelSolve, MinEnergyDoubleIntegratorReachesTarget) {
    const double T = 2.0;
    const std::size_t intervals = 20;
    goss::model::Model model;
    auto x = model.add_state("position");
    auto u = model.add_control("accel");
    model.set_control_bounds(u, -10.0, 10.0);
    model.set_initial_state(x, 0.0);
    model.set_final_state(x, 1.0);
    model.set_mesh(0.0, T, intervals);

    // dynamics dx/dt = u ; running cost u^2.
    auto dynamics = [](const auto& xx, const auto& uu, auto /*t*/) {
        using T2 = typename std::decay_t<decltype(xx)>::value_type;
        return std::vector<T2>{ uu[0] };
    };
    auto cost = [](const auto& /*xx*/, const auto& uu, auto /*t*/) {
        return uu[0] * uu[0];
    };

    auto ocp = model.build(dynamics, cost);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "model_minenergy");
    goss::solver::IpoptSolver solver;
    std::vector<double> z0(compiled.problem->num_variables(), 0.5);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);

    // final position must be 1.0 (pinned); objective ~= 1/T = 0.5 for the analytic min-energy control.
    std::size_t last = compiled.layout.num_nodes() - 1;
    double x_final = result.x[compiled.layout.state_index(last, 0)];
    EXPECT_NEAR(x_final, 1.0, 1e-6);
    EXPECT_NEAR(result.objective_value, 1.0 / T, 1e-3);
}

// --- Test 2: flagship queue model ---
// dq/dt = ARRIVAL - rate, q >= 0, 0 <= rate <= MAX_RATE, q(0) = 10.
// cost = integral(q + WEIGHT * rate^2). Qualitative: feasibility + bounds + q(0) pinned.
TEST(ModelSolve, QueueModelKeepsQueueNonNegative) {
    const double ARRIVAL = 3.0, MAX_RATE = 5.0, WEIGHT = 0.1, T = 5.0;
    const std::size_t intervals = 30;
    goss::model::Model model;
    auto q = model.add_state("queue_length");
    auto rate = model.add_control("service_rate");
    model.set_state_bounds(q, 0.0, goss::transcription::kInf);   // q >= 0
    model.set_control_bounds(rate, 0.0, MAX_RATE);               // 0 <= rate <= MAX
    model.set_initial_state(q, 10.0);                            // q(0) = 10
    model.set_mesh(0.0, T, intervals);

    // dq/dt = ARRIVAL - rate ; running cost q + WEIGHT*rate^2.
    auto dynamics = [ARRIVAL](const auto& xx, const auto& uu, auto /*t*/) {
        using T2 = typename std::decay_t<decltype(xx)>::value_type;
        return std::vector<T2>{ T2(ARRIVAL) - uu[0] };
    };
    auto cost = [WEIGHT](const auto& xx, const auto& uu, auto /*t*/) {
        using T2 = typename std::decay_t<decltype(xx)>::value_type;
        return xx[0] + T2(WEIGHT) * uu[0] * uu[0];
    };

    auto ocp = model.build(dynamics, cost);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "model_queue");
    goss::solver::IpoptSolver solver;
    std::vector<double> z0(compiled.problem->num_variables(), 5.0);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);

    const auto& layout = compiled.layout;
    // q(0) pinned to 10.
    EXPECT_NEAR(result.x[layout.state_index(0, 0)], 10.0, 1e-6);
    // q stays >= 0 at every node (allow tiny solver slack).
    for (std::size_t k = 0; k < layout.num_nodes(); ++k) {
        EXPECT_GE(result.x[layout.state_index(k, 0)], -1e-4) << "queue negative at node " << k;
    }
    // rate respected its box [0, MAX_RATE].
    for (std::size_t k = 0; k < layout.num_nodes(); ++k) {
        double r = result.x[layout.control_index(k, 0)];
        EXPECT_GE(r, -1e-4);
        EXPECT_LE(r, MAX_RATE + 1e-4);
    }
    EXPECT_GT(result.objective_value, 0.0);
}
