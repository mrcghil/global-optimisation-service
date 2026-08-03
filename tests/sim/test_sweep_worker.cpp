// tests/sim/test_sweep_worker.cpp
#include <gtest/gtest.h>
#include <vector>
#include "goss/sim/sweep_worker.hpp"

// Additional headers needed for the fork-solve test.
#include "goss/model/model.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "goss/sim/initial_guess.hpp"

// ---------------------------------------------------------------------------
// Test 1: Serialization round-trip (in-process, fast — no fork)
// ---------------------------------------------------------------------------

TEST(SweepWorker, SerializeRoundTrip) {
    goss::sim::SweepPoint point;
    point.parameters = {1.5, 2.5};
    point.status = goss::solver::SolverStatus::Success;
    point.objective_value = 42.25;
    point.x = {0.1, 0.2, 0.3};
    point.message = "solved";

    const auto bytes = goss::sim::serialize_sweep_point(point);
    const auto back  = goss::sim::deserialize_sweep_point(bytes);

    EXPECT_EQ(back.parameters, point.parameters);
    EXPECT_EQ(back.status, point.status);
    EXPECT_DOUBLE_EQ(back.objective_value, point.objective_value);
    EXPECT_EQ(back.x, point.x);
    EXPECT_EQ(back.message, point.message);
}

// ---------------------------------------------------------------------------
// Test 2: Fork-solve — end-to-end child process inheriting compiled .so
// ---------------------------------------------------------------------------

TEST(SweepWorker, SolvesOnePointInChildProcess) {
    goss::model::Model model;
    auto q    = model.add_state("queue_length");
    auto rate = model.add_control("service_rate");
    model.add_parameter("arrival_rate", 2.0, 0.0, 10.0);
    model.set_state_bounds(q, 0.0, 1e19);
    model.set_control_bounds(rate, 0.0, 5.0);
    model.set_initial_state(q, 10.0);
    model.set_mesh(0.0, 5.0, 30);
    auto ocp = model.build(
        [](const auto& x, const auto& u, const auto& p, auto){
            using T = std::decay_t<decltype(x[0])>; return std::vector<T>{ p[0]-u[0] }; },
        [](const auto& x, const auto& u, const auto&, auto){
            using T = std::decay_t<decltype(x[0])>; return x[0] + T(0.1)*u[0]*u[0]; });
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "sweep_worker_queue");
    const auto z0 = goss::sim::linear_guess(model, compiled.layout);
    goss::solver::IpoptSolver solver;

    auto point = goss::sim::solve_point_in_child(
        *compiled.problem, compiled.validator, solver, {2.0}, z0);

    EXPECT_EQ(point.status, goss::solver::SolverStatus::Success);
    EXPECT_EQ(point.parameters, (std::vector<double>{2.0}));
    EXPECT_GT(point.objective_value, 0.0);
}
