// tests/sim/test_sweep_serial.cpp
#include <gtest/gtest.h>
#include <vector>
#include "goss/model/model.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "goss/sim/initial_guess.hpp"
#include "goss/sim/sweep.hpp"

namespace {
goss::transcription::CompiledOcp build_queue(goss::model::Model& model) {
    auto q    = model.add_state("queue_length");
    auto rate = model.add_control("service_rate");
    model.add_parameter("arrival_rate", 2.0, 0.0, 10.0);
    model.set_state_bounds(q, 0.0, 1e19);
    model.set_control_bounds(rate, 0.0, 5.0);
    model.set_initial_state(q, 10.0);
    model.set_mesh(0.0, 5.0, 30);
    auto ocp = model.build(
        [](const auto& x, const auto& u, const auto& p, auto){
            using T = std::decay_t<decltype(x[0])>; return std::vector<T>{ p[0] - u[0] }; },
        [](const auto& x, const auto& u, const auto&, auto){
            using T = std::decay_t<decltype(x[0])>; return x[0] + T(0.1)*u[0]*u[0]; });
    return goss::transcription::HermiteSimpson::compile(ocp, "sweep_serial_queue");
}
}  // namespace

TEST(SweepSerial, SolvesEveryPointAndRecordsResults) {
    goss::model::Model model;
    auto compiled = build_queue(model);
    const auto z0 = goss::sim::linear_guess(model, compiled.layout);
    goss::solver::IpoptSolver solver;

    std::vector<std::vector<double>> grid = {{1.0}, {2.0}, {3.0}, {4.0}};
    auto result = goss::sim::run_sweep_serial(
        *compiled.problem, compiled.validator, solver, grid, z0);

    ASSERT_EQ(result.points.size(), 4u);
    EXPECT_EQ(result.num_succeeded(), 4u);
    // Objective monotonic in arrival rate.
    EXPECT_LT(result.points[0].objective_value, result.points[3].objective_value);
    // Parameters echoed back in order.
    EXPECT_EQ(result.points[2].parameters, (std::vector<double>{3.0}));
}

// Verify that each call to solve_with_parameters is a clean full re-assignment
// (solve-time injection) with no residual state leaking between consecutive
// grid points.  We place the same arrival_rate value (2.0) in non-adjacent
// positions (indices 0 and 2) and a distinct value (3.0) in between (index 1),
// then assert:
//   - points[0] and points[2] (both 2.0) have identical status AND objective_value
//   - points[0] and points[1] (2.0 vs 3.0) have distinct objective values
// If any residual state from point 1 bled into point 2 the objectives at
// indices 0 and 2 would differ, falsifying the first assertion.
TEST(SweepSerial, PointsAreIndependent) {
    goss::model::Model model;
    auto compiled = build_queue(model);
    const auto z0 = goss::sim::linear_guess(model, compiled.layout);
    goss::solver::IpoptSolver solver;

    // Grid: duplicate value at indices 0 and 2, distinct value at index 1.
    std::vector<std::vector<double>> grid = {{2.0}, {3.0}, {2.0}};
    auto result = goss::sim::run_sweep_serial(
        *compiled.problem, compiled.validator, solver, grid, z0);

    ASSERT_EQ(result.points.size(), 3u);
    // Both 2.0 points must succeed.
    EXPECT_EQ(result.points[0].status, goss::solver::SolverStatus::Success);
    EXPECT_EQ(result.points[2].status, goss::solver::SolverStatus::Success);
    // Same parameter → same objective (to floating-point equality; both runs
    // use the identical initial guess and problem — any residual state leaking
    // from point 1 into point 2 would break this).
    EXPECT_NEAR(result.points[0].objective_value,
                result.points[2].objective_value, 1e-9);
    // Distinct parameter → distinct objective.
    EXPECT_NE(result.points[0].objective_value, result.points[1].objective_value);
}

TEST(SweepSerial, InvalidPointRecordedNotThrown) {
    goss::model::Model model;
    auto compiled = build_queue(model);
    const auto z0 = goss::sim::linear_guess(model, compiled.layout);
    goss::solver::IpoptSolver solver;

    std::vector<std::vector<double>> grid = {{1.0}, {999.0}};  // 999 out of [0,10]
    auto result = goss::sim::run_sweep_serial(
        *compiled.problem, compiled.validator, solver, grid, z0);

    ASSERT_EQ(result.points.size(), 2u);
    EXPECT_EQ(result.points[0].status, goss::solver::SolverStatus::Success);
    EXPECT_EQ(result.points[1].status, goss::solver::SolverStatus::Failure);
    EXPECT_NE(result.points[1].message.find("arrival_rate"), std::string::npos);
}
