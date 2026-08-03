// tests/sim/test_sweep_workflow.cpp
//
// End-to-end 2-D sweep test: exercises the full public sweep API composed
// together as a user would.  Builds a compiled parametric OCP (arrival rate ×
// cost weight, queue problem), produces a 9-point Cartesian-product grid via
// make_grid, runs run_sweep_parallel over it, and asserts correctness.
//
// Correctness gate: parallel 2-D results must match run_sweep_serial over the
// same make_grid output — same status, objective within 1e-6, parameters equal.
#include <gtest/gtest.h>
#include <vector>
#include "goss/model/model.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "goss/sim/initial_guess.hpp"
#include "goss/sim/sweep.hpp"

TEST(SweepWorkflow, TwoDimensionalArrivalRateByCostWeight) {
    goss::model::Model model;
    auto q    = model.add_state("queue_length");
    auto rate = model.add_control("service_rate");
    model.add_parameter("arrival_rate", 2.0, 0.0, 10.0);   // param 0
    model.add_parameter("cost_weight",  0.1, 0.0, 10.0);   // param 1
    model.set_state_bounds(q, 0.0, 1e19);
    model.set_control_bounds(rate, 0.0, 5.0);
    model.set_initial_state(q, 10.0);
    model.set_mesh(0.0, 5.0, 25);
    auto ocp = model.build(
        [](const auto& x, const auto& u, const auto& p, auto){
            using T = std::decay_t<decltype(x[0])>; return std::vector<T>{ p[0] - u[0] }; },
        [](const auto& x, const auto& u, const auto& p, auto){
            using T = std::decay_t<decltype(x[0])>; return x[0] + p[1]*u[0]*u[0]; });
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "sweep_workflow_queue");
    const auto z0 = goss::sim::linear_guess(model, compiled.layout);
    goss::solver::IpoptSolver solver;

    // 3 arrival-rate values × 3 cost-weight values → 9-point Cartesian product.
    // Axis 0 (arrival_rate) varies slowest (row-major).
    auto grid = goss::sim::make_grid({{1.0, 2.0, 3.0}, {0.05, 0.1, 0.5}});  // 9 points

    ASSERT_EQ(grid.size(), 9u) << "make_grid must produce 3×3=9 combinations";

    goss::sim::SweepConfig config; config.max_parallel_workers = 4;
    auto result = goss::sim::run_sweep_parallel(
        *compiled.problem, compiled.validator, solver, grid, z0, config);

    ASSERT_EQ(result.points.size(), 9u);
    EXPECT_EQ(result.num_succeeded(), 9u);
}

// Serial-vs-parallel correctness gate: every point from the 2-D grid must
// have the same status, parameters, and objective value (within 1e-6) whether
// solved by the serial oracle or the parallel process pool.
TEST(SweepWorkflow, TwoDimensionalParallelMatchesSerial) {
    // Serial model + compiled problem
    goss::model::Model model_serial;
    {
        auto q    = model_serial.add_state("queue_length");
        auto rate = model_serial.add_control("service_rate");
        model_serial.add_parameter("arrival_rate", 2.0, 0.0, 10.0);
        model_serial.add_parameter("cost_weight",  0.1, 0.0, 10.0);
        model_serial.set_state_bounds(q, 0.0, 1e19);
        model_serial.set_control_bounds(rate, 0.0, 5.0);
        model_serial.set_initial_state(q, 10.0);
        model_serial.set_mesh(0.0, 5.0, 25);
    }
    auto ocp_serial = model_serial.build(
        [](const auto& x, const auto& u, const auto& p, auto){
            using T = std::decay_t<decltype(x[0])>; return std::vector<T>{ p[0] - u[0] }; },
        [](const auto& x, const auto& u, const auto& p, auto){
            using T = std::decay_t<decltype(x[0])>; return x[0] + p[1]*u[0]*u[0]; });
    auto compiled_serial = goss::transcription::HermiteSimpson::compile(
        ocp_serial, "sweep_workflow_serial");
    const auto z0_serial = goss::sim::linear_guess(model_serial, compiled_serial.layout);
    goss::solver::IpoptSolver solver_serial;

    // Parallel model + compiled problem (separate compilation unit — distinct name)
    goss::model::Model model_parallel;
    {
        auto q    = model_parallel.add_state("queue_length");
        auto rate = model_parallel.add_control("service_rate");
        model_parallel.add_parameter("arrival_rate", 2.0, 0.0, 10.0);
        model_parallel.add_parameter("cost_weight",  0.1, 0.0, 10.0);
        model_parallel.set_state_bounds(q, 0.0, 1e19);
        model_parallel.set_control_bounds(rate, 0.0, 5.0);
        model_parallel.set_initial_state(q, 10.0);
        model_parallel.set_mesh(0.0, 5.0, 25);
    }
    auto ocp_parallel = model_parallel.build(
        [](const auto& x, const auto& u, const auto& p, auto){
            using T = std::decay_t<decltype(x[0])>; return std::vector<T>{ p[0] - u[0] }; },
        [](const auto& x, const auto& u, const auto& p, auto){
            using T = std::decay_t<decltype(x[0])>; return x[0] + p[1]*u[0]*u[0]; });
    auto compiled_parallel = goss::transcription::HermiteSimpson::compile(
        ocp_parallel, "sweep_workflow_parallel");
    const auto z0_parallel = goss::sim::linear_guess(model_parallel, compiled_parallel.layout);
    goss::solver::IpoptSolver solver_parallel;

    // 3 arrival-rate values × 3 cost-weight values → 9-point Cartesian product.
    const std::vector<std::vector<double>> grid =
        goss::sim::make_grid({{1.0, 2.0, 3.0}, {0.05, 0.1, 0.5}});
    ASSERT_EQ(grid.size(), 9u);

    // Serial oracle
    const goss::sim::SweepResult serial_result = goss::sim::run_sweep_serial(
        *compiled_serial.problem, compiled_serial.validator,
        solver_serial, grid, z0_serial);

    // Parallel result (process pool, 4 workers)
    goss::sim::SweepConfig config; config.max_parallel_workers = 4;
    const goss::sim::SweepResult parallel_result = goss::sim::run_sweep_parallel(
        *compiled_parallel.problem, compiled_parallel.validator,
        solver_parallel, grid, z0_parallel, config);

    ASSERT_EQ(parallel_result.points.size(), serial_result.points.size());
    EXPECT_EQ(parallel_result.num_succeeded(), 9u);

    // Per-index: parameters equal, status equal, objective within 1e-6.
    for (std::size_t point_index = 0; point_index < grid.size(); ++point_index) {
        EXPECT_EQ(parallel_result.points[point_index].parameters,
                  serial_result.points[point_index].parameters)
            << "parameter mismatch at grid index " << point_index;
        EXPECT_EQ(parallel_result.points[point_index].status,
                  serial_result.points[point_index].status)
            << "status mismatch at grid index " << point_index;
        EXPECT_NEAR(parallel_result.points[point_index].objective_value,
                    serial_result.points[point_index].objective_value, 1e-6)
            << "objective mismatch at grid index " << point_index
            << " (parallel=" << parallel_result.points[point_index].objective_value
            << " serial=" << serial_result.points[point_index].objective_value << ")";
    }
}
