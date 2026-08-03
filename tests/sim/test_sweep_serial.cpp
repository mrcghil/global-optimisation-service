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
