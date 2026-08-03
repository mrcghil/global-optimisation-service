// tests/sim/test_sweep_parallel.cpp
#include <gtest/gtest.h>
#include <vector>
#include "goss/model/model.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "goss/sim/initial_guess.hpp"
#include "goss/sim/sweep.hpp"

namespace {
goss::transcription::CompiledOcp build_queue(goss::model::Model& model, const char* name) {
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
    return goss::transcription::HermiteSimpson::compile(ocp, name);
}
}  // namespace

TEST(SweepParallel, MatchesSerialResultsInOrder) {
    std::vector<std::vector<double>> grid = {{1.0},{2.0},{3.0},{4.0},{5.0},{6.0}};

    goss::model::Model model_s;
    auto compiled_s = build_queue(model_s, "sweep_par_serial");
    const auto z0_s = goss::sim::linear_guess(model_s, compiled_s.layout);
    goss::solver::IpoptSolver solver_s;
    auto serial = goss::sim::run_sweep_serial(
        *compiled_s.problem, compiled_s.validator, solver_s, grid, z0_s);

    goss::model::Model model_p;
    auto compiled_p = build_queue(model_p, "sweep_par_parallel");
    const auto z0_p = goss::sim::linear_guess(model_p, compiled_p.layout);
    goss::solver::IpoptSolver solver_p;
    goss::sim::SweepConfig config; config.max_parallel_workers = 4;
    auto parallel = goss::sim::run_sweep_parallel(
        *compiled_p.problem, compiled_p.validator, solver_p, grid, z0_p, config);

    ASSERT_EQ(parallel.points.size(), serial.points.size());
    for (std::size_t i = 0; i < grid.size(); ++i) {
        EXPECT_EQ(parallel.points[i].parameters, serial.points[i].parameters);
        EXPECT_EQ(parallel.points[i].status, serial.points[i].status);
        EXPECT_NEAR(parallel.points[i].objective_value,
                    serial.points[i].objective_value, 1e-6);
    }
}

TEST(SweepParallel, InvalidPointsRecordedAtCorrectIndices) {
    std::vector<std::vector<double>> grid = {{1.0},{999.0},{3.0}};  // middle invalid
    goss::model::Model model;
    auto compiled = build_queue(model, "sweep_par_bad");
    const auto z0 = goss::sim::linear_guess(model, compiled.layout);
    goss::solver::IpoptSolver solver;
    goss::sim::SweepConfig config; config.max_parallel_workers = 3;
    auto result = goss::sim::run_sweep_parallel(
        *compiled.problem, compiled.validator, solver, grid, z0, config);

    ASSERT_EQ(result.points.size(), 3u);
    EXPECT_EQ(result.points[0].status, goss::solver::SolverStatus::Success);
    EXPECT_EQ(result.points[1].status, goss::solver::SolverStatus::Failure);
    EXPECT_NE(result.points[1].message.find("arrival_rate"), std::string::npos);
    EXPECT_EQ(result.points[2].status, goss::solver::SolverStatus::Success);
}
