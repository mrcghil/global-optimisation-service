// tests/spec/test_executor.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "goss/sim/initial_guess.hpp"
#include "goss/sim/sweep.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "goss/spec/errors.hpp"
#include "goss/spec/executor.hpp"
#include "goss/spec/registry.hpp"
#include "goss/spec/specs.hpp"
#include "queue_fixture.hpp"

using goss::spec::test::build_queue;

namespace {

goss::spec::RunSpec base_run() {
    goss::spec::RunSpec run;
    run.problem = {"queue", "v1"};
    run.parameters = {{"arrival_rate", 2.0}, {"cost_weight", 0.1}};
    run.discretization.scheme = "hermite_simpson";
    run.discretization.t_initial = 0.0;
    run.discretization.t_final = 5.0;
    run.discretization.num_intervals = 25;
    run.solver.kind = "ipopt";
    return run;
}

goss::spec::ProblemRegistry make_registry() {
    goss::spec::ProblemRegistry registry;
    registry.register_problem({"queue", "v1"}, build_queue);
    return registry;
}

}  // namespace

TEST(Executor, SingleRunConvergesAndRetainsTrajectory) {
    const auto registry = make_registry();
    const auto archive = goss::spec::execute_run(base_run(), registry);

    EXPECT_EQ(archive.result.status, goss::solver::SolverStatus::Success);
    EXPECT_TRUE(archive.diagnosis.ok);
    EXPECT_EQ(archive.run_id.size(), 16u);
    EXPECT_FALSE(archive.provenance.created_utc.empty());

    // The trajectory is RETAINED (the flat SweepPoint::x path throws it away).
    ASSERT_EQ(archive.trajectory.times.size(), 26u);  // 25 intervals -> 26 nodes
    const auto& q = archive.trajectory.state("queue_length");
    EXPECT_EQ(q.size(), 26u);
    EXPECT_NEAR(q.front(), 10.0, 1e-6);  // pinned initial state
}

TEST(Executor, SweepMatchesRawParallelSweep) {
    const auto registry = make_registry();

    goss::spec::SweepSpec sweep;
    sweep.base = base_run();
    sweep.axes = {{"arrival_rate", {1.0, 2.0, 3.0}}, {"cost_weight", {0.05, 0.1, 0.5}}};

    const auto archive = goss::spec::execute_sweep(sweep, registry, 4);
    ASSERT_EQ(archive.runs.size(), 9u);
    EXPECT_EQ(archive.num_succeeded(), 9u);

    // Oracle: the raw positional parallel sweep over the same grid, ordered as
    // (arrival_rate, cost_weight) to match the model's declared parameter order.
    auto built = build_queue(sweep.base.discretization);
    const auto guess = goss::sim::linear_guess(built.model, built.compiled.layout);
    goss::solver::IpoptSolver solver;
    const auto grid = goss::sim::make_grid({{1.0, 2.0, 3.0}, {0.05, 0.1, 0.5}});
    goss::sim::SweepConfig config; config.max_parallel_workers = 4;
    const auto oracle = goss::sim::run_sweep_parallel(
        *built.compiled.problem, built.compiled.validator, solver, grid, guess, config);

    ASSERT_EQ(oracle.points.size(), archive.runs.size());
    for (std::size_t i = 0; i < archive.runs.size(); ++i) {
        EXPECT_EQ(archive.runs[i].result.status, oracle.points[i].status) << "point " << i;
        EXPECT_NEAR(archive.runs[i].result.objective_value,
                    oracle.points[i].objective_value, 1e-6) << "point " << i;
        // Each succeeded point retained a trajectory.
        EXPECT_EQ(archive.runs[i].trajectory.times.size(), 26u) << "point " << i;
    }

    // run_ids are unique across distinct parameter combinations.
    EXPECT_NE(archive.runs.front().run_id, archive.runs.back().run_id);
}

TEST(Executor, CampaignGroupsSweeps) {
    const auto registry = make_registry();
    goss::spec::SweepSpec sweep;
    sweep.base = base_run();
    sweep.axes = {{"arrival_rate", {1.0, 2.0}}};

    goss::spec::CampaignSpec campaign;
    campaign.name = "queue_study";
    campaign.sweeps = {sweep, sweep};

    const auto archive = goss::spec::execute_campaign(campaign, registry, 2);
    EXPECT_EQ(archive.name, "queue_study");
    ASSERT_EQ(archive.sweeps.size(), 2u);
    EXPECT_EQ(archive.sweeps[0].runs.size(), 2u);
}

TEST(Executor, UnknownParameterNameThrows) {
    const auto registry = make_registry();
    goss::spec::RunSpec run = base_run();
    run.parameters = {{"arrival_rate", 2.0}, {"typo_weight", 0.1}};  // misnamed
    EXPECT_THROW(goss::spec::execute_run(run, registry), goss::spec::SpecError);
}

TEST(Executor, MissingParameterThrows) {
    const auto registry = make_registry();
    goss::spec::RunSpec run = base_run();
    run.parameters = {{"arrival_rate", 2.0}};  // cost_weight omitted
    EXPECT_THROW(goss::spec::execute_run(run, registry), goss::spec::SpecError);
}

TEST(Executor, UnknownSolverKindThrows) {
    const auto registry = make_registry();
    goss::spec::RunSpec run = base_run();
    run.solver.kind = "snopt";  // not supported
    EXPECT_THROW(goss::spec::execute_run(run, registry), goss::spec::SpecError);
}
