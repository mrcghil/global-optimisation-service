// tests/spec/test_dashboard_export.cpp
// Stage C: the static exporter turns a results folder (run archives + manifests)
// into the dashboard JSON contract.  Compiled only with GOSS_HAVE_HDF5.
#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>
#include "goss/sim/archive.hpp"
#include "goss/sim/dashboard_export.hpp"
#include "goss/spec/executor.hpp"
#include "goss/spec/registry.hpp"
#include "goss/spec/specs.hpp"
#include "queue_fixture.hpp"

using goss::spec::test::build_queue;

namespace {

std::string temp_dir(const std::string& tag) {
    const auto dir = std::filesystem::temp_directory_path() /
                     ("goss_dash_" + tag + "_" + std::to_string(::getpid()));
    std::filesystem::remove_all(dir);
    return dir.string();
}

nlohmann::json read_json(const std::string& path) {
    std::ifstream in(path);
    return nlohmann::json::parse(in);
}

goss::spec::SweepSpec queue_sweep(const std::string& root) {
    goss::spec::RunSpec base;
    base.problem = {"queue", "v1"};
    base.parameters = {{"arrival_rate", 2.0}, {"cost_weight", 0.1}};
    base.discretization.scheme = "hermite_simpson";
    base.discretization.t_initial = 0.0;
    base.discretization.t_final = 5.0;
    base.discretization.num_intervals = 25;
    base.solver.kind = "ipopt";
    base.storage.root = root;

    goss::spec::SweepSpec sweep;
    sweep.base = base;
    sweep.label = "arrival_x_cost";
    sweep.axes = {{"arrival_rate", {1.0, 2.0}}, {"cost_weight", {0.05, 0.1}}};
    return sweep;
}

}  // namespace

TEST(DashboardExport, EmitsContractFromSweep) {
    const std::string root = temp_dir("results");
    const std::string out = temp_dir("out");

    goss::spec::ProblemRegistry registry;
    registry.register_problem({"queue", "v1"}, build_queue);
    const auto sweep = goss::spec::execute_sweep(queue_sweep(root), registry, 4);
    goss::sim::write_sweep(sweep);

    const auto counts = goss::sim::export_dashboard_data(root, out);
    EXPECT_EQ(counts.sweeps, 1u);
    EXPECT_EQ(counts.runs, 4u);

    // index.json lists the sweep with its axes.
    const auto index = read_json(out + "/index.json");
    ASSERT_EQ(index.at("sweeps").size(), 1u);
    EXPECT_EQ(index.at("sweeps")[0].at("slug"), "arrival_x_cost");
    EXPECT_EQ(index.at("sweeps")[0].at("axes").size(), 2u);
    EXPECT_EQ(index.at("sweeps")[0].at("num_runs"), 4u);

    // sweep/<slug>.json carries the run refs.
    const auto sweep_json = read_json(out + "/sweep/arrival_x_cost.json");
    ASSERT_EQ(sweep_json.at("runs").size(), 4u);
    const std::string first_run_id = sweep_json.at("runs")[0].at("run_id");

    // run/<run_id>.json carries the trajectory.
    const auto run_json = read_json(out + "/run/" + first_run_id + ".json");
    EXPECT_EQ(run_json.at("problem"), "queue");
    EXPECT_TRUE(run_json.at("trajectory").at("states").contains("queue_length"));
    EXPECT_GT(run_json.at("trajectory").at("time").size(), 0u);
    EXPECT_EQ(run_json.at("trajectory").at("time").size(), 26u);

    std::filesystem::remove_all(root);
    std::filesystem::remove_all(out);
}

TEST(DashboardExport, EmitsContractFromCampaign) {
    const std::string root = temp_dir("cresults");
    const std::string out = temp_dir("cout");

    goss::spec::ProblemRegistry registry;
    registry.register_problem({"queue", "v1"}, build_queue);
    goss::spec::CampaignSpec campaign;
    campaign.name = "study";
    campaign.sweeps = {queue_sweep(root)};
    const auto archive = goss::spec::execute_campaign(campaign, registry, 4);
    goss::sim::write_campaign(archive);

    const auto counts = goss::sim::export_dashboard_data(root, out);
    EXPECT_EQ(counts.campaigns, 1u);
    EXPECT_EQ(counts.sweeps, 1u);
    EXPECT_EQ(counts.runs, 4u);

    const auto index = read_json(out + "/index.json");
    ASSERT_EQ(index.at("campaigns").size(), 1u);
    EXPECT_EQ(index.at("campaigns")[0].at("name"), "study");
    EXPECT_EQ(index.at("campaigns")[0].at("num_runs"), 4u);

    std::filesystem::remove_all(root);
    std::filesystem::remove_all(out);
}
