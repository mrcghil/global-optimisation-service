// tests/spec/test_manifest.cpp
// Stage A: sweep/campaign manifests are the on-disk grouping the dashboard reads
// (replacing the dropped Zarr idea).  Verifies write_sweep/write_campaign emit
// location-independent JSON that references runs LOGICALLY by
// {problem, version, run_id} and records the named axes.
#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>
#include "goss/sim/archive.hpp"
#include "goss/spec/executor.hpp"
#include "goss/spec/registry.hpp"
#include "goss/spec/specs.hpp"
#include "queue_fixture.hpp"

using goss::spec::test::build_queue;

namespace {

std::string temp_root(const std::string& tag) {
    const auto dir = std::filesystem::temp_directory_path() /
                     ("goss_manifest_" + tag + "_" + std::to_string(::getpid()));
    std::filesystem::remove_all(dir);
    return dir.string();
}

goss::spec::RunSpec base_run(const std::string& root) {
    goss::spec::RunSpec run;
    run.problem = {"queue", "v1"};
    run.parameters = {{"arrival_rate", 2.0}, {"cost_weight", 0.1}};
    run.discretization.scheme = "hermite_simpson";
    run.discretization.t_initial = 0.0;
    run.discretization.t_final = 5.0;
    run.discretization.num_intervals = 25;
    run.solver.kind = "ipopt";
    run.storage.root = root;
    return run;
}

goss::spec::ProblemRegistry make_registry() {
    goss::spec::ProblemRegistry registry;
    registry.register_problem({"queue", "v1"}, build_queue);
    return registry;
}

nlohmann::json read_json(const std::string& path) {
    std::ifstream in(path);
    return nlohmann::json::parse(in);
}

}  // namespace

TEST(Manifest, SweepManifestRecordsAxesAndLogicalRunRefs) {
    const std::string root = temp_root("sweep");
    const auto registry = make_registry();

    goss::spec::SweepSpec sweep;
    sweep.base = base_run(root);
    sweep.label = "arrival_x_cost";
    sweep.axes = {{"arrival_rate", {1.0, 2.0, 3.0}}, {"cost_weight", {0.05, 0.1, 0.5}}};

    const auto archive = goss::spec::execute_sweep(sweep, registry, 4);
    const std::string manifest_path = goss::sim::write_sweep(archive);

    ASSERT_TRUE(std::filesystem::exists(manifest_path));
    EXPECT_NE(manifest_path.find("/sweeps/arrival_x_cost/"), std::string::npos);

    const nlohmann::json j = read_json(manifest_path);
    EXPECT_EQ(j.at("combinator"), "product");
    EXPECT_EQ(j.at("num_runs"), 9u);
    EXPECT_EQ(j.at("num_succeeded"), 9u);
    ASSERT_EQ(j.at("axes").size(), 2u);
    EXPECT_EQ(j.at("axes")[0].at("parameter"), "arrival_rate");
    EXPECT_EQ(j.at("axes")[0].at("values").size(), 3u);

    ASSERT_EQ(j.at("runs").size(), 9u);
    const auto& first = j.at("runs")[0];
    // Logical reference — no filesystem path.
    EXPECT_TRUE(first.contains("run_id"));
    EXPECT_EQ(first.at("problem"), "queue");
    EXPECT_EQ(first.at("version"), "v1");
    EXPECT_FALSE(first.contains("path"));
    EXPECT_EQ(first.at("status"), "Success");
    EXPECT_TRUE(first.at("parameters").contains("arrival_rate"));

    // Every referenced run was actually written to the run store.
    for (const auto& run_entry : j.at("runs")) {
        const std::string run_id = run_entry.at("run_id");
        const std::string run_dir =
            root + "/queue/v1/" + run_id;
        EXPECT_TRUE(std::filesystem::exists(run_dir + ".json"))
            << "missing sidecar for " << run_id;
    }

    std::filesystem::remove_all(root);
}

TEST(Manifest, CampaignManifestEmbedsSweepTree) {
    const std::string root = temp_root("campaign");
    const auto registry = make_registry();

    goss::spec::SweepSpec sweep;
    sweep.base = base_run(root);
    sweep.label = "sub_sweep";
    sweep.axes = {{"arrival_rate", {1.0, 2.0}}};

    goss::spec::CampaignSpec campaign;
    campaign.name = "queue study #1";  // exercises slug sanitization
    campaign.sweeps = {sweep};

    const auto archive = goss::spec::execute_campaign(campaign, registry, 2);
    const std::string manifest_path = goss::sim::write_campaign(archive);

    ASSERT_TRUE(std::filesystem::exists(manifest_path));
    // Name is slugged for the directory.
    EXPECT_NE(manifest_path.find("/campaigns/queue_study__1/"), std::string::npos);

    const nlohmann::json j = read_json(manifest_path);
    EXPECT_EQ(j.at("name"), "queue study #1");  // original name preserved in content
    ASSERT_EQ(j.at("sweeps").size(), 1u);
    EXPECT_EQ(j.at("sweeps")[0].at("num_runs"), 2u);
    EXPECT_EQ(j.at("sweeps")[0].at("runs").size(), 2u);

    std::filesystem::remove_all(root);
}

TEST(Manifest, WriteRunSkipsExistingWhenRequested) {
    const std::string root = temp_root("skip");
    const auto registry = make_registry();
    goss::spec::RunSpec spec = base_run(root);
    spec.storage.skip_if_exists = true;

    const auto archive = goss::spec::execute_run(spec, registry);
    const std::string first = goss::sim::write_run(archive);
    ASSERT_TRUE(std::filesystem::exists(first));

    // Second write returns the same path without error (resume path).
    const std::string second = goss::sim::write_run(archive);
    EXPECT_EQ(first, second);

    std::filesystem::remove_all(root);
}
