// tests/spec/test_specs_json.cpp
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "goss/spec/errors.hpp"
#include "goss/spec/json.hpp"
#include "goss/spec/specs.hpp"

namespace {

goss::spec::RunSpec make_queue_run() {
    goss::spec::RunSpec run;
    run.problem = {"queue", "v1"};
    run.parameters = {{"arrival_rate", 2.0}, {"cost_weight", 0.1}};
    run.discretization.scheme = "hermite_simpson";
    run.discretization.t_initial = 0.0;
    run.discretization.t_final = 5.0;
    run.discretization.num_intervals = 25;
    run.solver.kind = "ipopt";
    run.solver.tolerance = 1e-8;
    run.storage.root = "/tmp/goss";
    run.image_pipeline = "trajectory_overlay";
    run.label = "baseline";
    return run;
}

}  // namespace

TEST(SpecJson, RunSpecRoundTrips) {
    const goss::spec::RunSpec original = make_queue_run();
    const nlohmann::json j = original;
    const auto restored = j.get<goss::spec::RunSpec>();

    EXPECT_EQ(restored.problem, original.problem);
    EXPECT_EQ(restored.parameters, original.parameters);
    EXPECT_EQ(restored.discretization.scheme, original.discretization.scheme);
    EXPECT_EQ(restored.discretization.num_intervals, original.discretization.num_intervals);
    EXPECT_EQ(restored.solver.kind, original.solver.kind);
    EXPECT_EQ(restored.storage.root, original.storage.root);
    EXPECT_EQ(restored.image_pipeline, original.image_pipeline);
    EXPECT_EQ(restored.label, original.label);
}

TEST(SpecJson, CampaignRoundTrips) {
    goss::spec::SweepSpec sweep;
    sweep.base = make_queue_run();
    sweep.axes = {{"arrival_rate", {1.0, 2.0, 3.0}}, {"cost_weight", {0.05, 0.1, 0.5}}};
    sweep.label = "arrival_x_cost";
    goss::spec::CampaignSpec campaign;
    campaign.name = "queue_study";
    campaign.sweeps = {sweep};

    const nlohmann::json j = campaign;
    const auto restored = j.get<goss::spec::CampaignSpec>();
    ASSERT_EQ(restored.sweeps.size(), 1u);
    EXPECT_EQ(restored.name, "queue_study");
    EXPECT_EQ(restored.sweeps[0].axes.size(), 2u);
    EXPECT_EQ(restored.sweeps[0].axes[0].parameter, "arrival_rate");
    EXPECT_EQ(restored.sweeps[0].axes[1].values.size(), 3u);
}

TEST(SpecIdentity, RunIdIsStableAndVolatileFieldsExcluded) {
    const goss::spec::RunSpec a = make_queue_run();
    goss::spec::RunSpec b = a;
    b.storage.root = "/some/other/place";
    b.label = "different label";
    b.image_pipeline = "other_pipeline";

    // Volatile fields must NOT change identity.
    EXPECT_EQ(goss::spec::run_id(a), goss::spec::run_id(b));

    // A meaningful change (a parameter value) MUST change identity.
    goss::spec::RunSpec c = a;
    c.parameters["arrival_rate"] = 2.5;
    EXPECT_NE(goss::spec::run_id(a), goss::spec::run_id(c));

    // A discretization change MUST change identity.
    goss::spec::RunSpec d = a;
    d.discretization.num_intervals = 40;
    EXPECT_NE(goss::spec::run_id(a), goss::spec::run_id(d));

    // 16 lowercase hex chars.
    EXPECT_EQ(goss::spec::run_id(a).size(), 16u);
}

TEST(SweepExpand, ProductOverlaysNamedAxes) {
    goss::spec::SweepSpec sweep;
    sweep.base = make_queue_run();
    sweep.combinator = "product";
    sweep.axes = {{"arrival_rate", {1.0, 2.0, 3.0}}, {"cost_weight", {0.05, 0.1, 0.5}}};

    const auto runs = sweep.expand();
    ASSERT_EQ(runs.size(), 9u);

    // Axis 0 varies slowest (row-major, matching make_grid).
    EXPECT_DOUBLE_EQ(runs.front().parameters.at("arrival_rate"), 1.0);
    EXPECT_DOUBLE_EQ(runs.front().parameters.at("cost_weight"), 0.05);
    EXPECT_DOUBLE_EQ(runs[1].parameters.at("arrival_rate"), 1.0);
    EXPECT_DOUBLE_EQ(runs[1].parameters.at("cost_weight"), 0.1);
    EXPECT_DOUBLE_EQ(runs[3].parameters.at("arrival_rate"), 2.0);
    EXPECT_DOUBLE_EQ(runs.back().parameters.at("arrival_rate"), 3.0);
    EXPECT_DOUBLE_EQ(runs.back().parameters.at("cost_weight"), 0.5);

    // Non-axis base fields are carried through unchanged.
    EXPECT_EQ(runs.back().problem.name, "queue");
    EXPECT_EQ(runs.back().discretization.num_intervals, 25u);
}

TEST(SweepExpand, ZipPairsIndexWise) {
    goss::spec::SweepSpec sweep;
    sweep.base = make_queue_run();
    sweep.combinator = "zip";
    sweep.axes = {{"arrival_rate", {1.0, 2.0, 3.0}}, {"cost_weight", {0.05, 0.1, 0.5}}};

    const auto runs = sweep.expand();
    ASSERT_EQ(runs.size(), 3u);
    EXPECT_DOUBLE_EQ(runs[0].parameters.at("arrival_rate"), 1.0);
    EXPECT_DOUBLE_EQ(runs[0].parameters.at("cost_weight"), 0.05);
    EXPECT_DOUBLE_EQ(runs[2].parameters.at("arrival_rate"), 3.0);
    EXPECT_DOUBLE_EQ(runs[2].parameters.at("cost_weight"), 0.5);
}

TEST(SweepExpand, ZipRejectsUnequalLengths) {
    goss::spec::SweepSpec sweep;
    sweep.base = make_queue_run();
    sweep.combinator = "zip";
    sweep.axes = {{"arrival_rate", {1.0, 2.0}}, {"cost_weight", {0.05}}};
    EXPECT_THROW(sweep.expand(), goss::spec::SpecError);
}

TEST(SweepExpand, RejectsUnknownCombinatorAndEmptyAxis) {
    goss::spec::SweepSpec bad_combinator;
    bad_combinator.base = make_queue_run();
    bad_combinator.combinator = "cross";
    bad_combinator.axes = {{"arrival_rate", {1.0}}};
    EXPECT_THROW(bad_combinator.expand(), goss::spec::SpecError);

    goss::spec::SweepSpec empty_axis;
    empty_axis.base = make_queue_run();
    empty_axis.axes = {{"arrival_rate", {}}};
    EXPECT_THROW(empty_axis.expand(), goss::spec::SpecError);
}

TEST(SweepExpand, NoAxesYieldsBaseRun) {
    goss::spec::SweepSpec sweep;
    sweep.base = make_queue_run();
    const auto runs = sweep.expand();
    ASSERT_EQ(runs.size(), 1u);
    EXPECT_EQ(runs[0].parameters, sweep.base.parameters);
}

TEST(SweepExpandGroups, ZipsWithinGroupAndProductsAcrossGroups) {
    goss::spec::SweepSpec sweep;
    sweep.base = make_queue_run();
    // Group 0: two axes changing together (zip), length 2.
    // Group 1: one axis of length 3.  Expect 2 * 3 = 6 runs.
    goss::spec::AxisGroup group_zip;
    group_zip.axes = {{"arrival_rate", {1.0, 2.0}}, {"cost_weight", {0.1, 0.2}}};
    goss::spec::AxisGroup group_single;
    group_single.axes = {{"cost_weight", {0.05, 0.1, 0.5}}};
    sweep.groups = {group_zip, group_single};

    const auto runs = sweep.expand();
    ASSERT_EQ(runs.size(), 6u);
    // Group 0 varies slowest (its two axes move together); group 1 varies fastest.
    EXPECT_DOUBLE_EQ(runs.front().parameters.at("arrival_rate"), 1.0);
    EXPECT_DOUBLE_EQ(runs.front().parameters.at("cost_weight"), 0.05);
    EXPECT_DOUBLE_EQ(runs[1].parameters.at("arrival_rate"), 1.0);
    EXPECT_DOUBLE_EQ(runs[1].parameters.at("cost_weight"), 0.1);
    EXPECT_DOUBLE_EQ(runs[3].parameters.at("arrival_rate"), 2.0);
    EXPECT_DOUBLE_EQ(runs.back().parameters.at("arrival_rate"), 2.0);
    EXPECT_DOUBLE_EQ(runs.back().parameters.at("cost_weight"), 0.5);

    // "Last group wins": both groups name cost_weight.  Group 1 is processed
    // after group 0, so its value overwrites group 0's zipped value.
    // runs[3] is (group0=row1, group1=row0): group0 sets cost_weight=0.2,
    // then group1 overwrites with 0.05.  Verify group1's value prevails.
    EXPECT_DOUBLE_EQ(runs[3].parameters.at("cost_weight"), 0.05);
}

TEST(SweepExpandGroups, RejectsUnequalLengthAxesWithinGroup) {
    goss::spec::SweepSpec sweep;
    sweep.base = make_queue_run();
    goss::spec::AxisGroup bad;
    bad.axes = {{"arrival_rate", {1.0, 2.0}}, {"cost_weight", {0.1}}};
    sweep.groups = {bad};
    EXPECT_THROW(sweep.expand(), goss::spec::SpecError);
}

TEST(SweepExpandGroups, SingleAxisGroupsReproduceProductGrid) {
    goss::spec::SweepSpec sweep;
    sweep.base = make_queue_run();
    goss::spec::AxisGroup g0; g0.axes = {{"arrival_rate", {1.0, 2.0, 3.0}}};
    goss::spec::AxisGroup g1; g1.axes = {{"cost_weight", {0.05, 0.1, 0.5}}};
    sweep.groups = {g0, g1};
    const auto runs = sweep.expand();
    ASSERT_EQ(runs.size(), 9u);
    EXPECT_DOUBLE_EQ(runs.front().parameters.at("arrival_rate"), 1.0);
    EXPECT_DOUBLE_EQ(runs.front().parameters.at("cost_weight"), 0.05);
    EXPECT_DOUBLE_EQ(runs.back().parameters.at("arrival_rate"), 3.0);
    EXPECT_DOUBLE_EQ(runs.back().parameters.at("cost_weight"), 0.5);
}

TEST(SweepExpandGroups, RejectsEmptyAxisWithinGroup) {
    // A non-first axis in a group with empty values should produce a clear
    // SpecError naming the offending parameter, not an equal-length mismatch.
    goss::spec::SweepSpec sweep;
    sweep.base = make_queue_run();
    goss::spec::AxisGroup bad;
    // First axis is valid; second axis has no values — this is the case that
    // previously produced a misleading "axes in a group must be equal length" error.
    bad.axes = {{"arrival_rate", {1.0, 2.0}}, {"cost_weight", {}}};
    sweep.groups = {bad};
    EXPECT_THROW(sweep.expand(), goss::spec::SpecError);
}

TEST(SpecJson, GroupsRoundTrip) {
    goss::spec::SweepSpec sweep;
    sweep.base = make_queue_run();
    goss::spec::AxisGroup group;
    group.axes = {{"arrival_rate", {1.0, 2.0}}, {"cost_weight", {0.1, 0.2}}};
    sweep.groups = {group};
    sweep.label = "grouped";

    const nlohmann::json j = sweep;
    const auto restored = j.get<goss::spec::SweepSpec>();
    ASSERT_EQ(restored.groups.size(), 1u);
    ASSERT_EQ(restored.groups[0].axes.size(), 2u);
    EXPECT_EQ(restored.groups[0].axes[0].parameter, "arrival_rate");
    EXPECT_EQ(restored.groups[0].axes[1].values.size(), 2u);
}

TEST(SpecJson, LegacySweepWithoutGroupsStillParses) {
    // A sweep JSON produced before `groups` existed must still deserialize.
    nlohmann::json j = {
        {"base", make_queue_run()},
        {"axes", {{{"parameter", "arrival_rate"}, {"values", {1.0, 2.0}}}}},
        {"combinator", "product"},
        {"label", "legacy"}};
    const auto restored = j.get<goss::spec::SweepSpec>();
    EXPECT_TRUE(restored.groups.empty());
    ASSERT_EQ(restored.axes.size(), 1u);
    EXPECT_EQ(restored.combinator, "product");
}

TEST(SpecJson, SweepFromJsonToleratesMissingLabel) {
    // Hand-authored JSON that omits the optional "label" field must not throw.
    nlohmann::json j = {
        {"base", make_queue_run()},
        {"axes", {{{"parameter", "arrival_rate"}, {"values", {1.0, 2.0}}}}},
        {"combinator", "product"}};
    // Verify "label" is absent (defensive: ensure the test is meaningful).
    ASSERT_FALSE(j.contains("label"));
    goss::spec::SweepSpec restored;
    ASSERT_NO_THROW(restored = j.get<goss::spec::SweepSpec>());
    EXPECT_TRUE(restored.label.empty());
}

TEST(SpecJson, RunFromJsonToleratesMissingLabelAndImagePipeline) {
    // Hand-authored JSON that omits the optional "label" and "image_pipeline"
    // fields must not throw; both should default to empty strings.
    nlohmann::json j = make_queue_run();
    j.erase("label");
    j.erase("image_pipeline");
    ASSERT_FALSE(j.contains("label"));
    ASSERT_FALSE(j.contains("image_pipeline"));
    goss::spec::RunSpec restored;
    ASSERT_NO_THROW(restored = j.get<goss::spec::RunSpec>());
    EXPECT_TRUE(restored.label.empty());
    EXPECT_TRUE(restored.image_pipeline.empty());
}
