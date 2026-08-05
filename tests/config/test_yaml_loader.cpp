// tests/config/test_yaml_loader.cpp
#include <gtest/gtest.h>
#include <cstdio>
#include <fstream>
#include <string>
#include "goss/config/yaml_loader.hpp"
#include "goss/spec/errors.hpp"

namespace {

std::string write_temp_yaml(const std::string& contents) {
    const std::string path =
        std::string(std::tmpnam(nullptr)) + "_goss_test.yaml";
    std::ofstream out(path);
    out << contents;
    out.close();
    return path;
}

}  // namespace

TEST(YamlLoader, RangeTagExpandsInclusiveLinspace) {
    const std::string yaml = R"(
name: t
sweeps:
  - label: s
    base:
      problem: { name: queue, version: v1 }
      parameters: { arrival_rate: 2.0, cost_weight: 0.1 }
    groups:
      - [ { parameter: arrival_rate, values: !range [0.0, 1.0, 5] } ]
)";
    const std::string path = write_temp_yaml(yaml);
    const auto campaign = goss::config::load_campaign_from_yaml(path);
    std::remove(path.c_str());

    ASSERT_EQ(campaign.sweeps.size(), 1u);
    ASSERT_EQ(campaign.sweeps[0].groups.size(), 1u);
    const auto& values = campaign.sweeps[0].groups[0].axes[0].values;
    ASSERT_EQ(values.size(), 5u);
    EXPECT_DOUBLE_EQ(values.front(), 0.0);
    EXPECT_DOUBLE_EQ(values[2], 0.5);
    EXPECT_DOUBLE_EQ(values.back(), 1.0);
}

TEST(YamlLoader, RangeCountOneYieldsSinglePoint) {
    const std::string yaml = R"(
name: t
sweeps:
  - label: s
    base:
      problem: { name: queue, version: v1 }
      parameters: { arrival_rate: 2.0, cost_weight: 0.1 }
    groups:
      - [ { parameter: arrival_rate, values: !range [3.0, 9.0, 1] } ]
)";
    const std::string path = write_temp_yaml(yaml);
    const auto campaign = goss::config::load_campaign_from_yaml(path);
    std::remove(path.c_str());
    const auto& values = campaign.sweeps[0].groups[0].axes[0].values;
    ASSERT_EQ(values.size(), 1u);
    EXPECT_DOUBLE_EQ(values.front(), 3.0);
}

TEST(YamlLoader, ExplicitListValuesAndDefaultsFill) {
    const std::string yaml = R"(
name: queue study
sweeps:
  - label: arrival_x_cost
    base:
      problem: { name: queue, version: v1 }
      parameters: { arrival_rate: 2.0, cost_weight: 0.1 }
      discretization: { scheme: hermite_simpson, t_final: 5.0, num_intervals: 25 }
      solver: { kind: ipopt }
    groups:
      - [ { parameter: arrival_rate, values: [1.0, 2.0, 3.0] } ]
      - [ { parameter: cost_weight,  values: [0.05, 0.1, 0.5] } ]
)";
    const std::string path = write_temp_yaml(yaml);
    const auto campaign = goss::config::load_campaign_from_yaml(path);
    std::remove(path.c_str());

    EXPECT_EQ(campaign.name, "queue study");
    ASSERT_EQ(campaign.sweeps.size(), 1u);
    const auto& sweep = campaign.sweeps[0];
    EXPECT_EQ(sweep.label, "arrival_x_cost");
    ASSERT_EQ(sweep.groups.size(), 2u);
    EXPECT_EQ(sweep.base.problem.name, "queue");
    EXPECT_EQ(sweep.base.problem.version, "v1");
    EXPECT_EQ(sweep.base.discretization.num_intervals, 25u);
    EXPECT_DOUBLE_EQ(sweep.base.discretization.t_final, 5.0);
    EXPECT_EQ(sweep.base.solver.kind, "ipopt");
    // Field not present in YAML falls back to the SolverSpec default.
    EXPECT_EQ(sweep.base.solver.max_iterations, 3000);
    // Whole sweep expands to a 3x3 grid.
    EXPECT_EQ(sweep.expand().size(), 9u);
}

TEST(YamlLoader, RangeWithBadArityThrows) {
    const std::string yaml = R"(
name: t
sweeps:
  - label: s
    base:
      problem: { name: queue, version: v1 }
      parameters: { arrival_rate: 2.0, cost_weight: 0.1 }
    groups:
      - [ { parameter: arrival_rate, values: !range [0.0, 1.0] } ]
)";
    const std::string path = write_temp_yaml(yaml);
    EXPECT_THROW(goss::config::load_campaign_from_yaml(path), goss::spec::SpecError);
    std::remove(path.c_str());
}

TEST(YamlLoader, MissingFileThrows) {
    EXPECT_THROW(
        goss::config::load_campaign_from_yaml("/no/such/goss_config.yaml"),
        goss::spec::SpecError);
}

// I-3: a non-numeric entry in a plain axis values list must throw SpecError,
// not leak a raw YAML::TypeConversion exception.
TEST(YamlLoader, NonNumericValueThrowsSpecError) {
    const std::string yaml = R"(
name: t
sweeps:
  - label: s
    base:
      problem: { name: queue, version: v1 }
    groups:
      - [ { parameter: arrival_rate, values: [1.0, "abc"] } ]
)";
    const std::string path = write_temp_yaml(yaml);
    EXPECT_THROW(goss::config::load_campaign_from_yaml(path), goss::spec::SpecError);
    std::remove(path.c_str());
}

// I-3: omitting `version` from `base.problem` must throw SpecError, not a raw
// YAML::TypeConversion exception from the bare `.as<std::string>()` call.
TEST(YamlLoader, MissingProblemFieldsThrowSpecError) {
    const std::string yaml = R"(
name: t
sweeps:
  - label: s
    base:
      problem: { name: queue }
)";
    const std::string path = write_temp_yaml(yaml);
    EXPECT_THROW(goss::config::load_campaign_from_yaml(path), goss::spec::SpecError);
    std::remove(path.c_str());
}

// I-2: an excessively large !range count must throw SpecError before attempting
// to reserve or allocate the vector.
TEST(YamlLoader, RangeCountTooLargeThrows) {
    const std::string yaml = R"(
name: t
sweeps:
  - label: s
    base:
      problem: { name: queue, version: v1 }
    groups:
      - [ { parameter: arrival_rate, values: !range [0.0, 1.0, 1000000001] } ]
)";
    const std::string path = write_temp_yaml(yaml);
    EXPECT_THROW(goss::config::load_campaign_from_yaml(path), goss::spec::SpecError);
    std::remove(path.c_str());
}
