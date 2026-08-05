# Queue Sweep YAML Loader + Grouped Axes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author queue-model parameter sweeps in a YAML config (with a `!range` tag and grouped zip/product axes) and run them end-to-end to the results dashboard via a new `goss_run_sweep` CLI.

**Architecture:** Extend the C++ `SweepSpec` with an optional `groups` field (zip within a group, cartesian product across groups) so grouping is first-class and flows unchanged to the dashboard manifest. Add a thin `goss_config` library that parses YAML (via yaml-cpp) into a `CampaignSpec`, resolving the `!range` tag and groups. Add a `goss_run_sweep` CLI that loads the config, registers the queue model, executes the campaign, and exports dashboard JSON. The legacy flat `axes`+`combinator` path is retained for backward-compat only and marked for removal.

**Tech Stack:** C++17, CMake + FetchContent, nlohmann/json, yaml-cpp (new), GoogleTest, HDF5/HighFive (optional).

## Global Constraints

- C++ standard: `CMAKE_CXX_STANDARD 17` (copied from CMakeLists.txt:5).
- New third-party deps are fetched via `FetchContent` at a pinned tag — no system dependency (matches googletest/nlohmann pattern, CMakeLists.txt:8-19).
- Namespaces: spec types live in `goss::spec`; new config loader lives in `goss::config`.
- Config errors and spec-expansion errors are reported by throwing `goss::spec::SpecError` (include/goss/spec/errors.hpp).
- User code preferences (from CLAUDE.local.md): verbose descriptive variable names; type annotations always; comments explain *why* not *what*.
- Doc comments: match the existing house style (`///` block comments describing contracts, as seen throughout include/goss/spec/).
- The queue model is registered as `ProblemKey{"queue", "v1"}` and its builder MUST derive a JIT `.so` name unique to the discretization (registry.hpp:28-33 builder contract).
- HDF5-dependent behavior (dashboard export, `.h5` archives) MUST stay gated behind `#ifdef GOSS_HAVE_HDF5`, so the tree still builds without libhdf5.

---

### Task 1: Add `AxisGroup` + `groups` to the spec model and extend `expand()`

**Files:**
- Modify: `include/goss/spec/specs.hpp` (add `AxisGroup` struct after `Axis` at :81; add `groups` field to `SweepSpec` at :85-96; add deprecation doc comments to `axes`/`combinator`)
- Modify: `src/spec/specs.cpp` (extend `SweepSpec::expand()` :10-52)
- Test: `tests/spec/test_specs_json.cpp` (add grouped-expand tests)

**Interfaces:**
- Consumes: existing `Axis { std::string parameter; std::vector<double> values; }` (specs.hpp:78-81); `goss::sim::make_grid(const std::vector<std::vector<double>>&)` (sim/sweep.hpp:18).
- Produces:
  - `struct goss::spec::AxisGroup { std::vector<Axis> axes; };`
  - `std::vector<AxisGroup> SweepSpec::groups;` (default empty)
  - `SweepSpec::expand()` behavior: when `groups` non-empty, each group is zipped internally (all axes in a group must share length else `SpecError`), groups combined by cartesian product; parameter overlay onto `base.parameters` by name. When `groups` empty, behavior is unchanged (legacy `axes`+`combinator`).

- [ ] **Step 1: Write failing tests for the grouped expand path**

Add to `tests/spec/test_specs_json.cpp` (uses the existing `make_queue_run()` helper in that file):

```cpp
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
```

- [ ] **Step 2: Run tests to verify they fail to compile**

Run: `cmake --build build --target goss_spec_tests`
Expected: FAIL — `AxisGroup` and `SweepSpec::groups` do not exist yet.

- [ ] **Step 3: Add `AxisGroup` struct and `groups` field in specs.hpp**

Insert after the `Axis` struct (currently specs.hpp:81):

```cpp
/// A group of axes whose values advance together (index-wise "zip"). All axes in
/// a group MUST have equal-length `values`. Groups are the building block of a
/// grid: axes zip WITHIN a group, and groups are combined by cartesian PRODUCT.
struct AxisGroup {
    std::vector<Axis> axes;
};
```

In `SweepSpec` (specs.hpp:85-96), add the `groups` field and mark the legacy fields. Replace the struct body's field declarations with:

```cpp
struct SweepSpec {
    RunSpec base;

    /// Grouped axes: zip within a group, cartesian product across groups. This is
    /// the preferred way to describe a sweep. When non-empty, `axes`/`combinator`
    /// below are ignored.
    std::vector<AxisGroup> groups;

    /// DEPRECATED — backward-compat only; slated for removal once `groups` is
    /// validated as the superior model (a single group reproduces "zip",
    /// single-axis groups reproduce "product"). Do not add new callers.
    std::vector<Axis> axes;
    /// DEPRECATED — backward-compat only (see `axes`). "product" | "zip".
    std::string combinator = "product";

    std::string label;

    /// Expands to one concrete RunSpec per parameter combination. If `groups` is
    /// non-empty, axes zip within each group (equal-length required) and groups
    /// combine by cartesian product. Otherwise the legacy `axes` + `combinator`
    /// path is used. Throws SpecError on an empty axis, unequal-length axes in a
    /// group (or a zip), or an unknown combinator.
    std::vector<RunSpec> expand() const;
};
```

- [ ] **Step 4: Extend `SweepSpec::expand()` in specs.cpp**

Replace the body of `expand()` (specs.cpp:10-52) so it branches on `groups`. Keep the existing legacy branch intact for the empty-`groups` case:

```cpp
std::vector<RunSpec> SweepSpec::expand() const {
    // Preferred path: grouped axes (zip within a group, product across groups).
    if (!groups.empty()) {
        // Each group zips its axes into a list of per-combination rows, and
        // contributes one "meta-axis" whose entries are those rows. We then take
        // the cartesian product of the groups' row-indices via make_grid.
        std::vector<std::vector<std::vector<double>>> group_rows;  // group -> rows -> axis values
        std::vector<std::vector<std::string>> group_param_names;   // group -> axis parameter names
        group_rows.reserve(groups.size());
        group_param_names.reserve(groups.size());

        for (const AxisGroup& group : groups) {
            if (group.axes.empty())
                throw SpecError("SweepSpec::expand: an axis group is empty");
            const std::size_t length = group.axes.front().values.size();
            if (length == 0)
                throw SpecError("SweepSpec::expand: axis '" +
                                group.axes.front().parameter + "' is empty");
            std::vector<std::string> names;
            names.reserve(group.axes.size());
            for (const Axis& axis : group.axes) {
                if (axis.values.size() != length)
                    throw SpecError(
                        "SweepSpec::expand: axes in a group must be equal length");
                names.push_back(axis.parameter);
            }
            std::vector<std::vector<double>> rows;
            rows.reserve(length);
            for (std::size_t row = 0; row < length; ++row) {
                std::vector<double> values;
                values.reserve(group.axes.size());
                for (const Axis& axis : group.axes) values.push_back(axis.values[row]);
                rows.push_back(std::move(values));
            }
            group_rows.push_back(std::move(rows));
            group_param_names.push_back(std::move(names));
        }

        // Build integer index-axes (one entry per row in each group) and product
        // them with the shared grid algorithm; group 0 varies slowest.
        std::vector<std::vector<double>> index_axes;
        index_axes.reserve(group_rows.size());
        for (const auto& rows : group_rows) {
            std::vector<double> indices(rows.size());
            for (std::size_t i = 0; i < rows.size(); ++i)
                indices[i] = static_cast<double>(i);
            index_axes.push_back(std::move(indices));
        }
        const std::vector<std::vector<double>> index_combos =
            goss::sim::make_grid(index_axes);

        std::vector<RunSpec> runs;
        runs.reserve(index_combos.size());
        for (const std::vector<double>& index_combo : index_combos) {
            RunSpec run = base;
            for (std::size_t g = 0; g < groups.size(); ++g) {
                const std::size_t row_index =
                    static_cast<std::size_t>(index_combo[g]);
                const std::vector<double>& row = group_rows[g][row_index];
                for (std::size_t a = 0; a < group_param_names[g].size(); ++a)
                    run.parameters[group_param_names[g][a]] = row[a];
            }
            runs.push_back(std::move(run));
        }
        return runs;
    }

    // ---- Legacy path (DEPRECATED, backward-compat only): flat axes + combinator.
    if (axes.empty())
        return {base};  // a degenerate sweep with no axes is just the base run

    for (const Axis& axis : axes)
        if (axis.values.empty())
            throw SpecError("SweepSpec::expand: axis '" + axis.parameter + "' is empty");

    std::vector<std::vector<double>> combos;
    if (combinator == "product") {
        std::vector<std::vector<double>> axis_values;
        axis_values.reserve(axes.size());
        for (const Axis& axis : axes) axis_values.push_back(axis.values);
        combos = goss::sim::make_grid(axis_values);
    } else if (combinator == "zip") {
        const std::size_t length = axes.front().values.size();
        for (const Axis& axis : axes)
            if (axis.values.size() != length)
                throw SpecError("SweepSpec::expand: zip requires equal-length axes");
        combos.reserve(length);
        for (std::size_t row = 0; row < length; ++row) {
            std::vector<double> combo;
            combo.reserve(axes.size());
            for (const Axis& axis : axes) combo.push_back(axis.values[row]);
            combos.push_back(std::move(combo));
        }
    } else {
        throw SpecError("SweepSpec::expand: unknown combinator '" + combinator +
                        "' (expected 'product' or 'zip')");
    }

    std::vector<RunSpec> runs;
    runs.reserve(combos.size());
    for (const std::vector<double>& combo : combos) {
        RunSpec run = base;
        for (std::size_t a = 0; a < axes.size(); ++a)
            run.parameters[axes[a].parameter] = combo[a];
        runs.push_back(std::move(run));
    }
    return runs;
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cmake --build build --target goss_spec_tests && ctest --test-dir build -R SpecJson\|SweepExpand --output-on-failure`
Expected: PASS — the new grouped tests plus all pre-existing `SweepExpand*` legacy tests.

- [ ] **Step 6: Commit**

```bash
git add include/goss/spec/specs.hpp src/spec/specs.cpp tests/spec/test_specs_json.cpp
git commit -m "feat(spec): grouped sweep axes (zip within, product across)"
```

---

### Task 2: JSON (de)serialization for `AxisGroup` + `groups`

**Files:**
- Modify: `include/goss/spec/json.hpp` (declare `AxisGroup` to/from_json at :26)
- Modify: `src/spec/json.cpp` (define `AxisGroup` to/from_json; extend `SweepSpec` to/from_json :96-107)
- Test: `tests/spec/test_specs_json.cpp` (round-trip + legacy-parse tests)

**Interfaces:**
- Consumes: `goss::spec::AxisGroup`, `SweepSpec::groups` (from Task 1); existing `to_json`/`from_json(Axis)` (json.cpp:88-94).
- Produces: `to_json`/`from_json` overloads for `AxisGroup`; `SweepSpec` serialization that includes `"groups"` and reads it with a default when the key is absent.

- [ ] **Step 1: Write failing tests for groups JSON round-trip + legacy parse**

Add to `tests/spec/test_specs_json.cpp`:

```cpp
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cmake --build build --target goss_spec_tests`
Expected: FAIL — no `AxisGroup` serialization; `from_json(SweepSpec)` calls `j.at("groups")` does not exist yet / groups not populated.

- [ ] **Step 3: Declare `AxisGroup` (de)serialization in json.hpp**

Add after the `Axis` declarations (json.hpp:26):

```cpp
void to_json(nlohmann::json& j, const AxisGroup& v);
void from_json(const nlohmann::json& j, AxisGroup& v);
```

- [ ] **Step 4: Define serialization in json.cpp**

Add after the `Axis` to/from_json block (json.cpp:94):

```cpp
void to_json(json& j, const AxisGroup& v) {
    j = json{{"axes", v.axes}};
}
void from_json(const json& j, AxisGroup& v) {
    j.at("axes").get_to(v.axes);
}
```

Replace `SweepSpec` to/from_json (json.cpp:96-107) with a version that includes `groups` and reads it defensively:

```cpp
void to_json(json& j, const SweepSpec& v) {
    j = json{{"base", v.base},
             {"groups", v.groups},
             {"axes", v.axes},
             {"combinator", v.combinator},
             {"label", v.label}};
}
void from_json(const json& j, SweepSpec& v) {
    j.at("base").get_to(v.base);
    // `groups` is the preferred field but may be absent in older configs.
    if (j.contains("groups")) j.at("groups").get_to(v.groups);
    // `axes`/`combinator` are legacy and optional.
    if (j.contains("axes")) j.at("axes").get_to(v.axes);
    if (j.contains("combinator")) j.at("combinator").get_to(v.combinator);
    j.at("label").get_to(v.label);
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cmake --build build --target goss_spec_tests && ctest --test-dir build -R SpecJson --output-on-failure`
Expected: PASS — including the pre-existing `CampaignRoundTrips` test.

- [ ] **Step 6: Commit**

```bash
git add include/goss/spec/json.hpp src/spec/json.cpp tests/spec/test_specs_json.cpp
git commit -m "feat(spec): JSON (de)serialization for grouped sweep axes"
```

---

### Task 3: `goss_config` YAML loader library (`!range` + groups)

**Files:**
- Create: `include/goss/config/yaml_loader.hpp`
- Create: `src/config/yaml_loader.cpp`
- Modify: `CMakeLists.txt` (FetchContent yaml-cpp near :19; `add_library(goss_config …)` and `goss_config_tests` after the spec section ~:303)
- Test: `tests/config/test_yaml_loader.cpp`

**Interfaces:**
- Consumes: `goss::spec::CampaignSpec`, `SweepSpec`, `AxisGroup`, `Axis`, `RunSpec` and their defaults (specs.hpp); `goss::spec::SpecError` (spec/errors.hpp); yaml-cpp `YAML::Node`, `YAML::LoadFile`.
- Produces:
  - `goss::config::CampaignSpec load_campaign_from_yaml(const std::string& path);` returning `goss::spec::CampaignSpec`.
  - `!range [start, stop, count]` resolves to a `count`-point inclusive linspace on any `values` node (`count >= 1`; `count == 1` → `[start]`).
  - Helper (internal linkage) `std::vector<double> resolve_values(const YAML::Node& node);` — not part of the public header.

- [ ] **Step 1: Add yaml-cpp via FetchContent in CMakeLists.txt**

After the nlohmann_json block (CMakeLists.txt:19), add:

```cmake
# yaml-cpp — backs the goss::config YAML sweep loader. Header+static, fetched at
# a pinned tag so there is no system dependency (matches the json/gtest pattern).
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
FetchContent_Declare(yaml_cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 0.8.0)
FetchContent_MakeAvailable(yaml_cpp)
```

- [ ] **Step 2: Write the public header**

Create `include/goss/config/yaml_loader.hpp`:

```cpp
// include/goss/config/yaml_loader.hpp
#pragma once
#include <string>
#include "goss/spec/specs.hpp"

namespace goss::config {

/// Loads a CampaignSpec from a YAML file. Supports the `!range [start, stop,
/// count]` tag on any `values` node (inclusive linspace of `count` points,
/// count >= 1) and grouped axes under `sweeps[].groups` (zip within a group,
/// product across groups). Fields omitted in YAML fall back to spec defaults.
///
/// Throws goss::spec::SpecError on: missing file, malformed YAML, a `!range`
/// with the wrong arity or count < 1, or non-numeric axis values.
goss::spec::CampaignSpec load_campaign_from_yaml(const std::string& path);

}  // namespace goss::config
```

- [ ] **Step 3: Write the failing tests**

Create `tests/config/test_yaml_loader.cpp`. It writes YAML to a temp file, loads it, and asserts. Use a small helper to write the temp file.

```cpp
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
```

- [ ] **Step 4: Implement the loader**

Create `src/config/yaml_loader.cpp`. Parses defensively, resolves `!range`, maps YAML onto spec structs field-by-field with defaults preserved.

```cpp
// src/config/yaml_loader.cpp
#include "goss/config/yaml_loader.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "goss/spec/errors.hpp"

namespace goss::config {
namespace {

using goss::spec::SpecError;

// Resolves a `values` node into an explicit vector<double>. Supports either a
// plain YAML sequence of numbers or the `!range [start, stop, count]` tag, which
// expands to a `count`-point inclusive linspace (count >= 1; count == 1 yields
// [start]).
std::vector<double> resolve_values(const YAML::Node& node) {
    if (!node) throw SpecError("yaml_loader: axis is missing 'values'");
    if (node.Tag() == "!range") {
        if (!node.IsSequence() || node.size() != 3)
            throw SpecError(
                "yaml_loader: !range expects [start, stop, count]");
        const double start = node[0].as<double>();
        const double stop = node[1].as<double>();
        const long count = node[2].as<long>();
        if (count < 1)
            throw SpecError("yaml_loader: !range count must be >= 1");
        std::vector<double> values;
        values.reserve(static_cast<std::size_t>(count));
        if (count == 1) {
            values.push_back(start);
            return values;
        }
        const double step = (stop - start) / static_cast<double>(count - 1);
        for (long i = 0; i < count; ++i)
            values.push_back(start + step * static_cast<double>(i));
        values.back() = stop;  // pin the endpoint against float drift
        return values;
    }
    if (!node.IsSequence())
        throw SpecError("yaml_loader: axis 'values' must be a list or !range");
    std::vector<double> values;
    values.reserve(node.size());
    for (const YAML::Node& element : node) values.push_back(element.as<double>());
    return values;
}

goss::spec::Axis parse_axis(const YAML::Node& node) {
    goss::spec::Axis axis;
    if (!node["parameter"])
        throw SpecError("yaml_loader: axis is missing 'parameter'");
    axis.parameter = node["parameter"].as<std::string>();
    axis.values = resolve_values(node["values"]);
    return axis;
}

goss::spec::AxisGroup parse_group(const YAML::Node& node) {
    if (!node.IsSequence())
        throw SpecError("yaml_loader: each group must be a list of axes");
    goss::spec::AxisGroup group;
    group.axes.reserve(node.size());
    for (const YAML::Node& axis_node : node)
        group.axes.push_back(parse_axis(axis_node));
    return group;
}

void parse_problem(const YAML::Node& node, goss::spec::ProblemKey& key) {
    if (!node) throw SpecError("yaml_loader: run is missing 'problem'");
    key.name = node["name"].as<std::string>();
    key.version = node["version"].as<std::string>();
}

void parse_discretization(const YAML::Node& node,
                          goss::spec::DiscretizationSpec& d) {
    if (!node) return;  // keep defaults
    if (node["scheme"]) d.scheme = node["scheme"].as<std::string>();
    if (node["t_initial"]) d.t_initial = node["t_initial"].as<double>();
    if (node["t_final"]) d.t_final = node["t_final"].as<double>();
    if (node["num_intervals"])
        d.num_intervals = node["num_intervals"].as<std::size_t>();
    if (node["lgl_degree"]) d.lgl_degree = node["lgl_degree"].as<std::size_t>();
}

void parse_solver(const YAML::Node& node, goss::spec::SolverSpec& s) {
    if (!node) return;
    if (node["kind"]) s.kind = node["kind"].as<std::string>();
    if (node["tolerance"]) s.tolerance = node["tolerance"].as<double>();
    if (node["max_iterations"])
        s.max_iterations = node["max_iterations"].as<int>();
    if (node["print_level"]) s.print_level = node["print_level"].as<int>();
    if (node["max_evaluations"])
        s.max_evaluations = node["max_evaluations"].as<int>();
    if (node["xtol_rel"]) s.xtol_rel = node["xtol_rel"].as<double>();
}

void parse_storage(const YAML::Node& node, goss::spec::StorageSpec& st) {
    if (!node) return;
    if (node["root"]) st.root = node["root"].as<std::string>();
    if (node["skip_if_exists"])
        st.skip_if_exists = node["skip_if_exists"].as<bool>();
}

goss::spec::RunSpec parse_base(const YAML::Node& node) {
    if (!node) throw SpecError("yaml_loader: sweep is missing 'base'");
    goss::spec::RunSpec run;
    parse_problem(node["problem"], run.problem);
    if (node["parameters"])
        for (const auto& kv : node["parameters"])
            run.parameters[kv.first.as<std::string>()] = kv.second.as<double>();
    parse_discretization(node["discretization"], run.discretization);
    parse_solver(node["solver"], run.solver);
    parse_storage(node["storage"], run.storage);
    if (node["label"]) run.label = node["label"].as<std::string>();
    if (node["image_pipeline"])
        run.image_pipeline = node["image_pipeline"].as<std::string>();
    return run;
}

goss::spec::SweepSpec parse_sweep(const YAML::Node& node) {
    goss::spec::SweepSpec sweep;
    sweep.base = parse_base(node["base"]);
    if (node["label"]) sweep.label = node["label"].as<std::string>();
    if (node["groups"]) {
        if (!node["groups"].IsSequence())
            throw SpecError("yaml_loader: 'groups' must be a list");
        for (const YAML::Node& group_node : node["groups"])
            sweep.groups.push_back(parse_group(group_node));
    }
    return sweep;
}

}  // namespace

goss::spec::CampaignSpec load_campaign_from_yaml(const std::string& path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const YAML::Exception& e) {
        throw SpecError("yaml_loader: could not load '" + path + "': " + e.what());
    }
    goss::spec::CampaignSpec campaign;
    if (root["name"]) campaign.name = root["name"].as<std::string>();
    if (!root["sweeps"] || !root["sweeps"].IsSequence())
        throw SpecError("yaml_loader: config must have a 'sweeps' list");
    for (const YAML::Node& sweep_node : root["sweeps"])
        campaign.sweeps.push_back(parse_sweep(sweep_node));
    return campaign;
}

}  // namespace goss::config
```

- [ ] **Step 5: Wire the library and its test target in CMakeLists.txt**

After the spec exporter section (CMakeLists.txt:~303, after the `goss_dashboard_export` block), add:

```cmake
# ---- Config / YAML sweep loader ----
# Parses a YAML campaign (with the !range tag and grouped axes) into a
# goss::spec::CampaignSpec.  Depends only on the spec types + yaml-cpp.
add_library(goss_config STATIC src/config/yaml_loader.cpp)
target_include_directories(goss_config PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(goss_config PUBLIC goss_spec yaml-cpp::yaml-cpp)

add_executable(goss_config_tests tests/config/test_yaml_loader.cpp)
target_include_directories(goss_config_tests PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(goss_config_tests PRIVATE
  goss_config goss_spec goss_sim_impl goss_model goss_transcription goss_nlp
  goss_ad goss_ad_impl goss_solver goss_ipopt_iface goss_nlopt_iface cppadcg
  nlohmann_json::nlohmann_json
  $<$<BOOL:${CPPAD_LIB}>:${CPPAD_LIB}> GTest::gtest_main)
gtest_discover_tests(goss_config_tests)
```

Note: yaml-cpp 0.8.0 exports the target `yaml-cpp::yaml-cpp`. If configuration errors on that name, fall back to `yaml-cpp` (older alias).

- [ ] **Step 6: Configure and build the new test target**

Run: `cmake -S . -B build && cmake --build build --target goss_config_tests`
Expected: builds; yaml-cpp is fetched on first configure.

- [ ] **Step 7: Run the loader tests**

Run: `ctest --test-dir build -R YamlLoader --output-on-failure`
Expected: PASS — all five `YamlLoader` tests.

- [ ] **Step 8: Commit**

```bash
git add include/goss/config/yaml_loader.hpp src/config/yaml_loader.cpp \
  tests/config/test_yaml_loader.cpp CMakeLists.txt
git commit -m "feat(config): YAML sweep loader with !range tag and grouped axes"
```

---

### Task 4: Shared queue-model builder header

**Files:**
- Create: `tools/dashboard_export/queue_model.hpp`
- Modify: `tools/dashboard_export/gen_sample.cpp` (replace inline `build_queue` :19-41 with the shared header)

**Interfaces:**
- Consumes: `goss::model::Model`, `goss::spec::DiscretizationSpec`, `goss::spec::BuiltProblem`, `goss::spec::compile_dispatch` (compile_dispatch.hpp).
- Produces: `goss::spec::BuiltProblem goss_tools::build_queue(const goss::spec::DiscretizationSpec& d);` — the queue model builder, registered under `{"queue","v1"}`.

- [ ] **Step 1: Extract the builder into a header**

Create `tools/dashboard_export/queue_model.hpp` (verbatim logic from gen_sample.cpp:19-41, wrapped in a `goss_tools` namespace):

```cpp
// tools/dashboard_export/queue_model.hpp
#pragma once
#include <string>
#include <utility>
#include <vector>

#include "goss/model/model.hpp"
#include "goss/spec/compile_dispatch.hpp"
#include "goss/spec/registry.hpp"

namespace goss_tools {

/// Builds the demo "queue" optimal-control problem at a given discretization.
/// State: queue_length (init 10). Control: service_rate [0,5].
/// Params: arrival_rate (default 2, [0,10]), cost_weight (default 0.1, [0,10]).
/// Dynamics dq/dt = arrival_rate - service_rate; objective
/// integral(queue_length + cost_weight * service_rate^2).
inline goss::spec::BuiltProblem build_queue(
        const goss::spec::DiscretizationSpec& d) {
    goss::model::Model m;
    auto q = m.add_state("queue_length");
    auto r = m.add_control("service_rate");
    m.add_parameter("arrival_rate", 2.0, 0.0, 10.0);
    m.add_parameter("cost_weight", 0.1, 0.0, 10.0);
    m.set_state_bounds(q, 0.0, 1e19);
    m.set_control_bounds(r, 0.0, 5.0);
    m.set_initial_state(q, 10.0);
    m.set_mesh(d.t_initial, d.t_final, d.num_intervals);
    auto ocp = m.build(
        [](const auto& x, const auto& u, const auto& p, auto) {
            using T = std::decay_t<decltype(x[0])>;
            return std::vector<T>{p[0] - u[0]};
        },
        [](const auto& x, const auto& u, const auto& p, auto) {
            using T = std::decay_t<decltype(x[0])>;
            return x[0] + p[1] * u[0] * u[0];
        });
    auto c = goss::spec::compile_dispatch(
        ocp, d, "gensample_" + std::to_string(d.num_intervals));
    auto mesh = ocp.mesh;
    return goss::spec::BuiltProblem{std::move(m), std::move(c), mesh, d.scheme};
}

}  // namespace goss_tools
```

- [ ] **Step 2: Point gen_sample.cpp at the shared header**

In `tools/dashboard_export/gen_sample.cpp`: delete the local `build_queue` (:19-41) and its now-unneeded includes for model/compile_dispatch if still used elsewhere keep them; add `#include "queue_model.hpp"`; change the registration line (:49) to `reg.register_problem({"queue", "v1"}, goss_tools::build_queue);`.

- [ ] **Step 3: Sanity-check gen_sample still compiles (only when HDF5 present)**

Run: `cmake --build build --target goss_dashboard_export 2>/dev/null; echo "gen_sample is not a build target — verifying header compiles via the CLI in Task 5 instead"`
Expected: no error from this step (gen_sample.cpp is not a CMake target; the header is validated by Task 5 which includes it).

- [ ] **Step 4: Commit**

```bash
git add tools/dashboard_export/queue_model.hpp tools/dashboard_export/gen_sample.cpp
git commit -m "refactor(tools): extract shared queue-model builder header"
```

---

### Task 5: `goss_run_sweep` CLI

**Files:**
- Create: `tools/run_sweep/main.cpp`
- Modify: `CMakeLists.txt` (add `goss_run_sweep` executable after the `goss_config` block)

**Interfaces:**
- Consumes: `goss::config::load_campaign_from_yaml` (Task 3); `goss_tools::build_queue` (Task 4); `goss::spec::ProblemRegistry`, `execute_campaign` (executor.hpp:75); `goss::sim::write_campaign` (archive.hpp:59); `goss::sim::export_dashboard_data` (dashboard_export.hpp, HDF5-gated).
- Produces: an executable `goss_run_sweep <config.yaml> [out_data_dir] [max_workers]`.

- [ ] **Step 1: Write the CLI**

Create `tools/run_sweep/main.cpp`:

```cpp
// tools/run_sweep/main.cpp
//
// Standalone executable: load a YAML campaign, run every sweep, write result
// manifests, and (with HDF5) export the dashboard JSON contract.
//
// Usage: goss_run_sweep <config.yaml> [out_data_dir] [max_workers]
//   out_data_dir defaults to "dashboard-data"; max_workers defaults to 0
//   (hardware concurrency). Results manifests go to the storage.root in the
//   config (default ./goss-results).
#include <cstdlib>
#include <iostream>
#include <string>

#include "goss/config/yaml_loader.hpp"
#include "goss/sim/archive.hpp"
#include "goss/sim/errors.hpp"
#include "goss/spec/errors.hpp"
#include "goss/spec/executor.hpp"
#include "goss/spec/registry.hpp"
#include "queue_model.hpp"

#ifdef GOSS_HAVE_HDF5
#include "goss/sim/dashboard_export.hpp"
#endif

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage: goss_run_sweep <config.yaml> [out_data_dir] "
                     "[max_workers]\n";
        return EXIT_FAILURE;
    }
    const std::string config_path = argv[1];
    const std::string out_data_dir = argc > 2 ? argv[2] : "dashboard-data";
    const std::size_t max_workers =
        argc > 3 ? static_cast<std::size_t>(std::strtoul(argv[3], nullptr, 10)) : 0;

    try {
        const goss::spec::CampaignSpec campaign =
            goss::config::load_campaign_from_yaml(config_path);

        goss::spec::ProblemRegistry registry;
        registry.register_problem({"queue", "v1"}, goss_tools::build_queue);

        const goss::spec::CampaignArchive archive =
            goss::spec::execute_campaign(campaign, registry, max_workers);
        const std::string manifest_path = goss::sim::write_campaign(archive);

        std::size_t total_runs = 0;
        std::size_t total_succeeded = 0;
        for (const goss::spec::SweepArchive& sweep : archive.sweeps) {
            total_runs += sweep.runs.size();
            total_succeeded += sweep.num_succeeded();
        }
        std::cout << "ran campaign '" << archive.name << "': " << total_succeeded
                  << "/" << total_runs << " runs succeeded\n"
                  << "manifest: " << manifest_path << "\n";

#ifdef GOSS_HAVE_HDF5
        const std::string results_root =
            goss::sim::resolve_root(campaign.sweeps.empty()
                                        ? goss::spec::StorageSpec{}
                                        : campaign.sweeps.front().base.storage);
        const goss::sim::ExportCounts counts =
            goss::sim::export_dashboard_data(results_root, out_data_dir);
        std::cout << "exported dashboard data to '" << out_data_dir << "': "
                  << counts.campaigns << " campaigns, " << counts.sweeps
                  << " sweeps, " << counts.runs << " runs\n";
#else
        std::cout << "(built without HDF5 — skipped dashboard export; manifests "
                     "written)\n";
        (void)out_data_dir;
#endif
        return EXIT_SUCCESS;
    } catch (const goss::spec::SpecError& e) {
        std::cerr << "goss_run_sweep: config/spec error: " << e.what() << "\n";
        return EXIT_FAILURE;
    } catch (const goss::sim::SimError& e) {
        std::cerr << "goss_run_sweep: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
```

- [ ] **Step 2: Wire the executable in CMakeLists.txt**

After the `goss_config` block (from Task 3), add:

```cmake
# ---- YAML sweep runner (standalone tool) ----
# Loads a YAML campaign and runs it end-to-end: manifests always, dashboard JSON
# when HDF5 is available.  Includes the shared queue-model builder header.
add_executable(goss_run_sweep tools/run_sweep/main.cpp)
target_include_directories(goss_run_sweep PRIVATE
  ${CMAKE_SOURCE_DIR}/tools/dashboard_export)
target_link_libraries(goss_run_sweep PRIVATE
  goss_config goss_spec goss_sim_impl goss_model goss_transcription goss_nlp
  goss_ad goss_ad_impl goss_solver goss_ipopt_iface goss_nlopt_iface cppadcg
  nlohmann_json::nlohmann_json
  $<$<BOOL:${CPPAD_LIB}>:${CPPAD_LIB}>)
```

- [ ] **Step 3: Build the CLI**

Run: `cmake -S . -B build && cmake --build build --target goss_run_sweep`
Expected: links cleanly (this also validates `queue_model.hpp` from Task 4 compiles).

- [ ] **Step 4: Commit**

```bash
git add tools/run_sweep/main.cpp CMakeLists.txt
git commit -m "feat(tools): goss_run_sweep CLI runs a YAML campaign to the dashboard"
```

---

### Task 6: The queue sweep config + end-to-end verification

**Files:**
- Create: `configs/queue_sweep.yaml`

**Interfaces:**
- Consumes: `goss_run_sweep` (Task 5); the loader + grouped expand (Tasks 1-3).
- Produces: a runnable config; verified `goss-results/` manifests + dashboard `data/` JSON.

- [ ] **Step 1: Write the config**

Create `configs/queue_sweep.yaml`:

```yaml
# A 3x3 sweep over the queue model: arrival_rate x cost_weight = 9 runs.
# Two single-axis groups → cartesian product grid. arrival_rate uses the
# !range tag (3 inclusive points 1..3); cost_weight is an explicit list.
name: queue study
sweeps:
  - label: arrival_x_cost
    base:
      problem: { name: queue, version: v1 }
      parameters: { arrival_rate: 2.0, cost_weight: 0.1 }
      discretization: { scheme: hermite_simpson, t_initial: 0.0, t_final: 5.0, num_intervals: 25 }
      solver: { kind: ipopt }
      storage: { root: "goss-results" }
    groups:
      - [ { parameter: arrival_rate, values: !range [1.0, 3.0, 3] } ]
      - [ { parameter: cost_weight,  values: [0.05, 0.1, 0.5] } ]
```

- [ ] **Step 2: Run the sweep end-to-end**

Run: `./build/goss_run_sweep configs/queue_sweep.yaml dashboard-data 4`
Expected: prints "ran campaign 'queue study': 9/9 runs succeeded" (or close — some points may not converge, which is reported, not fatal), a manifest path, and — if HDF5 is present — a dashboard export line.

- [ ] **Step 3: Verify the output artifacts exist**

Run: `ls goss-results && echo "---" && find goss-results -name "campaign.json" && echo "---" && ls dashboard-data 2>/dev/null && find dashboard-data -name "index.json" 2>/dev/null`
Expected: a `campaigns/queue_study/campaign.json` manifest; with HDF5, a `dashboard-data/index.json` plus `sweep/` and `run/` subdirs.

- [ ] **Step 4: Verify the grid reached the dashboard contract (HDF5 builds)**

Run: `test -f dashboard-data/index.json && cat dashboard-data/index.json | head -40 || echo "no HDF5: inspect goss-results/campaigns/queue_study/campaign.json instead"`
Expected: the sweep summary lists 9 runs across the two axes (`arrival_rate`, `cost_weight`).

- [ ] **Step 5: (Optional, manual) View in the Astro dashboard**

Copy or symlink `dashboard-data` into the dashboard's served `/data` path (`dashboard/public/data`, matching the `PUBLIC_GOSS_DATA_BASE` default), then run the dashboard dev server (`scripts/dev.sh` or `npm --prefix dashboard run dev`) and confirm the 3×3 grid renders with 9 points. This step is a human visual check — report if it cannot be run in the environment.

- [ ] **Step 6: Commit**

```bash
git add configs/queue_sweep.yaml
git commit -m "feat(config): queue_sweep.yaml — 3x3 arrival x cost demo sweep"
```

---

## Self-Review

**Spec coverage:**
- YAML loader + `!range` + grouped axes → Task 3 (loader), Task 1 (grouped expand), Task 2 (JSON for groups). ✓
- `goss_run_sweep` CLI end-to-end to dashboard → Tasks 4-6. ✓
- Grouped-axes spec extension (`AxisGroup`, `groups`, flatten to manifest) → Task 1; manifest flatten note below. ✓
- The config (`configs/queue_sweep.yaml`, both params, 9 runs) → Task 6. ✓
- Backward-compat/deprecation doc comments on `axes`/`combinator` → Task 1 Step 3; legacy JSON parse retained → Task 2. ✓
- yaml-cpp via FetchContent, HDF5 gating, C++17 → Tasks 3, 5; Global Constraints. ✓

**Manifest flatten note:** The design says `execute_sweep` should flatten `groups` into `archive.axes` so the dashboard sees a flat axis list. `execute_sweep` currently sets `archive.axes = spec.axes` (executor.cpp:166). With grouped configs, `spec.axes` is empty, so the dashboard manifest's `axes` array would be empty (runs still export correctly; only the axis metadata is missing). **Added Task 1.5 below** to flatten groups into the archive axes.

**Placeholder scan:** No TBD/TODO; all code steps contain full code. ✓
**Type consistency:** `AxisGroup{std::vector<Axis> axes;}`, `SweepSpec::groups`, `load_campaign_from_yaml`, `build_queue`, `resolve_root`, `export_dashboard_data`, `write_campaign` names match across tasks. ✓

---

### Task 1.5: Flatten grouped axes into the sweep archive (dashboard axis metadata)

**Files:**
- Modify: `src/spec/executor.cpp:166` (`archive.axes = spec.axes;`)
- Test: `tests/spec/test_executor.cpp` (add an axes-metadata assertion) — only if that file exists and has a lightweight fixture; otherwise fold the check into Task 6 Step 4.

**Interfaces:**
- Consumes: `SweepSpec::groups`, `SweepSpec::axes` (Task 1); `SweepArchive::axes` (executor.hpp:39).
- Produces: `archive.axes` populated from grouped axes (in group-then-axis order) when `groups` is non-empty, else from the legacy `axes`.

- [ ] **Step 1: Replace the axes assignment in execute_sweep**

At `src/spec/executor.cpp:166`, replace `archive.axes = spec.axes;` with:

```cpp
    // Flatten grouped axes (in group-then-axis order) for the dashboard manifest,
    // which expects a single flat list of axes. Fall back to the legacy flat axes.
    if (!spec.groups.empty()) {
        for (const AxisGroup& group : spec.groups)
            for (const Axis& axis : group.axes) archive.axes.push_back(axis);
    } else {
        archive.axes = spec.axes;
    }
```

- [ ] **Step 2: Build the spec library**

Run: `cmake --build build --target goss_spec`
Expected: compiles.

- [ ] **Step 3: Verify via the executor test suite (or defer to Task 6)**

Run: `ctest --test-dir build -R "SpecExecutor|Executor" --output-on-failure || echo "no dedicated executor axes test — verified end-to-end in Task 6 Step 4"`
Expected: existing executor tests still pass; grouped-axes metadata is confirmed in Task 6 Step 4.

- [ ] **Step 4: Commit**

```bash
git add src/spec/executor.cpp
git commit -m "feat(spec): flatten grouped axes into sweep archive metadata"
```

---

## Execution order

Task 1 → Task 1.5 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6. Tasks 1/1.5/2 are the spec core; 3 is the loader; 4/5 the CLI; 6 the config + end-to-end check.
