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
