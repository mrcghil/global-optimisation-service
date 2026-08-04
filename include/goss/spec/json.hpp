// include/goss/spec/json.hpp
#pragma once
#include <cstdint>
#include <string>
#include <nlohmann/json.hpp>
#include "goss/spec/specs.hpp"

// nlohmann (de)serialization for the spec types, plus a stable content hash used
// as an idempotency / cache key.  to_json / from_json are found by ADL and back
// the free functions declared below.
namespace goss::spec {

void to_json(nlohmann::json& j, const ProblemKey& v);
void from_json(const nlohmann::json& j, ProblemKey& v);
void to_json(nlohmann::json& j, const DiscretizationSpec& v);
void from_json(const nlohmann::json& j, DiscretizationSpec& v);
void to_json(nlohmann::json& j, const SolverSpec& v);
void from_json(const nlohmann::json& j, SolverSpec& v);
void to_json(nlohmann::json& j, const GuessSpec& v);
void from_json(const nlohmann::json& j, GuessSpec& v);
void to_json(nlohmann::json& j, const StorageSpec& v);
void from_json(const nlohmann::json& j, StorageSpec& v);
void to_json(nlohmann::json& j, const RunSpec& v);
void from_json(const nlohmann::json& j, RunSpec& v);
void to_json(nlohmann::json& j, const Axis& v);
void from_json(const nlohmann::json& j, Axis& v);
void to_json(nlohmann::json& j, const SweepSpec& v);
void from_json(const nlohmann::json& j, SweepSpec& v);
void to_json(nlohmann::json& j, const CampaignSpec& v);
void from_json(const nlohmann::json& j, CampaignSpec& v);

/// Canonical JSON string for identity: the RunSpec with volatile fields
/// (storage, label, image_pipeline) cleared, serialized with sorted keys so the
/// same logical run always produces the same bytes.
std::string canonical_identity_json(const RunSpec& spec);

/// Stable 64-bit content hash of `canonical_identity_json`, rendered as 16 lower
/// case hex chars.  Same problem+parameters+discretization+solver+guess => same
/// id; changing any of them invalidates it.  Uses FNV-1a — no crypto dependency.
std::string run_id(const RunSpec& spec);

}  // namespace goss::spec
