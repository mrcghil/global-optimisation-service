// src/spec/json.cpp
#include "goss/spec/json.hpp"
#include <cstdint>
#include <string>
#include <nlohmann/json.hpp>

namespace goss::spec {

using nlohmann::json;

void to_json(json& j, const ProblemKey& v) {
    j = json{{"name", v.name}, {"version", v.version}};
}
void from_json(const json& j, ProblemKey& v) {
    j.at("name").get_to(v.name);
    j.at("version").get_to(v.version);
}

void to_json(json& j, const DiscretizationSpec& v) {
    j = json{{"scheme", v.scheme},
             {"t_initial", v.t_initial},
             {"t_final", v.t_final},
             {"num_intervals", v.num_intervals},
             {"lgl_degree", v.lgl_degree}};
}
void from_json(const json& j, DiscretizationSpec& v) {
    j.at("scheme").get_to(v.scheme);
    j.at("t_initial").get_to(v.t_initial);
    j.at("t_final").get_to(v.t_final);
    j.at("num_intervals").get_to(v.num_intervals);
    j.at("lgl_degree").get_to(v.lgl_degree);
}

void to_json(json& j, const SolverSpec& v) {
    j = json{{"kind", v.kind},
             {"tolerance", v.tolerance},
             {"max_iterations", v.max_iterations},
             {"print_level", v.print_level},
             {"max_evaluations", v.max_evaluations},
             {"xtol_rel", v.xtol_rel}};
}
void from_json(const json& j, SolverSpec& v) {
    j.at("kind").get_to(v.kind);
    j.at("tolerance").get_to(v.tolerance);
    j.at("max_iterations").get_to(v.max_iterations);
    j.at("print_level").get_to(v.print_level);
    j.at("max_evaluations").get_to(v.max_evaluations);
    j.at("xtol_rel").get_to(v.xtol_rel);
}

void to_json(json& j, const GuessSpec& v) {
    j = json{{"kind", v.kind}, {"values", v.values}};
}
void from_json(const json& j, GuessSpec& v) {
    j.at("kind").get_to(v.kind);
    j.at("values").get_to(v.values);
}

void to_json(json& j, const StorageSpec& v) {
    j = json{{"root", v.root}, {"skip_if_exists", v.skip_if_exists}};
}
void from_json(const json& j, StorageSpec& v) {
    j.at("root").get_to(v.root);
    j.at("skip_if_exists").get_to(v.skip_if_exists);
}

void to_json(json& j, const RunSpec& v) {
    j = json{{"problem", v.problem},
             {"parameters", v.parameters},
             {"discretization", v.discretization},
             {"solver", v.solver},
             {"guess", v.guess},
             {"storage", v.storage},
             {"image_pipeline", v.image_pipeline},
             {"label", v.label}};
}
void from_json(const json& j, RunSpec& v) {
    // Structurally required fields: missing any of these is a malformed RunSpec.
    j.at("problem").get_to(v.problem);
    j.at("parameters").get_to(v.parameters);
    j.at("discretization").get_to(v.discretization);
    j.at("solver").get_to(v.solver);
    j.at("guess").get_to(v.guess);
    j.at("storage").get_to(v.storage);
    // Optional fields that default to empty and are excluded from run identity:
    // tolerate hand-authored JSON that omits them.
    if (j.contains("image_pipeline")) j.at("image_pipeline").get_to(v.image_pipeline);
    if (j.contains("label")) j.at("label").get_to(v.label);
}

void to_json(json& j, const Axis& v) {
    j = json{{"parameter", v.parameter}, {"values", v.values}};
}
void from_json(const json& j, Axis& v) {
    j.at("parameter").get_to(v.parameter);
    j.at("values").get_to(v.values);
}

void to_json(json& j, const AxisGroup& v) {
    j = json{{"axes", v.axes}};
}
void from_json(const json& j, AxisGroup& v) {
    j.at("axes").get_to(v.axes);
}

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
    // `label` is optional; hand-authored JSON may omit it.
    if (j.contains("label")) j.at("label").get_to(v.label);
}

void to_json(json& j, const CampaignSpec& v) {
    j = json{{"name", v.name}, {"sweeps", v.sweeps}};
}
void from_json(const json& j, CampaignSpec& v) {
    j.at("name").get_to(v.name);
    j.at("sweeps").get_to(v.sweeps);
}

std::string canonical_identity_json(const RunSpec& spec) {
    // Clear volatile fields: storage location, human label, and image-pipeline
    // name do not change the computed solution, so they must not affect identity.
    RunSpec identity = spec;
    identity.storage = StorageSpec{};  // reset to a constant — storage never affects identity
    identity.label.clear();
    identity.image_pipeline.clear();

    // std::map already orders parameter keys; dump with sorted object keys so the
    // byte sequence is stable regardless of insertion order elsewhere.
    json j = identity;
    return j.dump(-1, ' ', /*ensure_ascii=*/false,
                  nlohmann::json::error_handler_t::strict);
}

std::string run_id(const RunSpec& spec) {
    const std::string canonical = canonical_identity_json(spec);
    // FNV-1a 64-bit — deterministic, dependency-free.
    std::uint64_t hash = 1469598103934665603ULL;
    for (unsigned char byte : canonical) {
        hash ^= static_cast<std::uint64_t>(byte);
        hash *= 1099511628211ULL;
    }
    static const char* kHex = "0123456789abcdef";
    std::string out(16, '0');
    for (int i = 15; i >= 0; --i) {
        out[i] = kHex[hash & 0xF];
        hash >>= 4;
    }
    return out;
}

}  // namespace goss::spec
