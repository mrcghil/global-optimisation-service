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
