// src/sim/dashboard_export.cpp
#include "goss/sim/dashboard_export.hpp"

#ifdef GOSS_HAVE_HDF5

#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "goss/bench/benchmark_result.hpp"  // solver_status_name
#include "goss/sim/archive.hpp"
#include "goss/sim/errors.hpp"
#include "goss/spec/executor.hpp"
#include "goss/spec/json.hpp"  // to_json(RunSpec)

namespace goss::sim {
namespace {

namespace fs = std::filesystem;
using nlohmann::json;

json read_json_file(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw SimError("dashboard_export: cannot read '" + path.string() + "'");
    return json::parse(in);
}

void write_json_file(const fs::path& path, const json& j) {
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    std::ofstream out(path);
    if (!out) throw SimError("dashboard_export: cannot write '" + path.string() + "'");
    out << j.dump(2) << "\n";
    if (!out) throw SimError("dashboard_export: write failed for '" + path.string() + "'");
}

/// The run store path for a logical run reference: <root>/<problem>/<version>/<run_id>.h5.
fs::path run_archive_file(const fs::path& results_root, const std::string& problem,
                          const std::string& version, const std::string& run_id) {
    return results_root / problem / version / (run_id + ".h5");
}

/// Emits <out>/run/<run_id>.json from the run's HDF5 archive.  Idempotent within
/// a single export (a run shared by multiple sweeps is written once).
void export_run(const fs::path& results_root, const fs::path& out_dir,
                const std::string& problem, const std::string& version,
                const std::string& run_id, std::set<std::string>& already_written) {
    if (!already_written.insert(run_id).second) return;

    const fs::path h5 = run_archive_file(results_root, problem, version, run_id);
    const spec::RunArchive a = read_run_archive(h5.string());

    json states = json::object();
    for (std::size_t i = 0; i < a.trajectory.state_names.size(); ++i)
        states[a.trajectory.state_names[i]] = a.trajectory.states[i];
    json controls = json::object();
    for (std::size_t j = 0; j < a.trajectory.control_names.size(); ++j)
        controls[a.trajectory.control_names[j]] = a.trajectory.controls[j];

    // Assignment style (not nested brace-init) — nested object VALUES inside a
    // basic_json constructor init-list defeat nlohmann's deduction.
    json detail;
    detail["run_id"] = a.run_id;
    detail["problem"] = a.spec.problem.name;
    detail["version"] = a.spec.problem.version;
    detail["spec"] = a.spec;
    detail["result"] = {{"status", goss::bench::solver_status_name(a.result.status)},
                        {"objective", a.result.objective_value},
                        {"message", a.result.message}};
    detail["diagnosis"] = {{"ok", a.diagnosis.ok},
                           {"summary", a.diagnosis.summary},
                           {"advice", a.diagnosis.advice}};
    detail["provenance"] = {{"created_utc", a.provenance.created_utc},
                            {"hostname", a.provenance.hostname},
                            {"scheme", a.provenance.scheme},
                            {"goss_version", a.provenance.goss_version}};
    detail["trajectory"] = {{"time", a.trajectory.times},
                            {"states", states},
                            {"controls", controls}};
    write_json_file(out_dir / "run" / (run_id + ".json"), detail);
}

/// Copies a sweep manifest to the contract's sweep/<slug>.json and exports each
/// referenced run.  Returns the summary object for index.json.  `slug` is the
/// manifest directory name.
json export_sweep(const fs::path& results_root, const fs::path& out_dir,
                  const std::string& slug, const json& manifest,
                  std::set<std::string>& runs_written) {
    // The manifest already matches the sweep contract shape; re-emit verbatim.
    write_json_file(out_dir / "sweep" / (slug + ".json"), manifest);

    for (const json& run : manifest.at("runs")) {
        export_run(results_root, out_dir, run.at("problem").get<std::string>(),
                   run.at("version").get<std::string>(),
                   run.at("run_id").get<std::string>(), runs_written);
    }

    return json{{"slug", slug},
                {"label", manifest.value("label", slug)},
                {"problem", manifest.value("problem", "")},
                {"version", manifest.value("version", "")},
                {"axes", manifest.value("axes", json::array())},
                {"num_runs", manifest.value("num_runs", 0)},
                {"num_succeeded", manifest.value("num_succeeded", 0)}};
}

}  // namespace

ExportCounts export_dashboard_data(const std::string& results_root,
                                   const std::string& out_dir) {
    const fs::path root(results_root);
    const fs::path out(out_dir);
    if (!fs::exists(root))
        throw SimError("dashboard_export: results root '" + results_root +
                       "' does not exist");

    ExportCounts counts;
    std::set<std::string> runs_written;
    json index_sweeps = json::array();
    json index_campaigns = json::array();

    // Standalone sweeps: <root>/sweeps/<slug>/sweep.json
    const fs::path sweeps_dir = root / "sweeps";
    if (fs::exists(sweeps_dir)) {
        for (const auto& entry : fs::directory_iterator(sweeps_dir)) {
            const fs::path manifest = entry.path() / "sweep.json";
            if (!fs::exists(manifest)) continue;
            const std::string slug = entry.path().filename().string();
            const json summary = export_sweep(root, out, slug,
                                              read_json_file(manifest), runs_written);
            index_sweeps.push_back(summary);
            ++counts.sweeps;
        }
    }

    // Campaigns: <root>/campaigns/<slug>/campaign.json
    const fs::path campaigns_dir = root / "campaigns";
    if (fs::exists(campaigns_dir)) {
        for (const auto& entry : fs::directory_iterator(campaigns_dir)) {
            const fs::path manifest = entry.path() / "campaign.json";
            if (!fs::exists(manifest)) continue;
            const json campaign = read_json_file(manifest);
            const std::string camp_slug = entry.path().filename().string();

            std::size_t campaign_runs = 0, campaign_succeeded = 0;
            const json& sweeps = campaign.at("sweeps");
            for (std::size_t s = 0; s < sweeps.size(); ++s) {
                // A campaign's sweeps may be unlabeled; derive a unique slug.
                const std::string sub_slug =
                    camp_slug + "__" + sweeps[s].value("label", std::to_string(s));
                const json summary = export_sweep(root, out, sub_slug,
                                                  sweeps[s], runs_written);
                index_sweeps.push_back(summary);
                ++counts.sweeps;
                campaign_runs += sweeps[s].value("num_runs", 0);
                campaign_succeeded += sweeps[s].value("num_succeeded", 0);
            }
            index_campaigns.push_back({{"name", campaign.value("name", camp_slug)},
                                       {"slug", camp_slug},
                                       {"num_sweeps", sweeps.size()},
                                       {"num_runs", campaign_runs},
                                       {"num_succeeded", campaign_succeeded}});
            ++counts.campaigns;
        }
    }

    counts.runs = runs_written.size();
    write_json_file(out / "index.json",
                    json{{"campaigns", index_campaigns}, {"sweeps", index_sweeps}});
    return counts;
}

}  // namespace goss::sim

#endif  // GOSS_HAVE_HDF5
