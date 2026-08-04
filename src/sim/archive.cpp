// src/sim/archive.cpp
#include "goss/sim/archive.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "goss/bench/benchmark_result.hpp"  // solver_status_name
#include "goss/sim/errors.hpp"
#include "goss/spec/json.hpp"

#ifdef GOSS_HAVE_HDF5
#include <highfive/H5File.hpp>
#include <highfive/H5Easy.hpp>
#endif

namespace goss::sim {
namespace {

std::string status_string(solver::SolverStatus status) {
    return goss::bench::solver_status_name(status);
}

}  // namespace

std::string resolve_run_dir(const spec::RunArchive& archive) {
    std::string root = archive.spec.storage.root;
    if (root.empty()) {
        if (const char* env = std::getenv("GOSS_RESULTS_DIR"); env && *env)
            root = env;
        else
            root = "./goss-results";
    }
    const std::filesystem::path dir = std::filesystem::path(root) /
                                      archive.spec.problem.name /
                                      archive.spec.problem.version;
    return dir.string();
}

std::string archive_path(const spec::RunArchive& archive) {
    return (std::filesystem::path(resolve_run_dir(archive)) /
            (archive.run_id + ".h5")).string();
}

std::string sidecar_path(const spec::RunArchive& archive) {
    return (std::filesystem::path(resolve_run_dir(archive)) /
            (archive.run_id + ".json")).string();
}

void write_sidecar(const std::string& path, const spec::RunArchive& archive) {
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);

    nlohmann::json j;
    j["run_id"] = archive.run_id;
    j["spec"] = archive.spec;  // full RunSpec (includes image_pipeline)
    j["result"] = {
        {"status", status_string(archive.result.status)},
        {"objective", archive.result.objective_value},
        {"message", archive.result.message},
    };
    j["provenance"] = {
        {"created_utc", archive.provenance.created_utc},
        {"hostname", archive.provenance.hostname},
        {"scheme", archive.provenance.scheme},
        {"goss_version", archive.provenance.goss_version},
    };
    j["diagnosis"] = {{"ok", archive.diagnosis.ok},
                      {"summary", archive.diagnosis.summary},
                      {"advice", archive.diagnosis.advice}};

    std::ofstream file(path);
    if (!file) throw SimError("write_sidecar: cannot open '" + path + "' for writing");
    file << j.dump(2) << "\n";
    if (!file) throw SimError("write_sidecar: write failed for '" + path + "'");
}

#ifdef GOSS_HAVE_HDF5

void write_run_archive(const std::string& path, const spec::RunArchive& archive) {
    try {
        std::error_code ec;
        std::filesystem::create_directories(
            std::filesystem::path(path).parent_path(), ec);

        HighFive::File file(path, HighFive::File::Overwrite);

        auto set_attr = [&](const std::string& name, const std::string& value) {
            file.createAttribute<std::string>(name, HighFive::DataSpace::From(value))
                .write(value);
        };
        set_attr("run_id", archive.run_id);
        set_attr("problem_name", archive.spec.problem.name);
        set_attr("problem_version", archive.spec.problem.version);
        set_attr("scheme", archive.provenance.scheme);
        set_attr("solver_kind", archive.spec.solver.kind);
        set_attr("image_pipeline", archive.spec.image_pipeline);
        set_attr("label", archive.spec.label);
        set_attr("created_utc", archive.provenance.created_utc);
        set_attr("hostname", archive.provenance.hostname);
        set_attr("goss_version", archive.provenance.goss_version);

        // Full round-trippable spec.
        const nlohmann::json spec_json = archive.spec;
        const std::string spec_str = spec_json.dump();
        file.createDataSet<std::string>("spec_json", HighFive::DataSpace::From(spec_str))
            .write(spec_str);

        // Result summary.
        HighFive::Group result = file.createGroup("result");
        const std::string status = status_string(archive.result.status);
        result.createAttribute<std::string>("status", HighFive::DataSpace::From(status))
            .write(status);
        result.createAttribute<double>("objective", HighFive::DataSpace::From(
            archive.result.objective_value)).write(archive.result.objective_value);
        result.createAttribute<std::string>("message", HighFive::DataSpace::From(
            archive.result.message)).write(archive.result.message);

        // Named parameters.
        HighFive::Group params = file.createGroup("parameters");
        for (const auto& [name, value] : archive.spec.parameters)
            params.createDataSet<double>(name, HighFive::DataSpace::From(value)).write(value);

        // Trajectory (present only for converged runs).
        const Trajectory& traj = archive.trajectory;
        if (!traj.times.empty()) {
            HighFive::Group tg = file.createGroup("trajectory");
            tg.createDataSet("time", traj.times);
            HighFive::Group sg = tg.createGroup("states");
            for (std::size_t i = 0; i < traj.state_names.size(); ++i)
                sg.createDataSet(traj.state_names[i], traj.states[i]);
            HighFive::Group cg = tg.createGroup("controls");
            for (std::size_t j = 0; j < traj.control_names.size(); ++j)
                cg.createDataSet(traj.control_names[j], traj.controls[j]);
        }
    } catch (const HighFive::Exception& e) {
        throw SimError(std::string("write_run_archive: HDF5 error: ") + e.what());
    }
}

spec::RunArchive read_run_archive(const std::string& path) {
    try {
        HighFive::File file(path, HighFive::File::ReadOnly);
        spec::RunArchive archive;

        file.getAttribute("run_id").read(archive.run_id);
        file.getAttribute("created_utc").read(archive.provenance.created_utc);
        file.getAttribute("hostname").read(archive.provenance.hostname);
        file.getAttribute("scheme").read(archive.provenance.scheme);
        file.getAttribute("goss_version").read(archive.provenance.goss_version);

        std::string spec_str;
        file.getDataSet("spec_json").read(spec_str);
        archive.spec = nlohmann::json::parse(spec_str).get<spec::RunSpec>();

        HighFive::Group result = file.getGroup("result");
        std::string status;
        result.getAttribute("status").read(status);
        // Map the label back to the enum (Success is the only value tests assert;
        // others are carried as-is via the message).
        using solver::SolverStatus;
        if (status == "Success") archive.result.status = SolverStatus::Success;
        else if (status == "InfeasibleProblem") archive.result.status = SolverStatus::InfeasibleProblem;
        else if (status == "IterationLimit") archive.result.status = SolverStatus::IterationLimit;
        else if (status == "NumericalError") archive.result.status = SolverStatus::NumericalError;
        else archive.result.status = SolverStatus::Failure;
        result.getAttribute("objective").read(archive.result.objective_value);
        result.getAttribute("message").read(archive.result.message);

        if (file.exist("trajectory")) {
            HighFive::Group tg = file.getGroup("trajectory");
            tg.getDataSet("time").read(archive.trajectory.times);
            HighFive::Group sg = tg.getGroup("states");
            for (const std::string& name : sg.listObjectNames()) {
                archive.trajectory.state_names.push_back(name);
                std::vector<double> series;
                sg.getDataSet(name).read(series);
                archive.trajectory.states.push_back(std::move(series));
            }
            HighFive::Group cg = tg.getGroup("controls");
            for (const std::string& name : cg.listObjectNames()) {
                archive.trajectory.control_names.push_back(name);
                std::vector<double> series;
                cg.getDataSet(name).read(series);
                archive.trajectory.controls.push_back(std::move(series));
            }
        }
        return archive;
    } catch (const HighFive::Exception& e) {
        throw SimError(std::string("read_run_archive: HDF5 error: ") + e.what());
    }
}

#endif  // GOSS_HAVE_HDF5

}  // namespace goss::sim
