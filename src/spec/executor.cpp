// src/spec/executor.cpp
#include "goss/spec/executor.hpp"

#include <chrono>
#include <ctime>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

#include "goss/model/errors.hpp"    // model::ModelError
#include "goss/sim/initial_guess.hpp"
#include "goss/sim/parameters.hpp"  // sim::solve_with_parameters
#include "goss/sim/sweep.hpp"
#include "goss/sim/trajectory.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "goss/solver/nlopt_solver.hpp"
#include "goss/spec/errors.hpp"
#include "goss/spec/json.hpp"

namespace goss::spec {
namespace {

/// Orders a run's NAMED parameters into the model's declared parameter order.
/// Every declared parameter must be present exactly once; unknown names are
/// rejected.  This is where naming becomes a safety property (the positional
/// grid could never catch a misnamed axis).
std::vector<double> order_parameters(const model::Model& model,
                                     const std::map<std::string, double>& named) {
    const std::size_t n = model.num_parameters();
    if (named.size() != n)
        throw SpecError("execute: problem declares " + std::to_string(n) +
                        " parameter(s) but the spec provides " +
                        std::to_string(named.size()));
    std::vector<double> ordered(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const std::string& name = model.parameter_name(i);
        const auto it = named.find(name);
        if (it == named.end())
            throw SpecError("execute: spec is missing a value for parameter '" +
                            name + "'");
        ordered[i] = it->second;
    }
    return ordered;
}

/// Builds a configured solver from the SolverSpec.  Returned via unique_ptr
/// because Solver is non-copyable/non-movable (see solver.hpp).
std::unique_ptr<solver::Solver> make_solver(const SolverSpec& spec) {
    if (spec.kind == "ipopt") {
        auto solver = std::make_unique<solver::IpoptSolver>();
        solver->set_tolerance(spec.tolerance);
        solver->set_max_iterations(spec.max_iterations);
        solver->set_print_level(spec.print_level);
        return solver;
    }
    if (spec.kind == "nlopt") {
        auto solver = std::make_unique<solver::NloptSolver>();
        solver->set_max_evaluations(spec.max_evaluations);
        solver->set_xtol_rel(spec.xtol_rel);
        return solver;
    }
    throw SpecError("make_solver: unknown solver kind '" + spec.kind +
                    "' (expected 'ipopt' or 'nlopt')");
}

/// Builds the initial guess for a compiled problem from the GuessSpec.
std::vector<double> make_guess(const GuessSpec& spec, const model::Model& model,
                               const transcription::VariableLayout& layout) {
    if (spec.kind == "linear")
        return sim::linear_guess(model, layout);
    if (spec.kind == "explicit") {
        if (spec.values.size() != layout.total_variables())
            throw SpecError("make_guess: explicit guess has " +
                            std::to_string(spec.values.size()) +
                            " values but the layout needs " +
                            std::to_string(layout.total_variables()));
        return spec.values;
    }
    throw SpecError("make_guess: unknown guess kind '" + spec.kind +
                    "' (expected 'linear' or 'explicit')");
}

Provenance capture_provenance(const std::string& scheme) {
    Provenance p;
    p.scheme = scheme;

    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
    gmtime_r(&t, &utc);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &utc);
    p.created_utc = buf;

    char host[256];
    if (gethostname(host, sizeof(host)) == 0) {
        host[sizeof(host) - 1] = '\0';
        p.hostname = host;
    }
#ifdef GOSS_VERSION
    p.goss_version = GOSS_VERSION;
#endif
    return p;
}

}  // namespace

RunArchive execute_run(const RunSpec& spec, const ProblemRegistry& registry) {
    BuiltProblem built = registry.build(spec.problem, spec.discretization);
    const std::vector<double> parameters = order_parameters(built.model, spec.parameters);
    const std::vector<double> guess =
        make_guess(spec.guess, built.model, built.compiled.layout);
    std::unique_ptr<solver::Solver> solver = make_solver(spec.solver);

    RunArchive archive;
    archive.spec = spec;
    archive.run_id = run_id(spec);
    archive.provenance = capture_provenance(built.scheme);

    // solve_with_parameters validates (naming-aware) then solves with solve-time
    // parameter injection.  A validation failure surfaces as a Failure result
    // rather than propagating, matching the sweep executor's point-level policy.
    try {
        archive.result = sim::solve_with_parameters(
            *solver, *built.compiled.problem, built.compiled.validator, guess, parameters);
    } catch (const model::ModelError& validation_error) {
        archive.result.status = solver::SolverStatus::Failure;
        archive.result.message = validation_error.what();
    }

    if (archive.result.status == solver::SolverStatus::Success) {
        archive.trajectory = sim::extract_trajectory(
            archive.result, built.compiled.layout, built.model, built.mesh);
    }
    archive.diagnosis = sim::diagnose(archive.result);
    return archive;
}

SweepArchive execute_sweep(const SweepSpec& spec, const ProblemRegistry& registry,
                           std::size_t max_parallel_workers) {
    const std::vector<RunSpec> runs = spec.expand();

    // Build the problem ONCE at the base discretization (axes vary parameters
    // only, so the compiled artifact is shared across all points).
    BuiltProblem built = registry.build(spec.base.problem, spec.base.discretization);
    const std::vector<double> guess =
        make_guess(spec.base.guess, built.model, built.compiled.layout);
    std::unique_ptr<solver::Solver> solver = make_solver(spec.base.solver);

    // Map each expanded run's named parameters into the ordered grid.
    std::vector<std::vector<double>> grid;
    grid.reserve(runs.size());
    for (const RunSpec& run : runs)
        grid.push_back(order_parameters(built.model, run.parameters));

    sim::SweepConfig config;
    config.max_parallel_workers = max_parallel_workers;
    const sim::SweepResult sweep_result = sim::run_sweep_parallel(
        *built.compiled.problem, built.compiled.validator, *solver, grid, guess, config);

    const Provenance provenance = capture_provenance(built.scheme);

    SweepArchive archive;
    archive.spec = spec;
    // Flatten grouped axes (in group-then-axis order) for the dashboard manifest,
    // which expects a single flat list of axes. Fall back to the legacy flat axes.
    if (!spec.groups.empty()) {
        for (const AxisGroup& group : spec.groups)
            for (const Axis& axis : group.axes) archive.axes.push_back(axis);
    } else {
        archive.axes = spec.axes;
    }
    archive.runs.reserve(runs.size());
    for (std::size_t i = 0; i < runs.size(); ++i) {
        const sim::SweepPoint& point = sweep_result.points[i];
        RunArchive run_archive;
        run_archive.spec = runs[i];
        run_archive.run_id = run_id(runs[i]);
        run_archive.provenance = provenance;
        run_archive.result.status = point.status;
        run_archive.result.objective_value = point.objective_value;
        run_archive.result.x = point.x;
        run_archive.result.message = point.message;
        if (point.status == solver::SolverStatus::Success && !point.x.empty()) {
            run_archive.trajectory = sim::extract_trajectory(
                run_archive.result, built.compiled.layout, built.model, built.mesh);
        }
        run_archive.diagnosis = sim::diagnose(run_archive.result);
        archive.runs.push_back(std::move(run_archive));
    }
    return archive;
}

CampaignArchive execute_campaign(const CampaignSpec& spec,
                                 const ProblemRegistry& registry,
                                 std::size_t max_parallel_workers) {
    CampaignArchive archive;
    archive.name = spec.name;
    archive.sweeps.reserve(spec.sweeps.size());
    for (const SweepSpec& sweep : spec.sweeps)
        archive.sweeps.push_back(execute_sweep(sweep, registry, max_parallel_workers));
    return archive;
}

}  // namespace goss::spec
