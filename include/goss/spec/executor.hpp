// include/goss/spec/executor.hpp
#pragma once
#include <string>
#include <vector>
#include "goss/sim/diagnostics.hpp"
#include "goss/sim/trajectory.hpp"
#include "goss/solver/solver_result.hpp"
#include "goss/spec/registry.hpp"
#include "goss/spec/specs.hpp"

namespace goss::spec {

/// Provenance captured at execution time — the "who/when/how" that the spec
/// itself does not carry.  All best-effort: unavailable fields are left empty.
struct Provenance {
    std::string created_utc;   // ISO-8601, e.g. "2026-08-04T12:34:56Z"
    std::string hostname;
    std::string scheme;        // resolved transcription scheme
    std::string goss_version;  // build-stamped version string
};

/// In-memory result of one solved run: the spec, its content-addressed id, the
/// solver outcome, the UNPACKED trajectory (which the flat SweepPoint::x throws
/// away today), a human diagnosis, and provenance.  This is the counterpart of
/// the on-disk HDF5 archive written in Stage 4.
struct RunArchive {
    RunSpec               spec;
    std::string           run_id;
    solver::SolverResult  result;
    sim::Trajectory       trajectory;  // empty when the solve did not converge
    sim::Diagnosis        diagnosis;
    Provenance            provenance;
};

/// Results of a sweep: one RunArchive per expanded point (index-aligned with
/// SweepSpec::expand()), plus the axis metadata needed for later N-D indexing.
struct SweepArchive {
    SweepSpec               spec;
    std::vector<Axis>       axes;     // copied from spec for convenient indexing
    std::vector<RunArchive> runs;

    std::size_t num_succeeded() const {
        std::size_t count = 0;
        for (const RunArchive& run : runs)
            if (run.result.status == solver::SolverStatus::Success) ++count;
        return count;
    }
};

/// Results of a campaign: its name plus one SweepArchive per sweep.
struct CampaignArchive {
    std::string               name;
    std::vector<SweepArchive> sweeps;
};

/// Executes a single run: resolves the problem from the registry at the spec's
/// discretization, maps the NAMED parameters onto the model's ordered parameter
/// vector (throwing SpecError on unknown/missing names), builds the solver and
/// initial guess from the spec, solves, and unpacks the trajectory.
///
/// Point-level solve outcomes (including non-convergence) are reported via
/// RunArchive::result.status — this does NOT throw for a failed solve.  It throws
/// SpecError only for configuration errors (unknown problem, bad parameter names,
/// unknown scheme/solver/guess kind).
RunArchive execute_run(const RunSpec& spec, const ProblemRegistry& registry);

/// Executes a sweep: builds the problem ONCE at base.discretization, expands the
/// named axes to a grid, runs the points through the existing parallel sweep
/// executor, and unpacks a trajectory per point.  `max_parallel_workers` maps to
/// sim::SweepConfig (0 => hardware_concurrency).
SweepArchive execute_sweep(const SweepSpec& spec, const ProblemRegistry& registry,
                           std::size_t max_parallel_workers = 0);

/// Executes each sweep of the campaign in order.
CampaignArchive execute_campaign(const CampaignSpec& spec,
                                 const ProblemRegistry& registry,
                                 std::size_t max_parallel_workers = 0);

}  // namespace goss::spec
