// include/goss/sim/archive.hpp
#pragma once
#include <string>
#include "goss/spec/executor.hpp"  // spec::RunArchive

// Self-describing HDF5 archive for a single solved run (best-ideas Idea 8:
// replace CSV with a self-describing, versioned, provenance-bearing format).
//
// The HDF5 body is compiled only when GOSS_HAVE_HDF5 is defined (HighFive +
// libhdf5 present).  The JSON sidecar and path helpers are always available so
// spec-driven storage/idempotency works even without HDF5.
namespace goss::sim {

/// Resolves the directory a run's artifacts are written under, applying the
/// StorageSpec precedence: explicit root, else $GOSS_RESULTS_DIR, else
/// "./goss-results".  The returned path includes the problem name/version
/// subdirectories: <root>/<problem.name>/<problem.version>.
std::string resolve_run_dir(const spec::RunArchive& archive);

/// Full path of the HDF5 archive for a run: <run_dir>/<run_id>.h5.
std::string archive_path(const spec::RunArchive& archive);

/// Full path of the JSON sidecar for a run: <run_dir>/<run_id>.json.  This is
/// the hook a downstream image pipeline reads (it carries the RunSpec, the
/// result summary, and the image_pipeline name).  Always available.
std::string sidecar_path(const spec::RunArchive& archive);

/// Writes the JSON sidecar (spec + result summary + provenance) to `path`.
/// Creates parent directories as needed.  Throws SimError on I/O failure.
void write_sidecar(const std::string& path, const spec::RunArchive& archive);

#ifdef GOSS_HAVE_HDF5
/// Writes the self-describing HDF5 archive to `path` (creating parent dirs).
/// Layout:
///   /                    attrs: run_id, problem_name/version, scheme, solver,
///                               image_pipeline, label, created_utc, hostname,
///                               goss_version
///   /spec_json           full RunSpec JSON (round-trippable)
///   /result              attrs: status, objective, message
///   /parameters/<name>   scalar<double> per named parameter
///   /trajectory/time     dataset<double>[N]
///   /trajectory/states/<name>    dataset<double>[N]
///   /trajectory/controls/<name>  dataset<double>[N]
/// Throws SimError on HDF5 failure.
void write_run_archive(const std::string& path, const spec::RunArchive& archive);

/// Reopens an archive written by write_run_archive.  Reconstructs the RunSpec
/// (from /spec_json), the result summary, and the trajectory.  Throws SimError
/// if the file is missing or malformed.
spec::RunArchive read_run_archive(const std::string& path);
#endif  // GOSS_HAVE_HDF5

}  // namespace goss::sim
