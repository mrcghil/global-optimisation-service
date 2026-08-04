// include/goss/sim/dashboard_export.hpp
#pragma once
#include <cstddef>
#include <string>

// Static exporter for the dashboard JSON data contract (Stage B/C).  Walks a
// results folder produced by write_run/write_sweep/write_campaign and emits:
//   <out>/index.json                 navigation tree (campaigns + sweeps)
//   <out>/sweep/<slug>.json          one sweep's parameter map
//   <out>/run/<run_id>.json          one run's full detail (spec + trajectory)
//
// Both this local exporter and a future server produce the SAME shapes, so the
// Astro dataClient codes against one contract regardless of source.  Reading run
// trajectories requires HDF5, so the whole exporter is gated on GOSS_HAVE_HDF5.
namespace goss::sim {

#ifdef GOSS_HAVE_HDF5

struct ExportCounts {
    std::size_t campaigns = 0;
    std::size_t sweeps = 0;
    std::size_t runs = 0;
};

/// Reads manifests + run archives under `results_root` and writes the dashboard
/// contract into `out_dir` (creating it).  `results_root` is the base passed as
/// StorageSpec.root at solve time (it contains the <problem>/<version>/ run
/// store plus sweeps/ and campaigns/ manifest trees).  Throws SimError on I/O or
/// HDF5 failure.  Returns how many campaigns/sweeps/runs were exported.
ExportCounts export_dashboard_data(const std::string& results_root,
                                   const std::string& out_dir);

#endif  // GOSS_HAVE_HDF5

}  // namespace goss::sim
