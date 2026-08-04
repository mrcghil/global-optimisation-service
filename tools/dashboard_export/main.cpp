// tools/dashboard_export/main.cpp
//
// Standalone executable: walk a goss results folder and emit the dashboard JSON
// data contract (index.json + sweep/<slug>.json + run/<run_id>.json).
//
// Usage: goss_dashboard_export <results_root> <out_dir>
//
// Prints "exported C campaigns, S sweeps, R runs" on success; non-zero on error.
// Requires HDF5 (run trajectories live in the .h5 archives).
#include <cstdlib>
#include <iostream>
#include <string>

#include "goss/sim/dashboard_export.hpp"
#include "goss/sim/errors.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: goss_dashboard_export <results_root> <out_dir>\n";
        return EXIT_FAILURE;
    }
#ifdef GOSS_HAVE_HDF5
    try {
        const goss::sim::ExportCounts counts =
            goss::sim::export_dashboard_data(argv[1], argv[2]);
        std::cout << "exported " << counts.campaigns << " campaigns, "
                  << counts.sweeps << " sweeps, " << counts.runs << " runs\n";
        return EXIT_SUCCESS;
    } catch (const goss::sim::SimError& e) {
        std::cerr << "goss_dashboard_export: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
#else
    std::cerr << "goss_dashboard_export: built without HDF5 support "
                 "(rebuild with libhdf5-dev to read run archives)\n";
    return EXIT_FAILURE;
#endif
}
