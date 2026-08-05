// tools/run_sweep/main.cpp
//
// Standalone executable: load a YAML campaign, run every sweep, write result
// manifests, and (with HDF5) export the dashboard JSON contract.
//
// Usage: goss_run_sweep <config.yaml> [out_data_dir] [max_workers]
//   out_data_dir defaults to "dashboard-data"; max_workers defaults to 0
//   (hardware concurrency). Results manifests go to the storage.root in the
//   config (default ./goss-results).
#include <cstdlib>
#include <iostream>
#include <string>

#include "goss/config/yaml_loader.hpp"
#include "goss/sim/archive.hpp"
#include "goss/sim/errors.hpp"
#include "goss/spec/errors.hpp"
#include "goss/spec/executor.hpp"
#include "goss/spec/registry.hpp"
#include "queue_model.hpp"

#ifdef GOSS_HAVE_HDF5
#include "goss/sim/dashboard_export.hpp"
#endif

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage: goss_run_sweep <config.yaml> [out_data_dir] "
                     "[max_workers]\n";
        return EXIT_FAILURE;
    }
    const std::string config_path   = argv[1];
    const std::string out_data_dir  = argc > 2 ? argv[2] : "dashboard-data";

    // Parse max_workers from argv[3] when provided.  strtoul returns 0 on
    // invalid input, which silently masks typos; use an endptr to detect any
    // leftover non-numeric characters and reject the argument explicitly.
    std::size_t max_workers = 0;
    if (argc > 3) {
        const char* const raw_max_workers = argv[3];
        char*             end_ptr         = nullptr;
        const unsigned long parsed_workers =
            std::strtoul(raw_max_workers, &end_ptr, 10);
        const bool has_leftover_chars = (end_ptr != nullptr && *end_ptr != '\0');
        const bool is_empty_string    = (raw_max_workers[0] == '\0');
        if (is_empty_string || has_leftover_chars) {
            std::cerr << "goss_run_sweep: max_workers must be a non-negative integer\n";
            return EXIT_FAILURE;
        }
        max_workers = static_cast<std::size_t>(parsed_workers);
    }

    try {
        const goss::spec::CampaignSpec campaign =
            goss::config::load_campaign_from_yaml(config_path);

        goss::spec::ProblemRegistry registry;
        registry.register_problem({"queue", "v1"}, goss_tools::build_queue);

        const goss::spec::CampaignArchive archive =
            goss::spec::execute_campaign(campaign, registry, max_workers);
        const std::string manifest_path = goss::sim::write_campaign(archive);

        std::size_t total_runs = 0;
        std::size_t total_succeeded = 0;
        for (const goss::spec::SweepArchive& sweep : archive.sweeps) {
            total_runs += sweep.runs.size();
            total_succeeded += sweep.num_succeeded();
        }
        std::cout << "ran campaign '" << archive.name << "': " << total_succeeded
                  << "/" << total_runs << " runs succeeded\n"
                  << "manifest: " << manifest_path << "\n";

#ifdef GOSS_HAVE_HDF5
        // The dashboard export walks a single results directory tree and writes
        // every campaign/sweep/run it finds there.  All sweeps in a campaign
        // must therefore share the same storage.root; if they differ, runs
        // stored under a different root would be silently omitted from the
        // export.  Enforce the invariant here before resolving the root so the
        // error is reported early with a clear message.
        if (campaign.sweeps.size() > 1) {
            const std::string& first_root =
                campaign.sweeps.front().base.storage.root;
            for (const goss::spec::SweepSpec& sweep_spec : campaign.sweeps) {
                if (sweep_spec.base.storage.root != first_root) {
                    throw goss::spec::SpecError(
                        "goss_run_sweep: all sweeps in a campaign must share "
                        "the same storage.root for dashboard export");
                }
            }
        }
        const std::string results_root =
            goss::sim::resolve_root(campaign.sweeps.empty()
                                        ? goss::spec::StorageSpec{}
                                        : campaign.sweeps.front().base.storage);
        const goss::sim::ExportCounts counts =
            goss::sim::export_dashboard_data(results_root, out_data_dir);
        std::cout << "exported dashboard data to '" << out_data_dir << "': "
                  << counts.campaigns << " campaigns, " << counts.sweeps
                  << " sweeps, " << counts.runs << " runs\n";
#else
        std::cout << "(built without HDF5 — skipped dashboard export; manifests "
                     "written)\n";
        (void)out_data_dir;
#endif
        return EXIT_SUCCESS;
    } catch (const goss::spec::SpecError& e) {
        std::cerr << "goss_run_sweep: config/spec error: " << e.what() << "\n";
        return EXIT_FAILURE;
    } catch (const goss::sim::SimError& e) {
        std::cerr << "goss_run_sweep: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
