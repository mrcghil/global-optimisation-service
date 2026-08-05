// tools/dashboard_export/gen_sample.cpp
//
// Developer utility: solve a small demo campaign (the queue problem) and export
// the dashboard JSON contract into dashboard/sample-data/.  Used to produce
// fixture data for developing the Astro app; not part of the normal build.
#include <string>
#include <utility>
#include <vector>

#include "goss/sim/archive.hpp"
#include "goss/sim/dashboard_export.hpp"
#include "goss/spec/executor.hpp"
#include "queue_model.hpp"

using namespace goss;

int main(int argc, char** argv) {
    const std::string base_dir = argc > 1 ? argv[1] : "dashboard/sample-data";
    const std::string results = base_dir + "/results";
    const std::string data = base_dir + "/data";

    spec::ProblemRegistry reg;
    reg.register_problem({"queue", "v1"}, goss_tools::build_queue);

    spec::RunSpec base;
    base.problem = {"queue", "v1"};
    base.parameters = {{"arrival_rate", 2.0}, {"cost_weight", 0.1}};
    base.discretization.t_final = 5.0;
    base.discretization.num_intervals = 25;
    base.storage.root = results;

    spec::SweepSpec sw;
    sw.base = base;
    sw.label = "arrival_x_cost";
    sw.axes = {{"arrival_rate", {1.0, 2.0, 3.0}}, {"cost_weight", {0.05, 0.1, 0.5}}};

    spec::CampaignSpec camp;
    camp.name = "queue study";
    camp.sweeps = {sw};

    auto arch = spec::execute_campaign(camp, reg, 4);
    sim::write_campaign(arch);
#ifdef GOSS_HAVE_HDF5
    sim::export_dashboard_data(results, data);
#endif
    return 0;
}
