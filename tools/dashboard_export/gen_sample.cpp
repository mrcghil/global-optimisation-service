// tools/dashboard_export/gen_sample.cpp
//
// Developer utility: solve a small demo campaign (the queue problem) and export
// the dashboard JSON contract into dashboard/sample-data/.  Used to produce
// fixture data for developing the Astro app; not part of the normal build.
#include <string>
#include <utility>
#include <vector>

#include "goss/model/model.hpp"
#include "goss/sim/archive.hpp"
#include "goss/sim/dashboard_export.hpp"
#include "goss/spec/compile_dispatch.hpp"
#include "goss/spec/executor.hpp"
#include "goss/spec/registry.hpp"

using namespace goss;

static spec::BuiltProblem build_queue(const spec::DiscretizationSpec& d) {
    model::Model m;
    auto q = m.add_state("queue_length");
    auto r = m.add_control("service_rate");
    m.add_parameter("arrival_rate", 2.0, 0.0, 10.0);
    m.add_parameter("cost_weight", 0.1, 0.0, 10.0);
    m.set_state_bounds(q, 0.0, 1e19);
    m.set_control_bounds(r, 0.0, 5.0);
    m.set_initial_state(q, 10.0);
    m.set_mesh(d.t_initial, d.t_final, d.num_intervals);
    auto ocp = m.build(
        [](const auto& x, const auto& u, const auto& p, auto) {
            using T = std::decay_t<decltype(x[0])>;
            return std::vector<T>{p[0] - u[0]};
        },
        [](const auto& x, const auto& u, const auto& p, auto) {
            using T = std::decay_t<decltype(x[0])>;
            return x[0] + p[1] * u[0] * u[0];
        });
    auto c = spec::compile_dispatch(ocp, d, "gensample_" + std::to_string(d.num_intervals));
    auto mesh = ocp.mesh;
    return spec::BuiltProblem{std::move(m), std::move(c), mesh, d.scheme};
}

int main(int argc, char** argv) {
    const std::string base_dir = argc > 1 ? argv[1] : "dashboard/sample-data";
    const std::string results = base_dir + "/results";
    const std::string data = base_dir + "/data";

    spec::ProblemRegistry reg;
    reg.register_problem({"queue", "v1"}, build_queue);

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
