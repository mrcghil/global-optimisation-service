// tools/dashboard_export/queue_model.hpp
#pragma once
#include <string>
#include <utility>
#include <vector>

#include "goss/model/model.hpp"
#include "goss/spec/compile_dispatch.hpp"
#include "goss/spec/registry.hpp"

namespace goss_tools {

/// Builds the demo "queue" optimal-control problem at a given discretization.
/// State: queue_length (init 10). Control: service_rate [0,5].
/// Params: arrival_rate (default 2, [0,10]), cost_weight (default 0.1, [0,10]).
/// Dynamics dq/dt = arrival_rate - service_rate; objective
/// integral(queue_length + cost_weight * service_rate^2).
inline goss::spec::BuiltProblem build_queue(
        const goss::spec::DiscretizationSpec& d) {
    goss::model::Model m;
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
    auto c = goss::spec::compile_dispatch(
        ocp, d, "gensample_" + std::to_string(d.num_intervals));
    auto mesh = ocp.mesh;
    return goss::spec::BuiltProblem{std::move(m), std::move(c), mesh, d.scheme};
}

}  // namespace goss_tools
