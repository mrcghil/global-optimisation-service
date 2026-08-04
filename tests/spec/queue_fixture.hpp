// tests/spec/queue_fixture.hpp
// Shared registry builder for the queue OCP used across the spec tests.  Mirrors
// the problem in tests/sim/test_sweep_workflow.cpp: a 1-state/1-control queue
// with arrival_rate (p0) and cost_weight (p1) parameters.
#pragma once
#include <atomic>
#include <string>
#include <utility>
#include <vector>
#include "goss/model/model.hpp"
#include "goss/spec/compile_dispatch.hpp"
#include "goss/spec/registry.hpp"
#include "goss/spec/specs.hpp"

namespace goss::spec::test {

inline BuiltProblem build_queue(const DiscretizationSpec& disc) {
    goss::model::Model model;
    auto q    = model.add_state("queue_length");
    auto rate = model.add_control("service_rate");
    model.add_parameter("arrival_rate", 2.0, 0.0, 10.0);
    model.add_parameter("cost_weight",  0.1, 0.0, 10.0);
    model.set_state_bounds(q, 0.0, 1e19);
    model.set_control_bounds(rate, 0.0, 5.0);
    model.set_initial_state(q, 10.0);
    model.set_mesh(disc.t_initial, disc.t_final, disc.num_intervals);

    auto ocp = model.build(
        [](const auto& x, const auto& u, const auto& p, auto) {
            using T = std::decay_t<decltype(x[0])>;
            return std::vector<T>{p[0] - u[0]};
        },
        [](const auto& x, const auto& u, const auto& p, auto) {
            using T = std::decay_t<decltype(x[0])>;
            return x[0] + p[1] * u[0] * u[0];
        });

    // The model_name doubles as the JIT .so path, so it must be unique per
    // distinct compilation (see registry.hpp builder contract).  A per-call
    // counter keeps names distinct even when the same discretization is compiled
    // more than once in a test (e.g. registry build + direct oracle build).
    static std::atomic<unsigned> counter{0};
    const std::string model_name = "spec_queue_" + disc.scheme + "_" +
                                    std::to_string(disc.num_intervals) + "_" +
                                    std::to_string(counter.fetch_add(1));
    auto compiled = compile_dispatch(ocp, disc, model_name);
    const auto mesh = ocp.mesh;
    return BuiltProblem{std::move(model), std::move(compiled), mesh, disc.scheme};
}

}  // namespace goss::spec::test
