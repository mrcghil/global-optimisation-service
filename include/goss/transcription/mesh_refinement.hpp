// include/goss/transcription/mesh_refinement.hpp
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
#include "goss/solver/solver_result.hpp"
#include "goss/transcription/errors.hpp"
#include "goss/transcription/mesh.hpp"
#include "goss/transcription/ocp_problem.hpp"
#include "goss/transcription/variable_layout.hpp"

namespace goss::transcription {

/// Estimate the local discretization error in each interval by comparing the
/// collocated solution to one step of RK4 re-integration.
///
/// Returns a vector of length mesh.num_intervals() where entry k is the
/// maximum absolute state deviation at the right endpoint of interval k.
///
/// Each interval integrates from its own solved start node (NOT a single global
/// integration from node 0), so the signal localises error to the interval that
/// actually needs refinement. Dynamics are called with double (not AD types).
///
/// This mirrors sim::validate_by_integration but returns per-interval data
/// instead of the global max, enabling targeted mesh bisection.
template <typename DynamicsFn, typename CostFn>
std::vector<double> estimate_interval_errors(
    const OcpProblem<DynamicsFn, CostFn>& ocp,
    const NonUniformMesh& mesh,
    const solver::SolverResult& result,
    const VariableLayout& layout) {

    if (result.x.size() != layout.total_variables())
        throw TranscriptionError(
            "estimate_interval_errors: result.x size != layout.total_variables");
    if (mesh.num_nodes() != layout.num_nodes())
        throw TranscriptionError(
            "estimate_interval_errors: mesh.num_nodes != layout.num_nodes");

    const std::size_t ns = layout.num_states();
    const std::size_t nc = layout.num_controls();
    const std::size_t ni = mesh.num_intervals();

    // Read solved state at a given node into a double vector.
    auto solved_state = [&](std::size_t node) {
        std::vector<double> x(ns);
        for (std::size_t i = 0; i < ns; ++i)
            x[i] = result.x[layout.state_index(node, i)];
        return x;
    };

    // Linearly interpolate control between nodes k and k+1 at local parameter s in [0,1].
    // When there are no controls, returns an empty vector (safe for dynamics callables that
    // ignore u).
    auto interpolated_control = [&](std::size_t k, double s) {
        std::vector<double> u(nc);
        for (std::size_t j = 0; j < nc; ++j) {
            const double u0 = result.x[layout.control_index(k,     j)];
            const double u1 = result.x[layout.control_index(k + 1, j)];
            u[j] = u0 * (1.0 - s) + u1 * s;
        }
        return u;
    };

    // Element-wise a + c*b for state vectors of length ns.
    auto add_scaled = [&](const std::vector<double>& a, double c,
                          const std::vector<double>& b) {
        std::vector<double> out(ns);
        for (std::size_t i = 0; i < ns; ++i) out[i] = a[i] + c * b[i];
        return out;
    };

    std::vector<double> interval_errors(ni, 0.0);

    for (std::size_t k = 0; k < ni; ++k) {
        const double tk  = mesh.node_times[k];
        const double tk1 = mesh.node_times[k + 1];
        const double hk  = tk1 - tk;
        const double tm  = 0.5 * (tk + tk1);

        // RK4 starting from the solved node-k state (each interval uses its own start).
        // Dynamics are called with double, matching the OcpProblem template contract.
        const std::vector<double> x = solved_state(k);

        const auto k1 = ocp.dynamics(x,                          interpolated_control(k, 0.0), tk);
        const auto k2 = ocp.dynamics(add_scaled(x, 0.5*hk, k1), interpolated_control(k, 0.5), tm);
        const auto k3 = ocp.dynamics(add_scaled(x, 0.5*hk, k2), interpolated_control(k, 0.5), tm);
        const auto k4 = ocp.dynamics(add_scaled(x, hk,     k3), interpolated_control(k, 1.0), tk1);

        std::vector<double> x_rk4(ns);
        for (std::size_t i = 0; i < ns; ++i)
            x_rk4[i] = x[i] + (hk / 6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);

        const std::vector<double> x_colloc = solved_state(k + 1);
        double max_state_deviation = 0.0;
        for (std::size_t i = 0; i < ns; ++i)
            max_state_deviation = std::max(max_state_deviation,
                                           std::abs(x_rk4[i] - x_colloc[i]));
        interval_errors[k] = max_state_deviation;
    }
    return interval_errors;
}

}  // namespace goss::transcription
