// include/goss/sim/validation.hpp
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
#include "goss/sim/errors.hpp"
#include "goss/solver/solver_result.hpp"
#include "goss/transcription/ocp_problem.hpp"
#include "goss/transcription/variable_layout.hpp"

namespace goss::sim {

/// Independent RK4 re-integration of the OCP dynamics from the solved node-0
/// state, using the solved controls (linearly interpolated between nodes).
/// Returns the maximum absolute deviation, over all states and nodes, between
/// the re-integrated trajectory and the collocated solution. Small for a
/// correctly-solved problem; large for a corrupted/incorrect one. This is an
/// independent check of the transcription+solve, not a re-run of collocation.
template <typename DynamicsFn, typename CostFn>
double validate_by_integration(const transcription::OcpProblem<DynamicsFn, CostFn>& ocp,
                               const solver::SolverResult& result,
                               const transcription::VariableLayout& layout) {
    if (result.x.size() != layout.total_variables())
        throw SimError("validate_by_integration: result.x size != layout.total_variables");

    const std::size_t ns = layout.num_states();
    const std::size_t nc = layout.num_controls();
    const std::size_t ni = ocp.mesh.num_intervals;
    const double t0 = ocp.mesh.t_initial;
    const double h = ocp.mesh.interval_width();

    // Linear control sampler at arbitrary time t within [t0, t0 + ni*h].
    // For nc==0 returns empty vector immediately (no crash path).
    auto control_at = [&](double t) -> std::vector<double> {
        std::vector<double> u(nc, 0.0);
        if (nc == 0) return u;
        double raw = (t - t0) / h;
        long idx_long = static_cast<long>(std::floor(raw));
        if (idx_long < 0) idx_long = 0;
        if (idx_long > static_cast<long>(ni) - 1) idx_long = static_cast<long>(ni) - 1;
        const std::size_t idx = static_cast<std::size_t>(idx_long);
        double s = (t - (t0 + static_cast<double>(idx) * h)) / h;
        s = std::max(0.0, std::min(1.0, s));
        for (std::size_t j = 0; j < nc; ++j) {
            const double u0 = result.x[layout.control_index(idx, j)];
            const double u1 = result.x[layout.control_index(idx + 1, j)];
            u[j] = u0 * (1.0 - s) + u1 * s;
        }
        return u;
    };

    // Helper: solved_state(node, state_idx) reads from result.x via layout.
    auto solved_state = [&](std::size_t node, std::size_t i) {
        return result.x[layout.state_index(node, i)];
    };

    // Helper: a + c * b (element-wise) for RK4 stage arguments.
    auto add_scaled = [&](const std::vector<double>& a, double c, const std::vector<double>& b) {
        std::vector<double> out(ns);
        for (std::size_t i = 0; i < ns; ++i) out[i] = a[i] + c * b[i];
        return out;
    };

    // Start RK4 integration from the solved node-0 state (not the analytic IC).
    // This ensures the comparison is purely between the integrated trajectory
    // and the collocated trajectory — both sharing the same initial point.
    std::vector<double> x(ns);
    for (std::size_t i = 0; i < ns; ++i) x[i] = solved_state(0, i);

    double max_deviation = 0.0;

    for (std::size_t k = 0; k < ni; ++k) {
        const double tk  = t0 + static_cast<double>(k) * h;
        const double tm  = tk + 0.5 * h;
        const double tk1 = tk + h;

        // RK4 stages — dynamics called with std::vector<double> (not AD types).
        const std::vector<double> k1 = ocp.dynamics(x,                      control_at(tk),  tk);
        const std::vector<double> k2 = ocp.dynamics(add_scaled(x, 0.5*h, k1), control_at(tm),  tm);
        const std::vector<double> k3 = ocp.dynamics(add_scaled(x, 0.5*h, k2), control_at(tm),  tm);
        const std::vector<double> k4 = ocp.dynamics(add_scaled(x, h,     k3), control_at(tk1), tk1);

        // Advance the integrated state.
        for (std::size_t i = 0; i < ns; ++i)
            x[i] += (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);

        // Compare integrated state at node k+1 against the solved node state.
        for (std::size_t i = 0; i < ns; ++i)
            max_deviation = std::max(max_deviation, std::abs(x[i] - solved_state(k + 1, i)));
    }
    return max_deviation;
}

}  // namespace goss::sim
