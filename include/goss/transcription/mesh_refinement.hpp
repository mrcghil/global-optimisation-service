// include/goss/transcription/mesh_refinement.hpp
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>
#include "goss/solver/ipopt_solver.hpp"
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

/// Bundle returned by refine_and_solve: the final solution, refined mesh,
/// layout, iteration count, and per-interval errors at convergence.
struct RefinementResult {
    solver::SolverResult final_solve_result;
    NonUniformMesh       final_mesh;
    VariableLayout       final_layout;
    std::size_t          num_refinement_iterations;  // how many bisect+solve cycles ran
    std::vector<double>  final_interval_errors;      // per-interval error at convergence
};

/// Adaptively refine the mesh and re-solve until max(interval_errors) <= error_tolerance
/// or max_iterations is reached.
///
/// On each iteration:
///   1. Solve the transcribed OCP on the current mesh.
///   2. Estimate per-interval errors via estimate_interval_errors().
///   3. If max(errors) <= error_tolerance: return.
///   4. Mark all intervals where error > bisection_threshold * max(errors) for bisection
///      (bisection_threshold in (0,1]; default 0.5 bisects the worst half).
///   5. Call bisect_intervals(), rebuild the compiled OCP, warm-start from interpolated z.
///
/// The Scheme template parameter is the scheme struct (Trapezoidal or HermiteSimpson).
/// It must expose: static CompiledOcp compile(ocp, NonUniformMesh, std::string).
template <typename Scheme, typename DynamicsFn, typename CostFn>
RefinementResult refine_and_solve(
    const OcpProblem<DynamicsFn, CostFn>& ocp,
    const NonUniformMesh& initial_mesh,
    const std::string& base_model_name,
    double error_tolerance,
    std::size_t max_iterations         = 10,
    double bisection_threshold         = 0.5) {

    if (max_iterations == 0) throw TranscriptionError("refine_and_solve: max_iterations must be >= 1");

    NonUniformMesh current_mesh = initial_mesh;
    current_mesh.validate();

    const std::size_t ns = ocp.num_states;
    const std::size_t nc = ocp.num_controls;

    solver::IpoptSolver ipopt_solver;
    // Use a tighter IPOPT tolerance so the collocation solution is accurate enough
    // for the RK4 error estimator to give meaningful per-interval residuals.
    ipopt_solver.set_tolerance(1e-10);

    solver::SolverResult current_result;
    VariableLayout current_layout(ns, nc, current_mesh.num_nodes());

    // The warm-start vector for the next iteration: populated after each solve
    // and interpolated into the new (refined) layout before the next compile.
    std::vector<double> z_warm;  // empty => cold start on first iteration

    std::size_t iteration = 0;
    std::vector<double> interval_errors;

    for (; iteration < max_iterations; ++iteration) {
        const std::string model_name =
            base_model_name + "_iter" + std::to_string(iteration);

        auto compiled = Scheme::compile(ocp, current_mesh, model_name);
        current_layout = compiled.layout;

        // Build initial guess for this iteration: warm-start if available,
        // otherwise fall back to a cold start using ocp.initial_state per state.
        std::vector<double> z0;
        if (z_warm.empty()) {
            // Cold start: for each node, set state i = ocp.initial_state[i] (or 0.0
            // if i >= initial_state.size()), and control j = 0.0.
            // This correctly seeds multi-state problems at each component's own value
            // rather than using the first state's value for all variables.
            const std::size_t nv = compiled.problem->num_variables();
            z0.assign(nv, 0.0);
            for (std::size_t node = 0; node < current_layout.num_nodes(); ++node) {
                for (std::size_t i = 0; i < ns; ++i) {
                    const double val = (i < ocp.initial_state.size()) ? ocp.initial_state[i] : 0.0;
                    z0[current_layout.state_index(node, i)] = val;
                }
                // Controls default to 0.0 (already set above).
            }
        } else {
            // z_warm was constructed for the current_mesh layout (set at the end
            // of the previous iteration before bisect_intervals was called).
            z0 = std::move(z_warm);
            z_warm.clear();
        }

        current_result = ipopt_solver.solve(*compiled.problem, z0);
        // On solver failure we still report the best result so far.
        if (current_result.status != solver::SolverStatus::Success) break;

        interval_errors = estimate_interval_errors(
            ocp, current_mesh, current_result, current_layout);

        const double max_error = *std::max_element(
            interval_errors.begin(), interval_errors.end());

        // Termination: error within tolerance.
        if (max_error <= error_tolerance) break;

        // Mark intervals whose error exceeds bisection_threshold * max_error.
        std::vector<std::size_t> intervals_to_bisect;
        for (std::size_t k = 0; k < interval_errors.size(); ++k) {
            if (interval_errors[k] > bisection_threshold * max_error)
                intervals_to_bisect.push_back(k);
        }

        if (intervals_to_bisect.empty()) break;  // nothing left to refine

        // Build a warm-start vector for the refined mesh by interpolating the
        // current solution into the new (post-bisection) node layout.
        //
        // Strategy: walk the new mesh in order. For each new node:
        //   - If it corresponds to an existing node k, copy z[k] directly.
        //   - If it is a newly inserted midpoint between old nodes k and k+1,
        //     linearly interpolate state and average the control.
        NonUniformMesh refined_mesh = bisect_intervals(current_mesh, intervals_to_bisect);

        const std::size_t new_nn = refined_mesh.num_nodes();
        const std::size_t vars_per_node = ns + nc;
        z_warm.resize(new_nn * vars_per_node);

        // Map each new node time to either an existing node or a midpoint.
        // Build a lookup from time -> old-node index for fast matching.
        std::size_t old_node = 0;  // pointer into current_mesh.node_times
        for (std::size_t new_k = 0; new_k < new_nn; ++new_k) {
            const double t_new = refined_mesh.node_times[new_k];
            // Advance old_node until current_mesh.node_times[old_node] >= t_new.
            while (old_node + 1 < current_mesh.num_nodes() &&
                   current_mesh.node_times[old_node + 1] < t_new - 1e-14) {
                ++old_node;
            }

            // Tolerance for "same time" matching.
            const double eps = 1e-12;
            const bool is_existing_node =
                (std::abs(current_mesh.node_times[old_node] - t_new) < eps);

            if (is_existing_node) {
                // Copy state and control from old node old_node.
                for (std::size_t i = 0; i < ns; ++i)
                    z_warm[new_k * vars_per_node + i] =
                        current_result.x[current_layout.state_index(old_node, i)];
                for (std::size_t j = 0; j < nc; ++j)
                    z_warm[new_k * vars_per_node + ns + j] =
                        current_result.x[current_layout.control_index(old_node, j)];
            } else {
                // New midpoint between old_node and old_node+1: linearly interpolate.
                const std::size_t ok  = old_node;
                const std::size_t ok1 = old_node + 1;
                const double t0 = current_mesh.node_times[ok];
                const double t1 = current_mesh.node_times[ok1];
                const double s  = (t1 > t0) ? (t_new - t0) / (t1 - t0) : 0.5;

                for (std::size_t i = 0; i < ns; ++i) {
                    const double x0i = current_result.x[current_layout.state_index(ok,  i)];
                    const double x1i = current_result.x[current_layout.state_index(ok1, i)];
                    z_warm[new_k * vars_per_node + i] = x0i * (1.0 - s) + x1i * s;
                }
                for (std::size_t j = 0; j < nc; ++j) {
                    const double u0j = current_result.x[current_layout.control_index(ok,  j)];
                    const double u1j = current_result.x[current_layout.control_index(ok1, j)];
                    z_warm[new_k * vars_per_node + ns + j] = 0.5 * (u0j + u1j);
                }
            }
        }

        current_mesh = std::move(refined_mesh);
        // current_layout will be rebuilt at the top of the next loop iteration via
        // Scheme::compile, so we do not update it here.
    }

    return RefinementResult{
        std::move(current_result),
        current_mesh,
        current_layout,
        iteration,
        interval_errors
    };
}

}  // namespace goss::transcription
