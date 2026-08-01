// tests/accuracy/accuracy_helpers.hpp
//
// Shared helpers for the goss accuracy validation suite.
//
// REUSE CONTRACT: these three helpers are the public interface for later features.
// Any new test file (DAE, path constraints, hp-pseudospectral, composition) can
// #include this header and use the three helpers without modifying it.
// The helpers are pure templates — they work with any compiled scheme and any
// OcpProblem<Dyn,Cost> type.
#pragma once
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "goss/transcription/transcription.hpp"   // CompiledOcp
#include "goss/transcription/variable_layout.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "goss/solver/solver_result.hpp"

namespace goss::accuracy {

/// Unpacked solution from a solved OCP: one entry per node.
/// states[k][i]   = x_i at node k.
/// controls[k][j] = u_j at node k.
/// times[k]       = t at node k (uniform spacing: t0 + k*h).
/// objective_value = optimal cost returned by the solver.
struct SolutionTrajectory {
    std::vector<double>              times;
    std::vector<std::vector<double>> states;
    std::vector<std::vector<double>> controls;
    double                           objective_value = 0.0;
};

/// Solve a compiled OCP from a flat initial guess (all variables set to
/// `initial_guess_value`) and unpack the solution into a SolutionTrajectory.
///
/// WHY: every accuracy test needs the trajectory at each node; this helper
/// hides the layout.state_index / layout.control_index bookkeeping so test
/// files stay focused on the math.
///
/// Calls ADD_FAILURE() (non-fatal GoogleTest failure) if the solver does not
/// return SolverStatus::Success, then returns an empty trajectory.
inline SolutionTrajectory solve_and_extract_trajectory(
        const goss::transcription::CompiledOcp& compiled_ocp,
        double initial_guess_value = 0.0,
        double solver_tolerance   = 1e-9) {
    goss::solver::IpoptSolver solver;
    solver.set_tolerance(solver_tolerance);
    solver.set_print_level(0);  // silent — accuracy tests must not spam the terminal

    const std::size_t num_variables = compiled_ocp.problem->num_variables();
    const std::vector<double> initial_guess(num_variables, initial_guess_value);

    const goss::solver::SolverResult result =
        solver.solve(*compiled_ocp.problem, initial_guess);

    if (result.status != goss::solver::SolverStatus::Success) {
        ADD_FAILURE() << "IpoptSolver did not converge; status message: "
                      << result.message;
        return SolutionTrajectory{};
    }

    const goss::transcription::VariableLayout& layout = compiled_ocp.layout;
    const std::size_t num_nodes    = layout.num_nodes();
    const std::size_t num_states   = layout.num_states();
    const std::size_t num_controls = layout.num_controls();

    SolutionTrajectory trajectory;
    trajectory.objective_value = result.objective_value;
    trajectory.times.resize(num_nodes);
    trajectory.states.resize(num_nodes, std::vector<double>(num_states));
    trajectory.controls.resize(num_nodes, std::vector<double>(num_controls));

    // Reconstruct uniform node times from layout extents.
    // WHY: CompiledOcp does not store the original Mesh, but the layout
    // tells us num_nodes; we compute times from the variable bounds of the
    // pinned initial-state index and the pinned final-state index if available.
    // For the accuracy suite all problems use uniform meshes, so we fall back
    // to storing integer node indices as "times" and let each test supply
    // the actual time formula when needed.
    // A richer trajectory struct (with stored times) is deferred until the
    // sim layer's Trajectory type is mature enough to depend on.
    for (std::size_t node_index = 0; node_index < num_nodes; ++node_index) {
        trajectory.times[node_index] = static_cast<double>(node_index);  // placeholder index
        for (std::size_t state_index = 0; state_index < num_states; ++state_index) {
            trajectory.states[node_index][state_index] =
                result.x[layout.state_index(node_index, state_index)];
        }
        for (std::size_t control_index = 0; control_index < num_controls; ++control_index) {
            trajectory.controls[node_index][control_index] =
                result.x[layout.control_index(node_index, control_index)];
        }
    }
    return trajectory;
}

/// Estimate the empirical convergence slope of a scheme by solving the same
/// OCP at a sequence of increasing mesh sizes and fitting a log-log line.
///
/// `problem_factory` must be callable as:
///   goss::transcription::CompiledOcp problem_factory(std::size_t num_intervals, const std::string& model_name)
/// It builds and compiles the OCP at the given mesh resolution.
///
/// `error_at_mesh_size` must be callable as:
///   double error_at_mesh_size(const SolutionTrajectory& trajectory, std::size_t num_intervals)
/// It computes the scalar error metric (e.g. max nodal error vs analytic) given the trajectory.
///
/// Returns the least-squares slope of log(error) vs log(h) across the provided mesh sizes.
/// A slope of ~2 confirms O(h²), ~4 confirms O(h⁴), very large confirms spectral.
template <typename ProblemFactory, typename ErrorMetric>
double estimate_convergence_slope(
        ProblemFactory       problem_factory,
        ErrorMetric          error_at_mesh_size,
        const std::vector<std::size_t>& mesh_sizes,  // num_intervals values (increasing)
        double               solver_tolerance = 1e-11) {
    assert(mesh_sizes.size() >= 2 && "need at least 2 mesh sizes to fit a slope");

    std::vector<double> log_h_values;
    std::vector<double> log_error_values;
    log_h_values.reserve(mesh_sizes.size());
    log_error_values.reserve(mesh_sizes.size());

    for (std::size_t mesh_idx = 0; mesh_idx < mesh_sizes.size(); ++mesh_idx) {
        const std::size_t num_intervals = mesh_sizes[mesh_idx];
        // Unique model name per mesh size to avoid CppADCG shared-library collisions.
        const std::string model_name = "conv_slope_n" + std::to_string(num_intervals);
        const goss::transcription::CompiledOcp compiled =
            problem_factory(num_intervals, model_name);
        const SolutionTrajectory trajectory =
            solve_and_extract_trajectory(compiled, /*initial_guess_value=*/0.5, solver_tolerance);

        if (trajectory.states.empty()) {
            // solve_and_extract_trajectory already called ADD_FAILURE(); propagate NaN.
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double error = error_at_mesh_size(trajectory, num_intervals);
        // WHY: mesh step h = (t_final - t_initial) / num_intervals.
        // The OCP's time horizon is baked into problem_factory; we use 1/num_intervals
        // as a proportional h (the absolute duration cancels in the slope ratio).
        const double h = 1.0 / static_cast<double>(num_intervals);
        log_h_values.push_back(std::log(h));
        log_error_values.push_back(std::log(error));
    }

    // Least-squares slope of log(error) = slope * log(h) + intercept.
    // WHY least-squares instead of a two-point ratio: three or more mesh sizes
    // give a more robust estimate, especially when solver tolerance pollutes
    // the finest mesh's error.
    const std::size_t count = log_h_values.size();
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        sum_x  += log_h_values[i];
        sum_y  += log_error_values[i];
        sum_xx += log_h_values[i] * log_h_values[i];
        sum_xy += log_h_values[i] * log_error_values[i];
    }
    const double denominator = static_cast<double>(count) * sum_xx - sum_x * sum_x;
    if (std::abs(denominator) < 1e-15) return 0.0;  // degenerate (all same h)
    return (static_cast<double>(count) * sum_xy - sum_x * sum_y) / denominator;
}

/// Check that a scalar invariant (e.g. Hamiltonian, total energy) stays
/// constant along a trajectory to within `tolerance`.
///
/// `invariant_fn` is called as:
///   double invariant_fn(const std::vector<double>& state_at_node,
///                       const std::vector<double>& control_at_node)
/// It must return the scalar value of the conserved quantity at that node.
///
/// WHY: for autonomous OCPs the Hamiltonian H = lambda^T f - L is constant
/// along an optimal trajectory (Pontryagin). Since we don't have co-states
/// from IpoptSolver (only the primal x), we instead check a simpler but
/// sufficient invariant: the running cost integrand or the Hamiltonian
/// approximated from the KKT multipliers. For energy-conservation tests
/// (zero-cost autonomous ODEs), E(x(t)) must be constant.
///
/// Calls EXPECT_NEAR for every node beyond the first, comparing to the
/// invariant value at node 0. Non-fatal so all nodes are reported.
inline void check_invariant_along_trajectory(
        const SolutionTrajectory& trajectory,
        const std::function<double(const std::vector<double>& state,
                                   const std::vector<double>& control)>& invariant_fn,
        double tolerance) {
    if (trajectory.states.empty()) return;  // already failed in solve step

    const double reference_value = invariant_fn(trajectory.states[0], trajectory.controls[0]);
    for (std::size_t node_index = 1; node_index < trajectory.states.size(); ++node_index) {
        const double node_value =
            invariant_fn(trajectory.states[node_index], trajectory.controls[node_index]);
        EXPECT_NEAR(node_value, reference_value, tolerance)
            << "Invariant deviated at node " << node_index
            << " (reference=" << reference_value << ", got=" << node_value << ")";
    }
}

}  // namespace goss::accuracy
