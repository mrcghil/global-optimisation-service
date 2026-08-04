#pragma once
#include <cstddef>
#include <string>
#include <vector>
#include "goss/model/errors.hpp"
#include "goss/model/parameter.hpp"
#include "goss/nlp/nlp_problem.hpp"
#include "goss/sim/errors.hpp"
#include "goss/sim/parameters.hpp"
#include "goss/sim/sweep_result.hpp"
#include "goss/solver/solver.hpp"

namespace goss::sim {

/// Cartesian product of per-parameter value axes. axes[i] is the list of
/// values for parameter i; returns every combination, with parameter 0 varying
/// slowest (row-major). Throws SimError if axes is empty or any axis is empty.
inline std::vector<std::vector<double>> make_grid(
        const std::vector<std::vector<double>>& axes) {
    if (axes.empty())
        throw SimError("make_grid: at least one parameter axis is required");
    for (std::size_t i = 0; i < axes.size(); ++i)
        if (axes[i].empty())
            throw SimError("make_grid: axis " + std::to_string(i) + " is empty");

    std::vector<std::vector<double>> grid = {{}};
    for (const std::vector<double>& axis : axes) {
        std::vector<std::vector<double>> next;
        next.reserve(grid.size() * axis.size());
        for (const std::vector<double>& prefix : grid)
            for (double value : axis) {
                std::vector<double> combination = prefix;
                combination.push_back(value);
                next.push_back(std::move(combination));
            }
        grid = std::move(next);
    }
    return grid;
}


/// Configuration for the parallel process-pool sweep executor.
struct SweepConfig {
    /// Maximum number of concurrently live child worker processes.
    /// 0 (the default) resolves at runtime to std::thread::hardware_concurrency(),
    /// with a fallback to 1 if hardware_concurrency() itself returns 0.
    std::size_t max_parallel_workers = 0;
};

inline SweepResult run_sweep_serial(
        nlp::NLPProblem& problem,
        const model::ParameterValidator& validator,
        solver::Solver& solver,
        const std::vector<std::vector<double>>& parameter_grid,
        const std::vector<double>& initial_guess) {
    SweepResult result;
    result.points.reserve(parameter_grid.size());

    for (const std::vector<double>& parameters : parameter_grid) {
        SweepPoint point;
        point.parameters = parameters;
        try {
            // validate then solve in one call — validation failure surfaces before
            // any (expensive) solver work is done.
            const solver::SolverResult solve_result =
                solve_with_parameters(solver, problem, validator, initial_guess, parameters);
            point.status = solve_result.status;
            point.objective_value = solve_result.objective_value;
            point.x = solve_result.x;
            point.message = solve_result.message;
        } catch (const model::ModelError& validation_error) {
            point.status = solver::SolverStatus::Failure;
            point.message = validation_error.what();  // explicit, names param
        }
        result.points.push_back(std::move(point));
    }
    return result;
}

/// Runs the grid across a bounded pool of forked worker processes.
///
/// Results are returned in the SAME ORDER as `parameter_grid` (order-preserving
/// despite out-of-order completion): result.points[i] always corresponds to
/// parameter_grid[i].
///
/// Concurrency is capped to config.max_parallel_workers live children.  When
/// max_parallel_workers == 0 the cap resolves to
/// std::thread::hardware_concurrency() (falling back to 1 if that returns 0).
///
/// Parameters are applied INSIDE each child via copy-on-write fork — the
/// parent's NLPProblem is never mutated.  The compiled .so is inherited by all
/// children and never recompiled.
///
/// Point-level failures (validation error, solve failure, child crash) are
/// recorded as Failure SweepPoints at their correct index — run_sweep_parallel
/// never throws for point-level failures.  SimError is thrown only for
/// pool-setup failures (pipe/fork).
SweepResult run_sweep_parallel(
    nlp::NLPProblem& problem,
    const model::ParameterValidator& validator,
    solver::Solver& solver,
    const std::vector<std::vector<double>>& parameter_grid,
    const std::vector<double>& initial_guess,
    const SweepConfig& config = {});

}  // namespace goss::sim
