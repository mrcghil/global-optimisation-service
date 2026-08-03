#pragma once
#include <vector>
#include "goss/model/errors.hpp"
#include "goss/model/parameter.hpp"
#include "goss/nlp/nlp_problem.hpp"
#include "goss/sim/parameters.hpp"
#include "goss/sim/sweep_result.hpp"
#include "goss/solver/solver.hpp"

namespace goss::sim {

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
            apply_parameters(problem, validator, parameters);   // validate + inject
        } catch (const model::ModelError& validation_error) {
            point.status = solver::SolverStatus::Failure;
            point.message = validation_error.what();            // explicit, names param
            result.points.push_back(std::move(point));
            continue;
        }
        const solver::SolverResult solve_result = solver.solve(problem, initial_guess);
        point.status = solve_result.status;
        point.objective_value = solve_result.objective_value;
        point.x = solve_result.x;
        point.message = solve_result.message;
        result.points.push_back(std::move(point));
    }
    return result;
}

}  // namespace goss::sim
