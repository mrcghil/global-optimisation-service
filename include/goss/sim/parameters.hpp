// include/goss/sim/parameters.hpp
#pragma once
#include <vector>
#include "goss/model/parameter.hpp"
#include "goss/nlp/nlp_problem.hpp"
#include "goss/solver/solver.hpp"
#include "goss/solver/solver_result.hpp"

namespace goss::sim {

/// Validates `parameters` against the compiled problem's validator (throwing
/// ModelError with an explicit, parameter-naming message on failure), THEN
/// solves with the parameters supplied at solve time.  Validation runs before
/// any solver work, so an invalid point never reaches the (expensive) solver.
///
/// This is the single entry point a sweep runner uses per parameter point.
inline solver::SolverResult solve_with_parameters(
        solver::Solver& solver,
        const nlp::NLPProblem& problem,
        const model::ParameterValidator& validator,
        const std::vector<double>& initial_guess,
        const std::vector<double>& parameters) {
    validator.validate(parameters);                           // explicit errors first
    return solver.solve(problem, initial_guess, parameters);  // solve-time injection
}

}  // namespace goss::sim
