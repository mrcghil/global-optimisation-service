// include/goss/solver/solver_result.hpp
#pragma once
#include <string>
#include <vector>
namespace goss::solver {
enum class SolverStatus {
    Success,            // converged to an optimal (or acceptable) point
    InfeasibleProblem,  // solver proved / detected infeasibility
    IterationLimit,     // hit max iterations / evaluations without converging
    NumericalError,     // NaN / invalid number / roundoff-limited
    Failure             // any other unrecoverable failure
};
struct SolverResult {
    SolverStatus status = SolverStatus::Failure;
    std::vector<double> x;                        // final primal solution
    double objective_value = 0.0;                 // objective at x
    std::vector<double> constraint_multipliers;   // λ (may be empty for derivative-free)
    std::string message;                          // human-readable solver status text
};
}  // namespace goss::solver
