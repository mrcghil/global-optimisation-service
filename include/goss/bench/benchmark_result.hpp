// include/goss/bench/benchmark_result.hpp
#pragma once
#include <cstddef>
#include <string>
#include "goss/solver/solver_result.hpp"

namespace goss::bench {

/// One row of benchmark output: the outcome of running one (scheme, solver) pair
/// on a single OCP instance.
struct BenchmarkResult {
    std::string scheme_name;               // e.g. "Trapezoidal", "HermiteSimpson"
    std::string solver_name;               // e.g. "IpoptSolver", "NloptSolver"
    goss::solver::SolverStatus solve_status = goss::solver::SolverStatus::Failure;
    double objective_value  = 0.0;         // objective at the solved x (0 if failed)
    double elapsed_seconds  = 0.0;         // wall-clock duration of solver.solve() alone
    double validation_error = 0.0;         // max RK4 re-integration deviation (0 if failed)
    std::size_t num_variables = 0;         // total decision variables = layout.total_variables()
};

/// Returns a short human-readable label for a SolverStatus enumerator.
/// Used by to_table() and to_csv() to produce readable status columns.
inline std::string solver_status_name(goss::solver::SolverStatus status) {
    using goss::solver::SolverStatus;
    switch (status) {
        case SolverStatus::Success:           return "Success";
        case SolverStatus::InfeasibleProblem: return "InfeasibleProblem";
        case SolverStatus::IterationLimit:    return "IterationLimit";
        case SolverStatus::NumericalError:    return "NumericalError";
        case SolverStatus::Failure:           return "Failure";
        default:                              return "Unknown";
    }
}

}  // namespace goss::bench
