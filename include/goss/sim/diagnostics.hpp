// include/goss/sim/diagnostics.hpp
#pragma once
#include <string>
#include "goss/solver/solver_result.hpp"

namespace goss::sim {

struct Diagnosis {
    bool ok;
    std::string summary;
    std::string advice;
};

inline Diagnosis diagnose(const solver::SolverResult& result) {
    using solver::SolverStatus;
    switch (result.status) {
        case SolverStatus::Success:
            return {true, "Solver converged.", ""};
        case SolverStatus::InfeasibleProblem:
            return {false, "Problem detected as infeasible.",
                    "Check that constraints/bounds are consistent (e.g. boundary "
                    "conditions reachable, path bounds not contradictory)."};
        case SolverStatus::IterationLimit:
            return {false, "Iteration limit reached before convergence.",
                    "Try a better initial guess (sim::linear_guess), a finer/coarser "
                    "mesh, or increase max iterations."};
        case SolverStatus::NumericalError:
            return {false, "Numerical error during solve (NaN/Inf).",
                    "Check dynamics/cost for divisions by zero or domain errors; "
                    "verify scaling."};
        case SolverStatus::Failure:
        default: {
            std::string summary = "Solver failed.";
            if (!result.message.empty()) summary += " (" + result.message + ")";
            return {false, summary, "See result.message for the underlying solver status."};
        }
    }
}

inline Diagnosis diagnose(const solver::SolverResult& result,
                          double integration_error,
                          double tolerance = 1e-3) {
    if (result.status == solver::SolverStatus::Success && integration_error > tolerance) {
        return {false,
                "Solver reported success but the solution fails RK4 re-integration "
                "(error " + std::to_string(integration_error) + ").",
                "The discretization may be too coarse — refine the mesh; or the "
                "dynamics may be stiff."};
    }
    return diagnose(result);
}

}  // namespace goss::sim
