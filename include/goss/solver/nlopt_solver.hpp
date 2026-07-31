// include/goss/solver/nlopt_solver.hpp
// Public header for the NLopt-backed COBYLA solver.  All NLopt types are
// confined to the corresponding .cpp — this header must not include any
// NLopt header.
#pragma once
#include <vector>
#include "goss/solver/solver.hpp"

namespace goss::solver {

/// Solver implementation backed by NLopt COBYLA (derivative-free,
/// Constrained Optimization BY Linear Approximations).
///
/// Serves as an independent baseline to cross-check gradient/Hessian-based
/// solvers (IPOPT) without relying on the AD derivative path.  All NLopt
/// types are confined to nlopt_solver.cpp — this header exposes only
/// standard-library and project types.
class NloptSolver : public Solver {
 public:
    NloptSolver() = default;

    /// Maximum number of objective function evaluations COBYLA may perform.
    /// Default: 20000.
    void set_max_evaluations(int evaluations) { max_evaluations_ = evaluations; }

    /// Relative tolerance on the optimisation variable vector used as a
    /// stopping criterion.  Default: 1e-8.
    void set_xtol_rel(double tolerance) { xtol_rel_ = tolerance; }

    /// Solve the NLP problem from the given initial guess.
    ///
    /// Convergence outcomes (including non-convergence) are reported via the
    /// returned SolverResult::status.  Only setup/usage errors throw SolverError.
    SolverResult solve(const nlp::NLPProblem& problem,
                       const std::vector<double>& initial_guess) override;

 private:
    int    max_evaluations_ = 20000;
    double xtol_rel_        = 1e-8;
};

}  // namespace goss::solver
