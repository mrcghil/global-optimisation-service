// include/goss/solver/ipopt_solver.hpp
// Public header for the IPOPT-backed solver.  All IPOPT types are confined to
// the corresponding .cpp — this header must not include any IPOPT header.
#pragma once
#include <vector>
#include "goss/solver/solver.hpp"

namespace goss::solver {

/// Solver implementation backed by IPOPT (Interior Point OPTimizer).
///
/// The solver wraps an internally-defined IpoptTNLPAdapter that translates a
/// goss::nlp::NLPProblem into the IPOPT TNLP interface.  All IPOPT types are
/// confined to the .cpp — this header exposes only standard-library types.
class IpoptSolver : public Solver {
 public:
    IpoptSolver() = default;

    /// Convergence tolerance passed to IPOPT via option "tol".
    /// Default: 1e-8.
    void set_tolerance(double tolerance) { tolerance_ = tolerance; }

    /// IPOPT print_level (0 = silent, up to 12 = very verbose).
    /// Default: 0 (silent).
    void set_print_level(int level) { print_level_ = level; }

    /// Maximum number of IPOPT iterations (option "max_iter").
    /// Default: 3000.
    void set_max_iterations(int iterations) { max_iterations_ = iterations; }

    /// Solve the NLP problem from the given initial guess.
    ///
    /// Convergence outcomes (including non-convergence) are reported via the
    /// returned SolverResult::status.  Only setup/usage errors throw SolverError.
    SolverResult solve(const nlp::NLPProblem& problem,
                       const std::vector<double>& initial_guess) override;

 private:
    double tolerance_     = 1e-8;
    int    print_level_   = 0;
    int    max_iterations_ = 3000;
};

}  // namespace goss::solver
