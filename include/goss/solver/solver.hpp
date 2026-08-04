// include/goss/solver/solver.hpp
#pragma once
#include <vector>
#include "goss/nlp/nlp_problem.hpp"
#include "goss/solver/solver_result.hpp"
namespace goss::solver {
/// Abstract solver over an NLPProblem. Concrete adapters (IPOPT, NLopt) must
/// confine all third-party solver types to their own .cpp — no solver library
/// type may appear in this or any public header.
class Solver {
 public:
    virtual ~Solver() = default;
    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;
    Solver(Solver&&) = delete;
    Solver& operator=(Solver&&) = delete;

    /// Solve the problem from initial_guess. parameters (empty by default) is
    /// injected into the problem's AD backend ONCE before solving and stays
    /// constant across the solver's iterations; only setup/usage errors throw
    /// (SolverError, or ADError on a parameter size mismatch).
    virtual SolverResult solve(const nlp::NLPProblem& problem,
                               const std::vector<double>& initial_guess,
                               const std::vector<double>& parameters = {}) = 0;

 protected:
    Solver() = default;
};
}  // namespace goss::solver
