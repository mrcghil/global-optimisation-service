// src/solver/nlopt_solver.cpp
// NloptSolver implementation — COBYLA derivative-free optimiser.
//
// All NLopt types are confined to this translation unit.  The public header
// (nlopt_solver.hpp) exposes only standard-library and project types.
//
// This task handles objective evaluation, variable bounds, and result/status
// mapping for bound-only problems (m=0).  General constraint handling is
// added in Task 6.
//
// NLopt 2.7.1 C++ header is <nlopt.hpp>; the goss_nlopt_iface INTERFACE
// target (Task 1) propagates the correct -I flag automatically.
#include <nlopt.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "goss/solver/nlopt_solver.hpp"
#include "goss/solver/errors.hpp"
#include "goss/nlp/nlp_problem.hpp"

namespace goss::solver {

// ---------------------------------------------------------------------------
// Anonymous-namespace helpers
// ---------------------------------------------------------------------------
namespace {

/// Data passed via void* to the NLopt objective callback.
/// The pointer is stored by NLopt WITHOUT copying — the struct must outlive
/// the optimize() call.  It is declared as a local in solve() just before
/// optimize() is called, which guarantees the correct lifetime.
struct ObjectiveData {
    const nlp::NLPProblem* problem;
};

/// NLopt objective callback.
/// COBYLA never supplies gradient information (grad is empty); leave it
/// untouched.  Any exception from eval_objective propagates up to the
/// try/catch in solve(), which translates it to a SolverStatus.
double objective_callback(const std::vector<double>& x,
                          std::vector<double>& /*grad*/,
                          void* data) {
    const auto* d = static_cast<const ObjectiveData*>(data);
    return d->problem->eval_objective(x);
}

/// Replace ±2e19 sentinel values (used by NLPProblem for "free" bounds)
/// with ±infinity so that COBYLA sees true unbounded directions rather
/// than a finite — but very large — box.
///
/// @param bounds  Raw bound vector from NLPProblem.
/// @param lower   true → lower-bound vector (replace values <= -1e19 with -∞),
///                false → upper-bound vector (replace values >= 1e19 with +∞).
std::vector<double> clamp_infinities(const std::vector<double>& bounds,
                                     bool lower) {
    const double pos_inf = std::numeric_limits<double>::infinity();
    const double neg_inf = -std::numeric_limits<double>::infinity();

    std::vector<double> result(bounds.size());
    for (std::size_t i = 0; i < bounds.size(); ++i) {
        if (lower && bounds[i] <= -1e19) {
            result[i] = neg_inf;
        } else if (!lower && bounds[i] >= 1e19) {
            result[i] = pos_inf;
        } else {
            result[i] = bounds[i];
        }
    }
    return result;
}

/// Map an nlopt::result code to goss::solver::SolverStatus.
///
/// Positive codes indicate termination criteria reached (success or limits);
/// negative codes indicate failure modes.
SolverStatus map_nlopt_result(nlopt::result code) {
    switch (code) {
        case nlopt::SUCCESS:
        case nlopt::STOPVAL_REACHED:
        case nlopt::FTOL_REACHED:
        case nlopt::XTOL_REACHED:
            return SolverStatus::Success;

        case nlopt::MAXEVAL_REACHED:
        case nlopt::MAXTIME_REACHED:
            return SolverStatus::IterationLimit;

        case nlopt::ROUNDOFF_LIMITED:
            return SolverStatus::NumericalError;

        case nlopt::FAILURE:
        case nlopt::INVALID_ARGS:
        case nlopt::OUT_OF_MEMORY:
        case nlopt::FORCED_STOP:
        default:
            return SolverStatus::Failure;
    }
}

/// Return a human-readable description of an nlopt::result code.
std::string nlopt_result_message(nlopt::result code) {
    switch (code) {
        case nlopt::SUCCESS:           return "Optimal solution found";
        case nlopt::STOPVAL_REACHED:   return "Stop-value reached";
        case nlopt::FTOL_REACHED:      return "Function tolerance reached";
        case nlopt::XTOL_REACHED:      return "Variable tolerance reached";
        case nlopt::MAXEVAL_REACHED:   return "Maximum evaluations exceeded";
        case nlopt::MAXTIME_REACHED:   return "Maximum time exceeded";
        case nlopt::ROUNDOFF_LIMITED:  return "Halted due to roundoff errors";
        case nlopt::FAILURE:           return "Generic NLopt failure";
        case nlopt::INVALID_ARGS:      return "Invalid arguments supplied to NLopt";
        case nlopt::OUT_OF_MEMORY:     return "NLopt ran out of memory";
        case nlopt::FORCED_STOP:       return "Optimisation was force-stopped";
        default:
            return "NLopt returned unknown code " + std::to_string(static_cast<int>(code));
    }
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// NloptSolver::solve
// ---------------------------------------------------------------------------

SolverResult NloptSolver::solve(const nlp::NLPProblem& problem,
                                const std::vector<double>& initial_guess) {
    if (initial_guess.size() != problem.num_variables()) {
        throw SolverError(
            "NloptSolver::solve: initial_guess size (" +
            std::to_string(initial_guess.size()) +
            ") does not match problem.num_variables() (" +
            std::to_string(problem.num_variables()) + ")");
    }

    const auto n = static_cast<unsigned>(problem.num_variables());

    nlopt::opt optimizer(nlopt::LN_COBYLA, n);

    // Set variable bounds, converting ±2e19 sentinels to true infinities.
    optimizer.set_lower_bounds(
        clamp_infinities(problem.variable_lower_bounds(), /*lower=*/true));
    optimizer.set_upper_bounds(
        clamp_infinities(problem.variable_upper_bounds(), /*lower=*/false));

    // ObjectiveData is a local whose lifetime spans the optimize() call.
    // NLopt stores the void* without copying, so we must not let data go
    // out of scope before optimize() returns.
    ObjectiveData data{&problem};
    optimizer.set_min_objective(&objective_callback, &data);

    optimizer.set_xtol_rel(xtol_rel_);
    optimizer.set_maxeval(max_evaluations_);

    std::vector<double> x = initial_guess;
    double minf = 0.0;
    SolverResult result;

    // NLopt 2.7 throws on non-success return codes (no set_exceptions_enabled).
    // We catch each specific NLopt exception type plus the generic std::exception
    // to guarantee no exception escapes solve().  The last_optimize_result()
    // accessor retrieves the underlying code when we handle a thrown exception.
    try {
        const nlopt::result code = optimizer.optimize(x, minf);
        result.status           = map_nlopt_result(code);
        result.x                = x;
        result.objective_value  = minf;
        result.message          = nlopt_result_message(code);
        // constraint_multipliers left empty: COBYLA does not produce them.
    } catch (const nlopt::roundoff_limited&) {
        // ROUNDOFF_LIMITED: optimisation halted due to floating-point precision.
        result.status          = SolverStatus::NumericalError;
        result.x               = x;
        result.objective_value = minf;
        result.message         = nlopt_result_message(nlopt::ROUNDOFF_LIMITED);
    } catch (const nlopt::forced_stop&) {
        // FORCED_STOP: optimisation was externally terminated.
        result.status          = SolverStatus::Failure;
        result.x               = x;
        result.objective_value = minf;
        result.message         = nlopt_result_message(nlopt::FORCED_STOP);
    } catch (const std::bad_alloc& e) {
        // OUT_OF_MEMORY: NLopt could not allocate internal storage.
        result.status          = SolverStatus::Failure;
        result.x               = x;
        result.objective_value = minf;
        result.message         = std::string("NLopt out of memory: ") + e.what();
    } catch (const std::invalid_argument& e) {
        // INVALID_ARGS: bad dimension or invalid option.
        result.status          = SolverStatus::Failure;
        result.x               = x;
        result.objective_value = minf;
        result.message         = std::string("NLopt invalid argument: ") + e.what();
    } catch (const std::runtime_error& e) {
        // FAILURE or unrecognised runtime error from NLopt.
        result.status          = SolverStatus::Failure;
        result.x               = x;
        result.objective_value = minf;
        result.message         = std::string("NLopt runtime error: ") + e.what();
    } catch (const std::exception& e) {
        // Belt-and-suspenders: catch anything else that might escape NLopt.
        result.status          = SolverStatus::Failure;
        result.x               = x;
        result.objective_value = minf;
        result.message         = std::string("NLopt threw exception: ") + e.what();
    }

    return result;
}

}  // namespace goss::solver
