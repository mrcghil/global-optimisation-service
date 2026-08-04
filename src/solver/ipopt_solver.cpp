// src/solver/ipopt_solver.cpp
// IpoptSolver + IpoptTNLPAdapter implementation.
//
// All IPOPT types are confined here.  The public header (ipopt_solver.hpp)
// exposes only standard-library and project types.
//
// Include paths: IPOPT 3.11.9 installs headers under /usr/include/coin/ and
// pkg-config reports -I/usr/include/coin, so includes use no subdirectory
// prefix.  The goss_ipopt_iface INTERFACE target (Task 1) propagates the
// correct -I flag and -DHAVE_CSTDDEF (required by IPOPT 3.11.9 headers).
#include <IpTNLP.hpp>
#include <IpIpoptApplication.hpp>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "goss/solver/ipopt_solver.hpp"
#include "goss/solver/errors.hpp"
#include "goss/nlp/nlp_problem.hpp"
#include "goss/ad/errors.hpp"

namespace goss::solver {

// ---------------------------------------------------------------------------
// Anonymous-namespace helpers
// ---------------------------------------------------------------------------
namespace {

/// Copy n doubles starting at pointer p into a std::vector<double>.
std::vector<double> to_vector(const Ipopt::Number* p, Ipopt::Index n) {
    return std::vector<double>(p, p + static_cast<std::size_t>(n));
}

/// Map Ipopt::SolverReturn to goss::solver::SolverStatus.
SolverStatus map_solver_return(Ipopt::SolverReturn status) {
    switch (status) {
        case Ipopt::SUCCESS:
        case Ipopt::STOP_AT_ACCEPTABLE_POINT:
            return SolverStatus::Success;
        case Ipopt::MAXITER_EXCEEDED:
            return SolverStatus::IterationLimit;
        case Ipopt::LOCAL_INFEASIBILITY:
            return SolverStatus::InfeasibleProblem;
        case Ipopt::INVALID_NUMBER_DETECTED:
            return SolverStatus::NumericalError;
        default:
            return SolverStatus::Failure;
    }
}

// ---------------------------------------------------------------------------
// IpoptTNLPAdapter
// ---------------------------------------------------------------------------

/// TNLP adapter that translates a goss::nlp::NLPProblem into IPOPT's TNLP
/// interface.  Holds references to the problem, initial guess, and the
/// caller-owned SolverResult that finalize_solution populates.
class IpoptTNLPAdapter : public Ipopt::TNLP {
 public:
    /// All three references must outlive the IPOPT solve call.
    IpoptTNLPAdapter(const nlp::NLPProblem& problem,
                     const std::vector<double>& initial_guess,
                     SolverResult& result)
        : problem_(problem),
          initial_guess_(initial_guess),
          result_(result) {}

    // ------------------------------------------------------------------
    // TNLP: problem dimensions and structure
    // ------------------------------------------------------------------

    bool get_nlp_info(Ipopt::Index& n, Ipopt::Index& m,
                      Ipopt::Index& nnz_jac_g, Ipopt::Index& nnz_h_lag,
                      IndexStyleEnum& index_style) override {
        n = static_cast<Ipopt::Index>(problem_.num_variables());
        m = static_cast<Ipopt::Index>(problem_.num_constraints());
        nnz_jac_g = static_cast<Ipopt::Index>(
            problem_.constraint_jacobian_sparsity().size());
        nnz_h_lag = static_cast<Ipopt::Index>(
            problem_.lagrangian_hessian_sparsity().size());
        index_style = TNLP::C_STYLE;  // 0-based indices
        return true;
    }

    bool get_bounds_info(Ipopt::Index n, Ipopt::Number* x_l, Ipopt::Number* x_u,
                         Ipopt::Index m, Ipopt::Number* g_l,
                         Ipopt::Number* g_u) override {
        (void)n;
        (void)m;
        const auto& xl = problem_.variable_lower_bounds();
        const auto& xu = problem_.variable_upper_bounds();
        const auto& gl = problem_.constraint_lower_bounds();
        const auto& gu = problem_.constraint_upper_bounds();
        std::copy(xl.begin(), xl.end(), x_l);
        std::copy(xu.begin(), xu.end(), x_u);
        std::copy(gl.begin(), gl.end(), g_l);
        std::copy(gu.begin(), gu.end(), g_u);
        return true;
    }

    bool get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number* x,
                             bool init_z, Ipopt::Number* z_L,
                             Ipopt::Number* z_U, Ipopt::Index m,
                             bool init_lambda, Ipopt::Number* lambda) override {
        (void)n;
        (void)m;
        (void)z_L;
        (void)z_U;
        (void)lambda;
        // We never enable warm_start_init_point, so IPOPT should never
        // request dual warm-start.  Fail safe if it ever does: returning false
        // signals a contract violation rather than silently leaving arrays
        // uninitialised.
        if (init_z || init_lambda) {
            return false;
        }
        // We only supply the primal starting point; multiplier warm-start is
        // not available from the NLPProblem interface.
        if (init_x) {
            std::copy(initial_guess_.begin(), initial_guess_.end(), x);
        }
        return true;
    }

    // ------------------------------------------------------------------
    // TNLP: function evaluations
    // ------------------------------------------------------------------

    bool eval_f(Ipopt::Index n, const Ipopt::Number* x, bool /*new_x*/,
                Ipopt::Number& obj_value) override {
        // n is used in to_vector; do not suppress it with (void)n.
        try {
            obj_value = problem_.eval_objective(to_vector(x, n));
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    bool eval_grad_f(Ipopt::Index n, const Ipopt::Number* x, bool /*new_x*/,
                     Ipopt::Number* grad_f) override {
        try {
            const auto g = problem_.eval_objective_gradient(to_vector(x, n));
            std::copy(g.begin(), g.end(), grad_f);
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    bool eval_g(Ipopt::Index n, const Ipopt::Number* x, bool /*new_x*/,
                Ipopt::Index m, Ipopt::Number* g) override {
        (void)m;
        try {
            const auto cv = problem_.eval_constraints(to_vector(x, n));
            std::copy(cv.begin(), cv.end(), g);
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    bool eval_jac_g(Ipopt::Index n, const Ipopt::Number* x, bool /*new_x*/,
                    Ipopt::Index m, Ipopt::Index nele_jac,
                    Ipopt::Index* iRow, Ipopt::Index* jCol,
                    Ipopt::Number* values) override {
        (void)n;
        (void)m;
        (void)nele_jac;

        const auto& pattern = problem_.constraint_jacobian_sparsity();

        try {
            if (values == nullptr) {
                // STRUCTURE pass: fill iRow/jCol from the sparsity pattern.
                // pattern[k].first  = constraint row (0-based)
                // pattern[k].second = variable column (0-based)
                for (std::size_t k = 0; k < pattern.size(); ++k) {
                    iRow[k] = static_cast<Ipopt::Index>(pattern[k].first);
                    jCol[k] = static_cast<Ipopt::Index>(pattern[k].second);
                }
            } else {
                // VALUES pass: evaluate and copy, aligned to the same pattern order.
                const auto vals = problem_.eval_constraint_jacobian(to_vector(x, n));
                std::copy(vals.begin(), vals.end(), values);
            }
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    bool eval_h(Ipopt::Index n, const Ipopt::Number* x, bool /*new_x*/,
                Ipopt::Number obj_factor, Ipopt::Index m,
                const Ipopt::Number* lambda, bool /*new_lambda*/,
                Ipopt::Index nele_hess, Ipopt::Index* iRow, Ipopt::Index* jCol,
                Ipopt::Number* values) override {
        (void)n;
        (void)m;
        (void)nele_hess;

        const auto& pattern = problem_.lagrangian_hessian_sparsity();

        try {
            if (values == nullptr) {
                // STRUCTURE pass: fill iRow/jCol from the lower-triangle Hessian pattern.
                for (std::size_t k = 0; k < pattern.size(); ++k) {
                    iRow[k] = static_cast<Ipopt::Index>(pattern[k].first);
                    jCol[k] = static_cast<Ipopt::Index>(pattern[k].second);
                }
            } else {
                // VALUES pass: evaluate σ·∇²f + Σλᵢ·∇²gᵢ.
                // eval_lagrangian_hessian packs weights as [obj_factor, λ₀, ..., λ_{m-1}]
                // and returns values aligned to lagrangian_hessian_sparsity().
                const std::vector<double> lam = to_vector(lambda, m);
                const auto vals = problem_.eval_lagrangian_hessian(
                    to_vector(x, n), obj_factor, lam);
                std::copy(vals.begin(), vals.end(), values);
            }
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    // ------------------------------------------------------------------
    // TNLP: solution callback
    // ------------------------------------------------------------------

    void finalize_solution(Ipopt::SolverReturn status,
                           Ipopt::Index n, const Ipopt::Number* x,
                           const Ipopt::Number* /*z_L*/,
                           const Ipopt::Number* /*z_U*/,
                           Ipopt::Index m, const Ipopt::Number* /*g*/,
                           const Ipopt::Number* lambda,
                           Ipopt::Number obj_value,
                           const Ipopt::IpoptData* /*ip_data*/,
                           Ipopt::IpoptCalculatedQuantities* /*ip_cq*/) override {
        result_.status           = map_solver_return(status);
        result_.x                = to_vector(x, n);
        result_.objective_value  = obj_value;
        result_.constraint_multipliers = to_vector(lambda, m);

        // Human-readable status message using the IPOPT return-code name.
        // A small inline map suffices; the full list is rarely needed here.
        switch (status) {
            case Ipopt::SUCCESS:
                result_.message = "Optimal solution found";
                break;
            case Ipopt::STOP_AT_ACCEPTABLE_POINT:
                result_.message = "Solved to acceptable level";
                break;
            case Ipopt::MAXITER_EXCEEDED:
                result_.message = "Maximum iterations exceeded";
                break;
            case Ipopt::LOCAL_INFEASIBILITY:
                result_.message = "Problem is locally infeasible";
                break;
            case Ipopt::INVALID_NUMBER_DETECTED:
                result_.message = "Invalid number (NaN/Inf) detected";
                break;
            default:
                result_.message = "Solver returned an unrecognised status code";
                break;
        }
    }

 private:
    const nlp::NLPProblem&       problem_;
    const std::vector<double>&   initial_guess_;
    SolverResult&                result_;
};

}  // anonymous namespace

// ---------------------------------------------------------------------------
// IpoptSolver::solve
// ---------------------------------------------------------------------------

SolverResult IpoptSolver::solve(const nlp::NLPProblem& problem,
                                const std::vector<double>& initial_guess,
                                const std::vector<double>& parameters) {
    // Validate initial_guess size before doing anything else — a mismatch would
    // silently overflow the IPOPT-allocated Number[n] buffer in get_starting_point.
    if (initial_guess.size() != problem.num_variables()) {
        throw SolverError(
            "IpoptSolver::solve: initial_guess size (" +
            std::to_string(initial_guess.size()) +
            ") does not match problem.num_variables() (" +
            std::to_string(problem.num_variables()) + ")");
    }

    // Inject solve-time parameters once; constant across all evaluation callbacks.
    // set_parameters is const (backend reached via unique_ptr), so injecting through
    // our const& problem is well-formed. Propagates ADError on a size mismatch — a
    // setup error, surfaced (not swallowed) rather than reported as a solve outcome.
    // Only inject when the caller explicitly provides parameters; an empty vector
    // means "no solve-time parameters supplied" — the backend retains whatever
    // was last injected (or its compiled default if never set).
    if (!parameters.empty())
        problem.set_parameters(parameters);

    SolverResult result;
    result.status = SolverStatus::Failure;

    // The adapter holds references to problem, initial_guess, and result.
    // All three outlive this function call.  SmartPtr owns the adapter; never
    // delete it manually.
    Ipopt::SmartPtr<Ipopt::TNLP> adapter =
        new IpoptTNLPAdapter(problem, initial_guess, result);

    Ipopt::SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();

    // Configure IPOPT options.
    app->Options()->SetNumericValue("tol", tolerance_);
    app->Options()->SetIntegerValue("print_level", print_level_);
    app->Options()->SetIntegerValue("max_iter", max_iterations_);
    // Use exact second-order information via the TNLP eval_h callback.
    app->Options()->SetStringValue("hessian_approximation", "exact");
    // Adaptive barrier parameter update strategy (generally more robust).
    app->Options()->SetStringValue("mu_strategy", "adaptive");

    // Initialize the IPOPT application.  Failures here indicate a
    // configuration error (bad option, missing library), not a solve failure.
    const Ipopt::ApplicationReturnStatus init_status = app->Initialize();
    if (init_status != Ipopt::Solve_Succeeded) {
        throw SolverError(
            "IpoptApplication::Initialize() failed with code " +
            std::to_string(static_cast<int>(init_status)));
    }

    // Run the optimisation.  finalize_solution populates result_ if the solve
    // reaches a terminal state (converged, infeasible, iteration limit, …).
    // Hard configuration failures (Invalid_Option, etc.) that prevent IPOPT
    // from even starting are caught here.
    const Ipopt::ApplicationReturnStatus opt_status = app->OptimizeTNLP(adapter);

    // If OptimizeTNLP reported a hard configuration/setup failure *and*
    // finalize_solution was never called (result still has Failure status with
    // no solution vector), set an explicit message so the caller has context.
    if (opt_status == Ipopt::Invalid_Option ||
        opt_status == Ipopt::Invalid_Problem_Definition ||
        opt_status == Ipopt::Unrecoverable_Exception) {
        if (result.x.empty()) {
            result.status  = SolverStatus::Failure;
            result.message = "IPOPT returned hard error code " +
                             std::to_string(static_cast<int>(opt_status));
        }
    }

    // Fallback: if we ended up in a Failure state without any message
    // (e.g. an unmapped terminal code where finalize_solution was called but
    // didn't set a human-readable text), populate a numeric description so
    // callers always have context on failure.
    if (result.status == SolverStatus::Failure && result.message.empty()) {
        result.message = "IPOPT terminated with application return status " +
                         std::to_string(static_cast<int>(opt_status));
    }

    return result;
}

}  // namespace goss::solver
