// include/goss/transcription/ocp_problem.hpp
#pragma once
#include <cstddef>
#include <vector>
#include "goss/transcription/errors.hpp"

namespace goss::transcription {

struct Mesh {
    double t_initial;
    double t_final;
    std::size_t num_intervals;
    std::size_t num_nodes() const { return num_intervals + 1; }
    double interval_width() const { return (t_final - t_initial) / static_cast<double>(num_intervals); }
    void validate() const {
        if (num_intervals == 0) throw TranscriptionError("Mesh: num_intervals must be >= 1");
        if (t_final <= t_initial) throw TranscriptionError("Mesh: t_final must be > t_initial");
    }
};

/// No-op algebraic residuals functor for problems with num_algebraic == 0.
/// Used as the default AlgResFn template argument so all existing
/// OcpProblem<Dyn,Cost> users are unaffected — the third template parameter
/// defaults to this type and is never named by existing callers.
struct NoAlgebraicResiduals {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& /*x*/,
                               const std::vector<T>& /*u*/,
                               const std::vector<T>& /*alg_vars*/,
                               T /*t*/) const {
        return {};
    }
};

/// No-op path-constraint functor for problems with num_path_constraints == 0.
/// Used as the default PathConstraintFn template argument (4th param) so all
/// existing OcpProblem<Dyn,Cost> and OcpProblem<Dyn,Cost,AlgResFn> users are
/// unaffected — the fourth template parameter defaults to this type and is
/// never named by existing callers.
///
/// Signature: g(x, u, t) → empty vector<T>.  Note: 3 args, NO alg_vars —
/// path constraints are purely g(x,u,t), distinct from algebraic residuals.
///
/// AD-safety: fully templated operator() — no std::function, no virtual dispatch.
struct NoPathConstraints {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& /*x*/,
                               const std::vector<T>& /*u*/,
                               T /*t*/) const {
        return {};
    }
};

/// Optimal-control problem description passed from the model DSL layer to a
/// transcription scheme's compile() function.
///
/// Template parameters:
///   DynamicsFn       — generic functor (x, u, t) → vector<T>, the ODE right-hand side.
///   CostFn           — generic functor (x, u, t) → T, the running cost integrand.
///   AlgResFn         — generic functor (x, u, alg_vars, t) → vector<T>, the algebraic
///                      residual vector (size == num_algebraic). Defaults to
///                      NoAlgebraicResiduals for num_algebraic == 0; existing callers
///                      that do not specify a 3rd param are unaffected.
///   PathConstraintFn — generic functor (x, u, t) → vector<T>, the path-constraint
///                      evaluation (size == num_path_constraints). Defaults to
///                      NoPathConstraints (empty return, zero additional NLP rows).
///                      This is the 4th template param; it adds constraint rows only,
///                      never decision variables — distinct from AlgResFn (3rd param)
///                      which adds algebraic state variables to the NLP decision vector.
///
/// DAE ORTHOGONALITY NOTE: AlgResFn (3rd param) and PathConstraintFn (4th param) occupy
/// distinct semantic slots. Algebraic-residual fields (num_algebraic, algebraic_*) govern
/// DAE algebraic variables that enter the NLP decision vector. Path-constraint fields
/// (num_path_constraints, path_constraint_*) add only NLP constraint rows. Both sets of
/// fields coexist in OcpProblem; there are no name collisions.
template <typename DynamicsFn, typename CostFn,
          typename AlgResFn = NoAlgebraicResiduals,
          typename PathConstraintFn = NoPathConstraints>
struct OcpProblem {
    std::size_t num_states;
    std::size_t num_controls;
    DynamicsFn dynamics;    // template<T> vector<T>(const vector<T>& x, const vector<T>& u, T t)
    CostFn cost;            // template<T> T(const vector<T>& x, const vector<T>& u, T t)
    Mesh mesh;
    std::vector<double> state_lower;
    std::vector<double> state_upper;
    std::vector<double> control_lower;
    std::vector<double> control_upper;
    std::vector<double> initial_state;
    std::vector<double> initial_state_fixed;   // nonzero => pin node 0 state i
    std::vector<double> final_state;
    std::vector<double> final_state_fixed;     // nonzero => pin last node state i

    // ---- Algebraic-variable (DAE Flavor 2) fields ----
    // When num_algebraic == 0 (the default), all three schemes behave exactly as before:
    // the algebraic_residuals_functor is never called, and the bound vectors are empty.
    std::size_t num_algebraic = 0;
    // Residual functor: g(x, u, alg_vars, t) → vector<T> of size num_algebraic.
    // The solver enforces g[j] == 0 at every collocation node via equality constraints.
    AlgResFn algebraic_residuals_functor = AlgResFn{};
    // Box bounds on each algebraic variable's value (not on the residual).
    // Size must equal num_algebraic.
    std::vector<double> algebraic_lower_bounds;
    std::vector<double> algebraic_upper_bounds;

    // ---- Path-constraint (nonlinear inequality/equality) fields ----
    // When num_path_constraints == 0 (the default), no additional NLP constraint rows
    // are added and path_constraints is never called. These fields are independent of
    // the algebraic-variable fields above: path constraints add rows only, not decision
    // variables.
    //
    // CANONICAL FIELD ORDER (for Task 4 aggregate-init via Model::build_with_path_constraints):
    //   1. num_path_constraints
    //   2. path_constraint_lower
    //   3. path_constraint_upper
    //   4. path_constraints      (functor — placed last to mirror the algebraic block)

    /// Number of scalar path constraints returned by path_constraints(x,u,t).
    /// Default 0 — no path-constraint NLP rows added.
    std::size_t num_path_constraints = 0;

    /// Per-constraint lower bounds on g(x,u,t). Size must equal num_path_constraints.
    /// Use -kInf for a one-sided upper bound; use the same value as upper for equality.
    std::vector<double> path_constraint_lower;

    /// Per-constraint upper bounds on g(x,u,t). Size must equal num_path_constraints.
    /// Use +kInf for a one-sided lower bound; use the same value as lower for equality.
    std::vector<double> path_constraint_upper;

    /// Path-constraint functor. Evaluated at every collocation node k:
    ///   g_vec = path_constraints(x_k, u_k, t_k)
    /// Each element g_vec[j] becomes one NLP constraint row bounded by
    /// [path_constraint_lower[j], path_constraint_upper[j]].
    ///
    /// AD-safety: PathConstraintFn must be fully templated so its operator()
    /// instantiates correctly under both double and CppAD::AD<CppAD::cg::CG<double>>.
    PathConstraintFn path_constraints = PathConstraintFn{};
};

}  // namespace goss::transcription
