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

/// Optimal-control problem description passed from the model DSL layer to a
/// transcription scheme's compile() function.
///
/// Template parameters:
///   DynamicsFn  — generic functor (x, u, t) → vector<T>, the ODE right-hand side.
///   CostFn      — generic functor (x, u, t) → T, the running cost integrand.
///   AlgResFn    — generic functor (x, u, alg_vars, t) → vector<T>, the algebraic
///                 residual vector (size == num_algebraic). Defaults to NoAlgebraicResiduals
///                 for num_algebraic == 0; existing callers never name this parameter.
template <typename DynamicsFn, typename CostFn,
          typename AlgResFn = NoAlgebraicResiduals>
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
};

}  // namespace goss::transcription
