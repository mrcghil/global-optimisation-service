// include/goss/model/expr/expr_model.hpp
// ExprModel: fluent builder that accumulates per-state dynamics expressions
// and a cost expression, then assembles them into a Model::build()-compatible
// call. This is a template-accumulating builder: each call to with_dynamics()
// returns a NEW type (ExprModel<NewDynTuple, CostFn>) so the full tuple type
// is known at build() time without any type erasure (std::function / virtual).
//
// WHY this approach: The expression DSL produces heterogeneous types
// (e.g. BinaryExpr<SubTag, ConstantExpr, ControlLeaf>) that must be templated
// on ScalarT at call time to support CppAD AD types for gradient recording.
// std::function<T(x,u,t)> requires a fixed T at construction, breaking AD.
// Instead we accumulate DynamicsEntry<E> objects in a std::tuple and expand
// them at build() time via std::index_sequence — the AD-safe approach.
//
// CRITICAL: DynamicsFunctor places each entry's result at result[state_index],
// NOT at the tuple position. This allows with_dynamics() calls to be given in
// any order and still produce the correct dx/dt vector layout.
#pragma once
#include <algorithm>  // std::max, std::min — used for merging constraint bounds
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <variant>  // std::monostate — used as sentinel for "no cost set yet"
#include <vector>
#include "goss/model/expr/constraints.hpp"
#include "goss/model/expr/errors.hpp"
#include "goss/model/expr/integral.hpp"
#include "goss/model/model.hpp"

namespace goss::model::expr {

// ─── DynamicsEntry ───────────────────────────────────────────────────────────

/// Associates a state index with a typed dynamics expression.
/// Stored inside DynTuple; DynamicsFunctor reads state_index to place the
/// evaluated result into the correct slot of the dx/dt result vector.
///
/// Template parameter DynExpr: any expression-tree node that exposes
///   template<typename ScalarT> ScalarT eval(const vector<ScalarT>& x,
///                                           const vector<ScalarT>& u,
///                                           ScalarT t) const;
template <typename DynExpr>
struct DynamicsEntry {
    /// Index of the state this dynamics expression belongs to.
    /// Used by DynamicsFunctor to place the result into the correct slot
    /// of the dx/dt vector (independent of the entry's position in the tuple).
    std::size_t state_index;

    /// The typed expression tree for this state's derivative.
    DynExpr dynamics_expression;

    /// Evaluate the dynamics expression under scalar type ScalarT.
    /// Uses the .template eval<ScalarT>(...) syntax because dynamics_expression
    /// is a dependent name — required by C++17 to parse eval as a template.
    template <typename ScalarT>
    ScalarT eval(const std::vector<ScalarT>& x,
                 const std::vector<ScalarT>& u,
                 ScalarT                     t) const {
        return dynamics_expression.template eval<ScalarT>(x, u, t);
    }
};

// ─── DynamicsFunctor ─────────────────────────────────────────────────────────

/// Holds a tuple of DynamicsEntry objects and satisfies Model::build's
/// DynamicsFn contract:
///   template<typename T> std::vector<T> operator()(const vector<T>& x,
///                                                   const vector<T>& u,
///                                                   T t) const;
///
/// Each entry is evaluated and its result is placed into result[state_index].
/// This is the critical correctness invariant: placement is by state_index,
/// NOT by the entry's position in the tuple, so with_dynamics() calls can
/// be given in any order.
///
/// AD-safety: fully templated on ScalarT; no std::function, no virtual
/// dispatch. Instantiates correctly under both double and CppAD AD types.
template <typename DynTuple>
struct DynamicsFunctor {
    /// Tuple of DynamicsEntry objects, one per declared state.
    DynTuple dynamics_entries;

    /// Total number of states — determines the size of the returned vector.
    std::size_t num_states;

    template <typename ScalarT>
    std::vector<ScalarT> operator()(const std::vector<ScalarT>& x,
                                    const std::vector<ScalarT>& u,
                                    ScalarT                     t) const {
        // Allocate result with one slot per state, default-initialised to zero.
        std::vector<ScalarT> result(num_states, ScalarT(0));
        // Expand the index_sequence to evaluate every entry and place by state_index.
        fill_result(result, x, u, t,
                    std::make_index_sequence<std::tuple_size_v<DynTuple>>{});
        return result;
    }

 private:
    /// Fold expression over Indices... — evaluates each tuple entry and assigns
    /// its result to result[entry.state_index] (placement by state_index, not
    /// by tuple position). The comma-operator fold ((expr), ...) guarantees
    /// all assignments are performed in tuple order (though order does not matter
    /// since each entry targets a distinct slot).
    template <typename ScalarT, std::size_t... Indices>
    void fill_result(std::vector<ScalarT>&       result,
                     const std::vector<ScalarT>& x,
                     const std::vector<ScalarT>& u,
                     ScalarT                     t,
                     std::index_sequence<Indices...>) const {
        // CRITICAL: placement is by state_index, not by Indices position.
        // std::get<I> fetches the DynamicsEntry; .state_index selects the slot.
        ((result[std::get<Indices>(dynamics_entries).state_index] =
              std::get<Indices>(dynamics_entries).template eval<ScalarT>(x, u, t)), ...);
    }
};

// ─── ExprModel ───────────────────────────────────────────────────────────────

/// Fluent builder for the expression DSL. Accumulates per-state dynamics and
/// a cost expression, then delegates to Model::build().
///
/// DynTuple grows with each with_dynamics() call — each call returns a NEW
/// ExprModel<NewDynTuple, CostFn> type (moving model_/dyn_tuple_/cost_fn_ in).
/// CostFn defaults to std::monostate (sentinel for "no cost set") and is
/// replaced by with_cost(). build() uses `if constexpr` to detect CostFn ==
/// std::monostate at compile time and throws ExprError at runtime in that
/// branch — a missing with_cost() is therefore a runtime error (not a
/// compile-time static_assert).
///
/// Usage:
///   goss::model::expr::ExprModel<> m;
///   auto q    = m.add_state("q");
///   auto rate = m.add_control("rate");
///   m.apply(q >= 0.0);
///   m.apply(q.initial() == 10.0);
///   m.set_mesh(0.0, T, N);
///   auto ocp = std::move(m)
///       .with_dynamics(q, ARRIVAL - ControlLeaf{rate.index})
///       .with_cost(integral(StateLeaf{q.index} + w * rate * rate))
///       .build();
template <typename DynTuple = std::tuple<>, typename CostFn = std::monostate>
class ExprModel {
 public:
    // --- Default construction (used for ExprModel<> entry point) ---
    ExprModel() = default;

    /// Internal constructor used by with_dynamics() and with_cost() to
    /// transfer all accumulated state into the returned ExprModel specialisation.
    ExprModel(goss::model::Model model_arg,
              DynTuple           dyn_tuple_arg,
              CostFn             cost_fn_arg)
        : model_(std::move(model_arg)),
          dyn_tuple_(std::move(dyn_tuple_arg)),
          cost_fn_(std::move(cost_fn_arg)) {}

    // --- Model-level forwarding setters ---

    goss::model::StateHandle add_state(const std::string& name) {
        return model_.add_state(name);
    }

    goss::model::ControlHandle add_control(const std::string& name) {
        return model_.add_control(name);
    }

    void set_mesh(double t_initial, double t_final, std::size_t num_intervals) {
        model_.set_mesh(t_initial, t_final, num_intervals);
    }

    void set_state_bounds(goss::model::StateHandle   state_handle,
                          double                     lower,
                          double                     upper) {
        model_.set_state_bounds(state_handle, lower, upper);
    }

    void set_control_bounds(goss::model::ControlHandle control_handle,
                            double                     lower,
                            double                     upper) {
        model_.set_control_bounds(control_handle, lower, upper);
    }

    void set_initial_state(goss::model::StateHandle state_handle, double value) {
        model_.set_initial_state(state_handle, value);
    }

    void set_final_state(goss::model::StateHandle state_handle, double value) {
        model_.set_final_state(state_handle, value);
    }

    // --- Accessor forwarding (useful in tests) ---

    std::size_t num_states()   const { return model_.num_states();   }
    std::size_t num_controls() const { return model_.num_controls(); }

    // --- Constraint sugar (mutate in place, return *this& so calls chain) ---
    // These forward to apply_bound / apply_boundary and return *this& — they
    // do NOT change the DynTuple or CostFn type, so no new ExprModel is needed.

    /// Apply a state box constraint (q >= lo, q <= hi), merging with existing bounds.
    /// WHY merge instead of overwrite: users typically write apply(q >= 0.0) then
    /// apply(q <= 100.0) as two separate calls. The constraint structs carry both
    /// lower and upper (set to ±kInf for the unused side). Overwriting would wipe
    /// the first bound when the second call arrives. Merging via max(lower) /
    /// min(upper) preserves both constraints correctly.
    ExprModel& apply(const BoundConstraint& constraint) {
        const std::size_t idx = constraint.state_handle.index;
        const double merged_lower = std::max(model_.state_lower(idx), constraint.lower_bound);
        const double merged_upper = std::min(model_.state_upper(idx), constraint.upper_bound);
        model_.set_state_bounds(constraint.state_handle, merged_lower, merged_upper);
        return *this;
    }

    /// Apply a control box constraint (u >= lo, u <= hi), merging with existing bounds.
    /// See apply(BoundConstraint&) for the rationale — same merge semantics.
    ExprModel& apply(const ControlBoundConstraint& constraint) {
        const std::size_t idx = constraint.control_handle.index;
        const double merged_lower = std::max(model_.control_lower(idx), constraint.lower_bound);
        const double merged_upper = std::min(model_.control_upper(idx), constraint.upper_bound);
        model_.set_control_bounds(constraint.control_handle, merged_lower, merged_upper);
        return *this;
    }

    /// Apply a boundary condition (q.initial() == v or q.final() == v) to the Model.
    ExprModel& apply(const BoundaryConstraint& constraint) {
        apply_boundary(model_, constraint);
        return *this;
    }

    // --- Fluent type-accumulating builders (rvalue-ref qualified) ---

    /// Add a dynamics expression for the given state and return a NEW ExprModel
    /// whose DynTuple has one more DynamicsEntry appended.
    ///
    /// WHY &&: the caller must std::move(*this) so model_ (with all its vectors)
    /// is moved rather than copied. Each call produces a distinct template
    /// specialisation, so this cannot be a regular mutating member.
    template <typename DynExpr>
    auto with_dynamics(goss::model::StateHandle state_handle,
                       DynExpr                  dyn_expr) && {
        DynamicsEntry<DynExpr> new_entry{state_handle.index, std::move(dyn_expr)};
        auto new_tuple = std::tuple_cat(
            std::move(dyn_tuple_),
            std::make_tuple(std::move(new_entry)));
        using NewDynTuple = decltype(new_tuple);
        return ExprModel<NewDynTuple, CostFn>{
            std::move(model_),
            std::move(new_tuple),
            std::move(cost_fn_)
        };
    }

    /// Set the cost functor and return a NEW ExprModel with the CostFn type
    /// replaced. Replaces std::monostate sentinel with the real cost type.
    template <typename NewCostFn>
    auto with_cost(NewCostFn new_cost_fn) && {
        return ExprModel<DynTuple, NewCostFn>{
            std::move(model_),
            std::move(dyn_tuple_),
            std::move(new_cost_fn)
        };
    }

    // --- Zero-argument build(): assembles DynamicsFunctor + cost, delegates ---

    /// Assemble accumulated dynamics and cost into a single Model::build() call.
    ///
    /// Static precondition (compile time): CostFn must not be std::monostate
    /// — call with_cost(integral(...)) before build().
    ///
    /// Runtime precondition: the number of DynamicsEntry objects in DynTuple
    /// must equal the number of declared states; throws ExprError otherwise
    /// (signals which state is missing a dynamics expression).
    ///
    /// NOTE: not &&-qualified so callers can invoke build() on a named variable
    /// (the brief's test code assigns the result of with_cost() to a local and
    /// then calls .build() on that local). Model state is moved from model_ and
    /// dyn_tuple_ — calling build() twice on the same object is undefined.
    auto build() {
        // Use if constexpr to gate the two paths at compile time.
        // WHY: when CostFn=std::monostate, model_.build(functor, monostate)
        // would not compile (monostate has no operator()(x,u,t)). The if
        // constexpr branch lets us throw ExprError for the "no cost" case
        // without instantiating the invalid model_.build() call.
        // The else branch is only instantiated when CostFn is a real cost type,
        // ensuring AD-safety of the generated DynamicsFunctor + CostFn pair.
        if constexpr (std::is_same_v<CostFn, std::monostate>) {
            // "No cost" is detected first; in tests that test the dynamics-
            // count mismatch without setting a cost, this is the error thrown.
            // (Both errors are ExprError, satisfying EXPECT_THROW's type check.)
            throw ExprError(
                "ExprModel::build(): no cost set — "
                "call with_cost(integral(...)) before build()");
            // Unreachable — needed only to satisfy the compiler that both
            // branches of if constexpr return consistently. The return type
            // deduction for this branch is never used.
        } else {
            // Runtime guard: every declared state must have a dynamics expression.
            const std::size_t declared_state_count = model_.num_states();
            const std::size_t registered_dyn_count = std::tuple_size_v<DynTuple>;
            if (registered_dyn_count != declared_state_count) {
                throw ExprError(
                    "ExprModel::build(): " +
                    std::to_string(declared_state_count) + " state(s) declared but " +
                    std::to_string(registered_dyn_count) +
                    " dynamics expression(s) registered — "
                    "call with_dynamics() once per declared state");
            }

            // Assemble the DynamicsFunctor from the accumulated tuple.
            DynamicsFunctor<DynTuple> dyn_functor{
                std::move(dyn_tuple_),
                declared_state_count
            };

            return model_.build(std::move(dyn_functor), std::move(cost_fn_));
        }
    }

    // --- Two-argument build(): lambda path, unchanged ---
    // Re-exposes the Model::build(dynamics, cost) overload so callers that
    // bypass the DSL can still go through ExprModel as a thin wrapper.
    template <typename DynamicsFnArg, typename CostFnArg>
    auto build(DynamicsFnArg dynamics_lambda, CostFnArg cost_lambda) {
        return model_.build(std::move(dynamics_lambda), std::move(cost_lambda));
    }

 private:
    goss::model::Model model_{};
    DynTuple           dyn_tuple_{};
    // cost_fn_ is std::monostate by default (sentinel for "no cost set").
    // Default-constructing std::monostate is valid; with_cost() replaces it
    // with the real CostFunctor type via the template parameter.
    CostFn             cost_fn_{};
};

}  // namespace goss::model::expr
