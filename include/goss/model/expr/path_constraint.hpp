// include/goss/model/expr/path_constraint.hpp
// PathConstraint DSL types and expression-typed comparison operators.
//
// This header provides:
//   1. is_expr_node<T> trait — true for all expr-DSL node types.
//   2. PathConstraintExpr<Expr> — one scalar path constraint with bounds.
//   3. PathConstraintEntry<Expr> — same as PathConstraintExpr (alias pattern).
//   4. PathConstraintFunctor<ExprTuple> — evaluates a tuple of entries under ScalarT.
//   5. operator>=/<=/== overloads for expression-typed LHS (SFINAE-gated).
//
// OVERLOAD RESOLUTION INVARIANT:
//   - q >= 0.0 where q is StateHandle  => goss::model::operator>= => BoundConstraint.
//   - (q_expr + 1.0) >= 0.0 where q_expr is BinaryExpr<...> => the overloads here
//     (namespace goss::model::expr, found by ADL on BinaryExpr) => PathConstraintExpr.
//   No ambiguity: StateHandle is not an expr node; expr nodes are not StateHandle.
//
// AD-SAFETY: PathConstraintFunctor::operator() is a member function template
// (not std::function). It instantiates under both double and CppAD AD types,
// so capturing a PathConstraintFunctor by value inside the HermiteSimpson
// packed generic lambda is correct and AD-safe.
#pragma once
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "goss/model/expr/nodes.hpp"
#include "goss/transcription/transcription.hpp"  // kInf

namespace goss::model::expr {

// ─── is_expr_node trait ──────────────────────────────────────────────────────

/// Trait to detect expr-DSL node types. Only these types are valid as the LHS
/// of expression-typed >= / <= / == (producing PathConstraintExpr).
/// StateHandle and ControlHandle are NOT expr nodes — they go through the
/// existing goss::model::operator>= which returns BoundConstraint.
template <typename T> struct is_expr_node : std::false_type {};

template <typename OpTag, typename LeftExpr, typename RightExpr>
struct is_expr_node<BinaryExpr<OpTag, LeftExpr, RightExpr>> : std::true_type {};

template <typename OperandExpr>
struct is_expr_node<UnaryNegExpr<OperandExpr>> : std::true_type {};

template <> struct is_expr_node<StateLeaf>    : std::true_type {};
template <> struct is_expr_node<ControlLeaf>  : std::true_type {};
template <> struct is_expr_node<ConstantExpr> : std::true_type {};
template <> struct is_expr_node<TimeLeaf>     : std::true_type {};

// Convenience alias.
template <typename T>
constexpr bool is_expr_node_v = is_expr_node<T>::value;

// ─── PathConstraintExpr ──────────────────────────────────────────────────────

/// Represents one scalar path constraint:  constraint_expression(x,u,t) in [lo, hi].
///
/// The convention is: g(x,u,t) = constraint_expression.eval(x,u,t).
/// For (expr >= rhs): constraint_expression = expr - rhs,  lo=0,   hi=+kInf.
/// For (expr <= rhs): constraint_expression = expr - rhs,  lo=-kInf, hi=0.
/// For (expr == rhs): constraint_expression = expr - rhs,  lo=0,   hi=0.
///
/// The subtraction of rhs is baked into the expression tree at DSL-call time:
///   (expr >= rhs)  =>  PathConstraintExpr{expr - ConstantExpr{rhs}, 0.0, +kInf}
/// This keeps the functor evaluation simple: just call constraint_expression.eval.
template <typename Expr>
struct PathConstraintExpr {
    Expr   constraint_expression;  // g(x,u,t) = this expression; must be >= 0 / in [lo,hi]
    double path_constraint_lower;
    double path_constraint_upper;
};

/// PathConstraintEntry is a synonym for PathConstraintExpr.
/// Used inside PathConstraintFunctor's tuple to mirror the DynamicsEntry pattern.
template <typename Expr>
using PathConstraintEntry = PathConstraintExpr<Expr>;

// ─── PathConstraintFunctor ───────────────────────────────────────────────────

/// Holds a tuple of PathConstraintEntry objects and satisfies the PathConstraintFn
/// contract expected by OcpProblem:
///   template<typename T> std::vector<T> operator()(const vector<T>& x,
///                                                   const vector<T>& u,
///                                                   T t) const;
///
/// Evaluates each entry in tuple order and packs results into the returned vector.
/// tuple order IS the constraint index order; unlike DynamicsFunctor, there is no
/// placement-by-index because path constraints are anonymous (no state-slot mapping).
///
/// AD-safety: fully templated operator() — instantiates under both double and
/// CppAD AD types. Captured by value inside HermiteSimpson's packed generic lambda.
template <typename ExprTuple>
struct PathConstraintFunctor {
    ExprTuple path_constraint_entries;

    /// Total number of path constraints — equal to std::tuple_size_v<ExprTuple>.
    /// Stored for runtime use (e.g. size checks before compile).
    std::size_t num_path_constraints;

    template <typename ScalarT>
    std::vector<ScalarT> operator()(const std::vector<ScalarT>& x,
                                    const std::vector<ScalarT>& u,
                                    ScalarT                     t) const {
        std::vector<ScalarT> result;
        result.reserve(num_path_constraints);
        fill_result(result, x, u, t,
                    std::make_index_sequence<std::tuple_size_v<ExprTuple>>{});
        return result;
    }

 private:
    template <typename ScalarT, std::size_t... Indices>
    void fill_result(std::vector<ScalarT>&       result,
                     const std::vector<ScalarT>& x,
                     const std::vector<ScalarT>& u,
                     ScalarT                     t,
                     std::index_sequence<Indices...>) const {
        // Fold expression: evaluate each entry in tuple order and push_back.
        // Comma-operator fold guarantees evaluation in Indices order.
        ((result.push_back(
            std::get<Indices>(path_constraint_entries)
                .constraint_expression.template eval<ScalarT>(x, u, t))), ...);
    }
};

// ─── Helpers to extract bounds from a PathConstraintFunctor tuple ─────────────

/// Implementation helper: pushes lower bounds from each tuple entry.
template <typename ExprTuple, std::size_t... Indices>
void extract_bounds_lower_impl(std::vector<double>& bounds,
                                const ExprTuple& entries,
                                std::index_sequence<Indices...>) {
    ((bounds.push_back(std::get<Indices>(entries).path_constraint_lower)), ...);
}

/// Extract the lower-bound vector from a PathConstraintFunctor (for OcpProblem setup).
template <typename ExprTuple>
std::vector<double> extract_path_constraint_lower(
        const PathConstraintFunctor<ExprTuple>& functor) {
    std::vector<double> bounds;
    bounds.reserve(functor.num_path_constraints);
    extract_bounds_lower_impl(bounds, functor.path_constraint_entries,
                              std::make_index_sequence<std::tuple_size_v<ExprTuple>>{});
    return bounds;
}

/// Implementation helper: pushes upper bounds from each tuple entry.
template <typename ExprTuple, std::size_t... Indices>
void extract_bounds_upper_impl(std::vector<double>& bounds,
                                const ExprTuple& entries,
                                std::index_sequence<Indices...>) {
    ((bounds.push_back(std::get<Indices>(entries).path_constraint_upper)), ...);
}

/// Extract the upper-bound vector from a PathConstraintFunctor (for OcpProblem setup).
template <typename ExprTuple>
std::vector<double> extract_path_constraint_upper(
        const PathConstraintFunctor<ExprTuple>& functor) {
    std::vector<double> bounds;
    bounds.reserve(functor.num_path_constraints);
    extract_bounds_upper_impl(bounds, functor.path_constraint_entries,
                              std::make_index_sequence<std::tuple_size_v<ExprTuple>>{});
    return bounds;
}

// ─── Expression-typed comparison operators (in goss::model::expr for ADL) ────
// These live in goss::model::expr so that ADL on BinaryExpr<...> etc. finds them.
// They do NOT overload with the existing goss::model::operator>=(StateHandle, double)
// because StateHandle is not an expr node (is_expr_node_v<StateHandle> == false).

/// (expr) >= rhs  =>  PathConstraintExpr{ expr - ConstantExpr{rhs}, 0.0, +kInf }
/// SFINAE guard: only fires when LhsExpr is an expr-DSL node (not StateHandle/ControlHandle).
template <typename LhsExpr,
          typename = std::enable_if_t<is_expr_node_v<LhsExpr>>>
PathConstraintExpr<BinaryExpr<SubTag, LhsExpr, ConstantExpr>>
operator>=(LhsExpr lhs_expression, double rhs_value) {
    using ExprType = BinaryExpr<SubTag, LhsExpr, ConstantExpr>;
    ExprType shifted_expression{
        std::move(lhs_expression),
        ConstantExpr{rhs_value}
    };
    return PathConstraintExpr<ExprType>{
        std::move(shifted_expression),
        0.0,
        goss::transcription::kInf
    };
}

/// (expr) <= rhs  =>  PathConstraintExpr{ expr - ConstantExpr{rhs}, -kInf, 0.0 }
template <typename LhsExpr,
          typename = std::enable_if_t<is_expr_node_v<LhsExpr>>>
PathConstraintExpr<BinaryExpr<SubTag, LhsExpr, ConstantExpr>>
operator<=(LhsExpr lhs_expression, double rhs_value) {
    using ExprType = BinaryExpr<SubTag, LhsExpr, ConstantExpr>;
    ExprType shifted_expression{
        std::move(lhs_expression),
        ConstantExpr{rhs_value}
    };
    return PathConstraintExpr<ExprType>{
        std::move(shifted_expression),
        -goss::transcription::kInf,
        0.0
    };
}

/// (expr) == rhs  =>  PathConstraintExpr{ expr - ConstantExpr{rhs}, 0.0, 0.0 }
template <typename LhsExpr,
          typename = std::enable_if_t<is_expr_node_v<LhsExpr>>>
PathConstraintExpr<BinaryExpr<SubTag, LhsExpr, ConstantExpr>>
operator==(LhsExpr lhs_expression, double rhs_value) {
    using ExprType = BinaryExpr<SubTag, LhsExpr, ConstantExpr>;
    ExprType shifted_expression{
        std::move(lhs_expression),
        ConstantExpr{rhs_value}
    };
    return PathConstraintExpr<ExprType>{
        std::move(shifted_expression),
        0.0,
        0.0
    };
}

}  // namespace goss::model::expr
