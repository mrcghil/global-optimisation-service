// include/goss/model/expr/integral.hpp
// Provides CostFunctor<CostExpr> and the integral() factory function.
//
// Purpose: wrap any expression-tree node (built with nodes.hpp + operators.hpp)
// into a callable functor whose template operator() satisfies the CostFn
// contract expected by Model::build() and ultimately by HermiteSimpson.
//
// The "integral" name reflects the spec DSL:
//   set_cost(integral(q + w * rate * rate))
// — the functor is evaluated *pointwise* at each collocation point; the
// numerical integration over the horizon is performed by HermiteSimpson via
// Simpson quadrature. integral() signals intent: "minimize the time-integral
// of this expression over the planning horizon".
#pragma once
#include <utility>   // std::move — used in the factory to avoid an extra copy
#include <vector>    // std::vector — the container type for x, u arguments
#include "goss/model/expr/nodes.hpp"  // expression node types (transitive dependency)

namespace goss::model::expr {

/// Wraps an expression-tree node into a callable functor that satisfies the
/// Model::build CostFn contract:
///   template<typename T> T operator()(const vector<T>& x, const vector<T>& u, T t) const
///
/// AD-safety: because operator() is a member function template (not std::function
/// or any type-erased wrapper), instantiating CostFunctor<E>::operator()<AD<CG<double>>>
/// calls cost_expression.eval<AD<CG<double>>>(x_ad, u_ad, t_ad), which correctly
/// records the computation on the CppAD tape for gradient/Hessian computation.
template <typename CostExpr>
struct CostFunctor {
    /// The wrapped expression tree node, stored by value (the tree is entirely
    /// stack-allocated — no heap allocation, no virtual dispatch).
    CostExpr cost_expression;

    /// Evaluate the expression under scalar type ScalarT.
    /// ScalarT may be double (direct evaluation) or any CppAD AD type
    /// (gradient recording). The forwarding to cost_expression.eval<ScalarT>
    /// keeps the full type information alive through the instantiation chain.
    template <typename ScalarT>
    ScalarT operator()(const std::vector<ScalarT>& x,
                       const std::vector<ScalarT>& u,
                       ScalarT                     t) const {
        return cost_expression.template eval<ScalarT>(x, u, t);
    }
};

/// Factory function: wrap any expression node in a CostFunctor.
///
/// Usage (spec DSL sugar):
///   auto ocp = model.build(dynamics, integral(q + weight * rate * rate));
///
/// The cost_expression argument is moved into the CostFunctor to avoid a
/// redundant copy of the (potentially deep) expression tree.
template <typename CostExpr>
CostFunctor<CostExpr> integral(CostExpr cost_expression) {
    return CostFunctor<CostExpr>{std::move(cost_expression)};
}

}  // namespace goss::model::expr
