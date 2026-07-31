// include/goss/nlp/nlp_problem.hpp
#pragma once
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>
#include "goss/ad/ad_backend.hpp"
#include "goss/ad/types.hpp"
#include "goss/nlp/errors.hpp"

namespace goss::nlp {

/// Wraps ONE ADBackend whose recorded function packs the objective in output 0
/// and the m constraints in outputs 1..m.  Presents solver-facing quantities:
/// objective, constraints, dense objective gradient, sparse constraint
/// Jacobian, sparse Lagrangian Hessian.  Depends only on the ADBackend
/// interface — never on any concrete AD backend.
class NLPProblem {
 public:
    NLPProblem(std::unique_ptr<ad::ADBackend> backend,
               std::vector<double> variable_lower_bounds,
               std::vector<double> variable_upper_bounds,
               std::vector<double> constraint_lower_bounds,
               std::vector<double> constraint_upper_bounds);

    std::size_t num_variables() const { return num_variables_; }
    std::size_t num_constraints() const { return num_constraints_; }

    const std::vector<double>& variable_lower_bounds() const { return variable_lower_bounds_; }
    const std::vector<double>& variable_upper_bounds() const { return variable_upper_bounds_; }
    const std::vector<double>& constraint_lower_bounds() const { return constraint_lower_bounds_; }
    const std::vector<double>& constraint_upper_bounds() const { return constraint_upper_bounds_; }

    double eval_objective(const std::vector<double>& x) const;
    std::vector<double> eval_constraints(const std::vector<double>& x) const;
    std::vector<double> eval_objective_gradient(const std::vector<double>& x) const;

    /// Returns the re-indexed constraint Jacobian sparsity pattern.
    /// Each entry (constraint_index, col) corresponds to a backend Jacobian row >= 1,
    /// re-indexed as constraint_index = backend_row - 1.  Aligned to eval_constraint_jacobian.
    const ad::SparsityPattern& constraint_jacobian_sparsity() const { return constraint_jacobian_sparsity_; }

    std::vector<double> eval_constraint_jacobian(const std::vector<double>& x) const;

    /// Returns the Lagrangian Hessian sparsity pattern (lower-triangle, w.r.t. x).
    /// Delegates directly to the backend's hessian_sparsity() — no re-indexing needed
    /// because the Hessian is over the variable space only.
    const ad::SparsityPattern& lagrangian_hessian_sparsity() const { return backend_->hessian_sparsity(); }

    /// Evaluates the Lagrangian Hessian: objective_factor·∇²f + Σ λ_i·∇²g_i.
    /// Packs weights as [objective_factor, λ_0, ..., λ_{m-1}] and delegates to
    /// the backend's weighted-sum Hessian, which returns the exact Lagrangian Hessian
    /// aligned to lagrangian_hessian_sparsity() (lower-triangle, col <= row).
    std::vector<double> eval_lagrangian_hessian(const std::vector<double>& x,
                                                double objective_factor,
                                                const std::vector<double>& constraint_multipliers) const;

 private:
    std::unique_ptr<ad::ADBackend> backend_;
    std::size_t num_variables_;
    std::size_t num_constraints_;
    std::vector<double> variable_lower_bounds_;
    std::vector<double> variable_upper_bounds_;
    std::vector<double> constraint_lower_bounds_;
    std::vector<double> constraint_upper_bounds_;

    // Precomputed index maps added in Tasks 4-5.

    // (col, value_index) pairs for entries of the backend Jacobian whose row == 0.
    // Used to scatter eval_jacobian()[value_index] into gradient[col].
    std::vector<std::pair<std::size_t, std::size_t>> objective_gradient_slots_;

    // Re-indexed constraint Jacobian sparsity: (constraint_index, col) where constraint_index = backend_row - 1.
    ad::SparsityPattern constraint_jacobian_sparsity_;

    // Maps each entry of constraint_jacobian_sparsity_ to its backend value_index k.
    std::vector<std::size_t> constraint_jacobian_slots_;
};

}  // namespace goss::nlp
