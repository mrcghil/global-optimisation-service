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

 private:
    std::unique_ptr<ad::ADBackend> backend_;
    std::size_t num_variables_;
    std::size_t num_constraints_;
    std::vector<double> variable_lower_bounds_;
    std::vector<double> variable_upper_bounds_;
    std::vector<double> constraint_lower_bounds_;
    std::vector<double> constraint_upper_bounds_;

    // (col, value_index) pairs for entries of the backend Jacobian whose row == 0.
    // Used to scatter eval_jacobian()[value_index] into gradient[col].
    std::vector<std::pair<std::size_t, std::size_t>> objective_gradient_slots_;
};

}  // namespace goss::nlp
