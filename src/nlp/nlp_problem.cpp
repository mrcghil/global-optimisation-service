#include "goss/nlp/nlp_problem.hpp"
#include <string>

namespace goss::nlp {

NLPProblem::NLPProblem(std::unique_ptr<ad::ADBackend> backend,
                       std::vector<double> variable_lower_bounds,
                       std::vector<double> variable_upper_bounds,
                       std::vector<double> constraint_lower_bounds,
                       std::vector<double> constraint_upper_bounds)
    : backend_(std::move(backend)),
      variable_lower_bounds_(std::move(variable_lower_bounds)),
      variable_upper_bounds_(std::move(variable_upper_bounds)),
      constraint_lower_bounds_(std::move(constraint_lower_bounds)),
      constraint_upper_bounds_(std::move(constraint_upper_bounds)) {
    if (!backend_) {
        throw NLPError("NLPProblem: backend must not be null");
    }
    if (backend_->output_size() < 1) {
        throw NLPError("NLPProblem: backend output_size must be >= 1 (output 0 is the objective)");
    }
    num_variables_ = backend_->input_size();
    num_constraints_ = backend_->output_size() - 1;

    if (variable_lower_bounds_.size() != num_variables_ ||
        variable_upper_bounds_.size() != num_variables_) {
        throw NLPError("NLPProblem: variable bound vectors must have size == num_variables (" +
                       std::to_string(num_variables_) + ")");
    }
    if (constraint_lower_bounds_.size() != num_constraints_ ||
        constraint_upper_bounds_.size() != num_constraints_) {
        throw NLPError("NLPProblem: constraint bound vectors must have size == num_constraints (" +
                       std::to_string(num_constraints_) + ")");
    }
    for (std::size_t i = 0; i < num_variables_; ++i) {
        if (variable_lower_bounds_[i] > variable_upper_bounds_[i]) {
            throw NLPError("NLPProblem: variable " + std::to_string(i) +
                           " has lower bound > upper bound");
        }
    }
    for (std::size_t i = 0; i < num_constraints_; ++i) {
        if (constraint_lower_bounds_[i] > constraint_upper_bounds_[i]) {
            throw NLPError("NLPProblem: constraint " + std::to_string(i) +
                           " has lower bound > upper bound");
        }
    }

    const ad::SparsityPattern& jacobian_pattern = backend_->jacobian_sparsity();
    for (std::size_t k = 0; k < jacobian_pattern.size(); ++k) {
        const std::size_t row = jacobian_pattern[k].first;
        const std::size_t col = jacobian_pattern[k].second;
        if (row == 0) {
            objective_gradient_slots_.emplace_back(col, k);
        } else {  // row >= 1: a constraint row
            constraint_jacobian_sparsity_.emplace_back(row - 1, col);
            constraint_jacobian_slots_.push_back(k);
        }
    }
}

std::size_t NLPProblem::num_parameters() const {
    return backend_->num_parameters();
}

void NLPProblem::set_parameters(const std::vector<double>& parameter_values) const {
    backend_->set_parameters(parameter_values);
}

double NLPProblem::eval_objective(const std::vector<double>& x) const {
    if (x.size() != num_variables_) {
        throw NLPError("eval_objective: x.size() != num_variables");
    }
    return backend_->eval(x)[0];
}

std::vector<double> NLPProblem::eval_constraints(const std::vector<double>& x) const {
    if (x.size() != num_variables_) {
        throw NLPError("eval_constraints: x.size() != num_variables");
    }
    const std::vector<double> all_outputs = backend_->eval(x);
    return std::vector<double>(all_outputs.begin() + 1, all_outputs.end());
}

std::vector<double> NLPProblem::eval_objective_gradient(const std::vector<double>& x) const {
    if (x.size() != num_variables_) {
        throw NLPError("eval_objective_gradient: x.size() != num_variables");
    }
    const std::vector<double> jacobian_values = backend_->eval_jacobian(x);
    std::vector<double> gradient(num_variables_, 0.0);
    for (const auto& [col, value_index] : objective_gradient_slots_) {
        gradient[col] = jacobian_values[value_index];
    }
    return gradient;
}

std::vector<double> NLPProblem::eval_constraint_jacobian(const std::vector<double>& x) const {
    if (x.size() != num_variables_) {
        throw NLPError("eval_constraint_jacobian: x.size() != num_variables");
    }
    const std::vector<double> jacobian_values = backend_->eval_jacobian(x);
    std::vector<double> constraint_values;
    constraint_values.reserve(constraint_jacobian_slots_.size());
    for (const std::size_t value_index : constraint_jacobian_slots_) {
        constraint_values.push_back(jacobian_values[value_index]);
    }
    return constraint_values;
}

std::vector<double> NLPProblem::eval_lagrangian_hessian(
    const std::vector<double>& x,
    double objective_factor,
    const std::vector<double>& constraint_multipliers) const {
    if (x.size() != num_variables_) {
        throw NLPError("eval_lagrangian_hessian: x.size() != num_variables");
    }
    if (constraint_multipliers.size() != num_constraints_) {
        throw NLPError("eval_lagrangian_hessian: constraint_multipliers.size() != num_constraints");
    }
    // Pack weights aligned to backend output layout: output 0 is the objective,
    // outputs 1..m are the constraints.  The backend's weighted-sum Hessian
    // Σ weights[i]·∇²(output_i) is therefore exactly objective_factor·∇²f + Σ λ_i·∇²g_i.
    std::vector<double> weights(num_constraints_ + 1);
    weights[0] = objective_factor;
    for (std::size_t i = 0; i < num_constraints_; ++i) {
        weights[1 + i] = constraint_multipliers[i];
    }
    return backend_->eval_hessian(x, weights);
}

}  // namespace goss::nlp
