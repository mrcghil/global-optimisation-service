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

}  // namespace goss::nlp
