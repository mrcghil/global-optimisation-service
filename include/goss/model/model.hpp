// include/goss/model/model.hpp
#pragma once
#include <cstddef>
#include <string>
#include <vector>
#include "goss/model/errors.hpp"
#include "goss/model/handles.hpp"
#include "goss/transcription/ocp_problem.hpp"
#include "goss/transcription/transcription.hpp"  // kInf

namespace goss::model {

/// Runtime builder for a continuous optimal-control problem.
///
/// Usage (lambda DSL, v1):
///   Model model;
///   auto q    = model.add_state("queue_length");
///   auto rate = model.add_control("service_rate");
///   model.set_state_bounds(q, 0.0, kInf);       // q >= 0
///   model.set_control_bounds(rate, 0.0, MAX);   // 0 <= rate <= MAX
///   model.set_initial_state(q, 10.0);           // q(0) = 10
///   model.set_mesh(0.0, T, num_intervals);
///   auto ocp = model.build(dynamics_lambda, cost_lambda);
///   auto compiled = transcription::HermiteSimpson::compile(ocp, "queue");
///
/// EXTENDING TO AN EXPRESSION DSL (future, not v1):
///   The spec's operator-overloaded syntax (`q >= 0.0`, `q.initial() == 10.0`,
///   `set_cost(integral(q + w*rate*rate))`) can be layered ON TOP of this core
///   without changing it. Add operators to StateHandle/ControlHandle that build
///   small expression nodes; a path-constraint expression like `q >= 0.0`
///   lowers to set_state_bounds(q, 0.0, kInf); a boundary expression
///   `q.initial() == 10.0` lowers to set_initial_state(q, 10.0); an
///   `integral(expr)` cost lowers to a cost lambda that evaluates the expr AST
///   under the templated scalar T. The AST evaluator must be templated on T
///   (so it records under CppAD AD types), exactly like the lambdas here. The
///   Model's build() and all downstream layers stay unchanged — the AST is a
///   front-end that produces the same lambdas/bounds this API takes directly.
class Model {
 public:
    StateHandle add_state(const std::string& name) {
        ensure_unique_name(name);
        const std::size_t index = state_names_.size();
        state_names_.push_back(name);
        state_lower_.push_back(-transcription::kInf);
        state_upper_.push_back(transcription::kInf);
        initial_value_.push_back(0.0);
        initial_fixed_.push_back(false);
        final_value_.push_back(0.0);
        final_fixed_.push_back(false);
        return StateHandle{index};
    }

    ControlHandle add_control(const std::string& name) {
        ensure_unique_name(name);
        const std::size_t index = control_names_.size();
        control_names_.push_back(name);
        control_lower_.push_back(-transcription::kInf);
        control_upper_.push_back(transcription::kInf);
        return ControlHandle{index};
    }

    std::size_t num_states() const { return state_names_.size(); }
    std::size_t num_controls() const { return control_names_.size(); }

    const std::string& state_name(std::size_t index) const {
        if (index >= state_names_.size()) {
            throw ModelError("Model::state_name: index out of range");
        }
        return state_names_[index];
    }

    const std::string& control_name(std::size_t index) const {
        if (index >= control_names_.size()) {
            throw ModelError("Model::control_name: index out of range");
        }
        return control_names_[index];
    }

    // Bounds / boundary / mesh setters added in Task 3-4.
    // build() added in Task 4.

 private:
    /// Throws ModelError if name is already registered as a state or control.
    void ensure_unique_name(const std::string& name) const {
        for (const auto& existing : state_names_) {
            if (existing == name) {
                throw ModelError("Model: duplicate name '" + name + "' (already a state)");
            }
        }
        for (const auto& existing : control_names_) {
            if (existing == name) {
                throw ModelError("Model: duplicate name '" + name + "' (already a control)");
            }
        }
    }

    std::vector<std::string> state_names_;
    std::vector<std::string> control_names_;
    std::vector<double> state_lower_;
    std::vector<double> state_upper_;
    std::vector<double> control_lower_;
    std::vector<double> control_upper_;
    std::vector<double> initial_value_;
    std::vector<bool>   initial_fixed_;
    std::vector<double> final_value_;
    std::vector<bool>   final_fixed_;

    bool                 mesh_set_ = false;
    transcription::Mesh  mesh_{0.0, 1.0, 1};
};

}  // namespace goss::model
