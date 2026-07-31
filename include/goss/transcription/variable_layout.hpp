#pragma once
#include <cstddef>
#include <string>
#include "goss/transcription/errors.hpp"

namespace goss::transcription {

/// Maps (node, state|control) coordinates to flat decision-vector indices.
/// Layout is grouped by node, states before controls within each node:
///   z = [ x_0, u_0, x_1, u_1, ..., x_{Nn-1}, u_{Nn-1} ]
/// This is the single source of truth for the z packing; every scheme and
/// test must use it so indices never diverge.
class VariableLayout {
 public:
    VariableLayout(std::size_t num_states, std::size_t num_controls, std::size_t num_nodes)
        : num_states_(num_states), num_controls_(num_controls), num_nodes_(num_nodes) {
        if (num_states_ == 0) {
            throw TranscriptionError("VariableLayout: num_states must be >= 1");
        }
        if (num_nodes_ < 2) {
            throw TranscriptionError("VariableLayout: num_nodes must be >= 2 (need >= 1 interval)");
        }
    }

    std::size_t num_states() const { return num_states_; }
    std::size_t num_controls() const { return num_controls_; }
    std::size_t num_nodes() const { return num_nodes_; }
    std::size_t variables_per_node() const { return num_states_ + num_controls_; }
    std::size_t total_variables() const { return num_nodes_ * variables_per_node(); }

    std::size_t state_index(std::size_t node, std::size_t state) const {
        if (node >= num_nodes_) throw TranscriptionError("VariableLayout::state_index: node out of range");
        if (state >= num_states_) throw TranscriptionError("VariableLayout::state_index: state out of range");
        return node * variables_per_node() + state;
    }

    std::size_t control_index(std::size_t node, std::size_t control) const {
        if (node >= num_nodes_) throw TranscriptionError("VariableLayout::control_index: node out of range");
        if (control >= num_controls_) throw TranscriptionError("VariableLayout::control_index: control out of range");
        return node * variables_per_node() + num_states_ + control;
    }

 private:
    std::size_t num_states_;
    std::size_t num_controls_;
    std::size_t num_nodes_;
};

}  // namespace goss::transcription
