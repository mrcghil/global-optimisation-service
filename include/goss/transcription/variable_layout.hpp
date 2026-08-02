#pragma once
#include <cstddef>
#include <string>
#include "goss/transcription/errors.hpp"

namespace goss::transcription {

/// Maps (node, state|control|algebraic) coordinates to flat decision-vector indices.
///
/// Layout is grouped by node; within each node, states precede controls which precede
/// algebraic variables:
///   z = [ x_0, u_0, alg_0,  x_1, u_1, alg_1,  ...,  x_{Nn-1}, u_{Nn-1}, alg_{Nn-1} ]
///
/// Per-node stride = num_states + num_controls + num_algebraic.
///
/// When num_algebraic == 0 (the default), the layout is identical to the pre-algebraic
/// layout (stride = num_states + num_controls), and state_index / control_index produce
/// exactly the same values. This is the single source of truth for z packing; every
/// scheme and test must use it so indices never diverge.
class VariableLayout {
 public:
    /// Primary constructor. num_algebraic defaults to 0 for backward compatibility.
    VariableLayout(std::size_t num_states, std::size_t num_controls,
                   std::size_t num_nodes, std::size_t num_algebraic = 0)
        : num_states_(num_states), num_controls_(num_controls),
          num_nodes_(num_nodes), num_algebraic_(num_algebraic) {
        if (num_states_ == 0) {
            throw TranscriptionError("VariableLayout: num_states must be >= 1");
        }
        if (num_nodes_ < 2) {
            throw TranscriptionError("VariableLayout: num_nodes must be >= 2 (need >= 1 interval)");
        }
    }

    std::size_t num_states()    const { return num_states_; }
    std::size_t num_controls()  const { return num_controls_; }
    std::size_t num_nodes()     const { return num_nodes_; }
    std::size_t num_algebraic() const { return num_algebraic_; }

    /// Number of decision variables per node: states + controls + algebraic variables.
    std::size_t variables_per_node() const {
        return num_states_ + num_controls_ + num_algebraic_;
    }

    /// Total decision variables across all nodes.
    std::size_t total_variables() const { return num_nodes_ * variables_per_node(); }

    /// Index of state component i at the given node.
    /// Formula: node * (ns + nc + na) + i
    std::size_t state_index(std::size_t node, std::size_t state) const {
        if (node >= num_nodes_)
            throw TranscriptionError("VariableLayout::state_index: node out of range");
        if (state >= num_states_)
            throw TranscriptionError("VariableLayout::state_index: state out of range");
        return node * variables_per_node() + state;
    }

    /// Index of control component j at the given node.
    /// Formula: node * (ns + nc + na) + ns + j
    std::size_t control_index(std::size_t node, std::size_t control) const {
        if (node >= num_nodes_)
            throw TranscriptionError("VariableLayout::control_index: node out of range");
        if (control >= num_controls_)
            throw TranscriptionError("VariableLayout::control_index: control out of range");
        return node * variables_per_node() + num_states_ + control;
    }

    /// Index of algebraic variable k at the given node.
    /// Formula: node * (ns + nc + na) + ns + nc + k
    /// Throws if num_algebraic_ == 0 (programming error: no algebraics registered).
    std::size_t algebraic_index(std::size_t node, std::size_t alg_var) const {
        if (num_algebraic_ == 0)
            throw TranscriptionError(
                "VariableLayout::algebraic_index: no algebraic variables registered "
                "(num_algebraic == 0)");
        if (node >= num_nodes_)
            throw TranscriptionError("VariableLayout::algebraic_index: node out of range");
        if (alg_var >= num_algebraic_)
            throw TranscriptionError("VariableLayout::algebraic_index: alg_var out of range");
        return node * variables_per_node() + num_states_ + num_controls_ + alg_var;
    }

 private:
    std::size_t num_states_;
    std::size_t num_controls_;
    std::size_t num_nodes_;
    /// Number of algebraic variables per node. Zero means this is a standard ODE problem;
    /// VariableLayout behaves exactly as the pre-algebraic version when this is zero.
    std::size_t num_algebraic_;
};

}  // namespace goss::transcription
