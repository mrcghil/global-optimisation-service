// include/goss/sim/initial_guess.hpp
#pragma once
#include <cmath>
#include <cstddef>
#include <vector>
#include "goss/model/model.hpp"
#include "goss/sim/errors.hpp"
#include "goss/transcription/transcription.hpp"
#include "goss/transcription/variable_layout.hpp"

namespace goss::sim {

/// Builds an initial decision vector z0 for a compiled OCP:
///  - each state is linearly interpolated between its pinned initial and final
///    values (held constant at the initial when the final is unpinned, or at 0
///    when neither is pinned);
///  - each control is set to the midpoint of its finite box bounds (0 if a side
///    is infinite).
/// This is a cheap, usually-feasible warm start that converges far more
/// reliably than an all-zeros or all-same-value guess.
inline std::vector<double> linear_guess(const model::Model& model,
                                        const transcription::VariableLayout& layout) {
    if (model.num_states() != layout.num_states())
        throw SimError("linear_guess: model.num_states != layout.num_states");
    if (model.num_controls() != layout.num_controls())
        throw SimError("linear_guess: model.num_controls != layout.num_controls");

    const std::size_t num_nodes = layout.num_nodes();
    std::vector<double> z0(layout.total_variables(), 0.0);

    for (std::size_t i = 0; i < layout.num_states(); ++i) {
        const double a = model.initial_fixed(i) ? model.initial_value(i) : 0.0;
        const double b = model.final_fixed(i) ? model.final_value(i) : a;
        for (std::size_t k = 0; k < num_nodes; ++k) {
            const double fraction = (num_nodes > 1)
                ? static_cast<double>(k) / static_cast<double>(num_nodes - 1) : 0.0;
            z0[layout.state_index(k, i)] = a + (b - a) * fraction;
        }
    }
    for (std::size_t j = 0; j < layout.num_controls(); ++j) {
        const double lo = model.control_lower(j);
        const double hi = model.control_upper(j);
        const bool finite = std::abs(lo) < transcription::kInf && std::abs(hi) < transcription::kInf;
        const double value = finite ? 0.5 * (lo + hi) : 0.0;
        for (std::size_t k = 0; k < num_nodes; ++k)
            z0[layout.control_index(k, j)] = value;
    }
    return z0;
}

}  // namespace goss::sim
