// include/goss/sim/trajectory.hpp
#pragma once
#include <cstddef>
#include <string>
#include <vector>
#include "goss/model/model.hpp"
#include "goss/sim/errors.hpp"
#include "goss/solver/solver_result.hpp"
#include "goss/transcription/ocp_problem.hpp"       // Mesh
#include "goss/transcription/variable_layout.hpp"

namespace goss::sim {

/// A solved trajectory unpacked from the flat decision vector: node times plus
/// named per-state and per-control series (each series is one value per node).
struct Trajectory {
    std::vector<double> times;
    std::vector<std::string> state_names;
    std::vector<std::string> control_names;
    std::vector<std::vector<double>> states;    // states[i][k]
    std::vector<std::vector<double>> controls;  // controls[j][k]

    const std::vector<double>& state(const std::string& name) const {
        for (std::size_t i = 0; i < state_names.size(); ++i)
            if (state_names[i] == name) return states[i];
        throw SimError("Trajectory::state: unknown state '" + name + "'");
    }
    const std::vector<double>& control(const std::string& name) const {
        for (std::size_t j = 0; j < control_names.size(); ++j)
            if (control_names[j] == name) return controls[j];
        throw SimError("Trajectory::control: unknown control '" + name + "'");
    }
};

inline Trajectory extract_trajectory(const solver::SolverResult& result,
                                     const transcription::VariableLayout& layout,
                                     const model::Model& model,
                                     const transcription::Mesh& mesh) {
    if (result.x.size() != layout.total_variables())
        throw SimError("extract_trajectory: result.x size != layout.total_variables");
    if (mesh.num_nodes() != layout.num_nodes())
        throw SimError("extract_trajectory: mesh.num_nodes != layout.num_nodes");

    const std::size_t num_nodes = layout.num_nodes();
    const double width = mesh.interval_width();
    Trajectory traj;
    traj.times.resize(num_nodes);
    for (std::size_t k = 0; k < num_nodes; ++k)
        traj.times[k] = mesh.t_initial + static_cast<double>(k) * width;

    traj.states.assign(layout.num_states(), std::vector<double>(num_nodes, 0.0));
    traj.state_names.resize(layout.num_states());
    for (std::size_t i = 0; i < layout.num_states(); ++i) {
        traj.state_names[i] = model.state_name(i);
        for (std::size_t k = 0; k < num_nodes; ++k)
            traj.states[i][k] = result.x[layout.state_index(k, i)];
    }
    traj.controls.assign(layout.num_controls(), std::vector<double>(num_nodes, 0.0));
    traj.control_names.resize(layout.num_controls());
    for (std::size_t j = 0; j < layout.num_controls(); ++j) {
        traj.control_names[j] = model.control_name(j);
        for (std::size_t k = 0; k < num_nodes; ++k)
            traj.controls[j][k] = result.x[layout.control_index(k, j)];
    }
    return traj;
}

}  // namespace goss::sim
