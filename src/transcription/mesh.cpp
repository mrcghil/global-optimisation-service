// src/transcription/mesh.cpp
#include "goss/transcription/mesh.hpp"
#include <algorithm>
#include <string>

namespace goss::transcription {

void NonUniformMesh::validate() const {
    if (node_times.size() < 2)
        throw TranscriptionError("NonUniformMesh: must have at least 2 node times");
    for (std::size_t k = 0; k + 1 < node_times.size(); ++k) {
        if (node_times[k + 1] <= node_times[k])
            throw TranscriptionError(
                "NonUniformMesh: node_times must be strictly increasing "
                "(violated at index " + std::to_string(k) + ")");
    }
}

NonUniformMesh to_nonuniform(const Mesh& uniform_mesh) {
    uniform_mesh.validate();
    NonUniformMesh result;
    const std::size_t n = uniform_mesh.num_nodes();
    result.node_times.resize(n);
    const double h = uniform_mesh.interval_width();
    for (std::size_t k = 0; k < n; ++k)
        result.node_times[k] = uniform_mesh.t_initial + static_cast<double>(k) * h;
    return result;
}

NonUniformMesh bisect_intervals(const NonUniformMesh& base_mesh,
                                const std::vector<std::size_t>& intervals_to_refine) {
    base_mesh.validate();

    // Mark which intervals to bisect.
    std::vector<bool> should_bisect(base_mesh.num_intervals(), false);
    for (std::size_t idx : intervals_to_refine) {
        if (idx >= base_mesh.num_intervals())
            throw TranscriptionError(
                "bisect_intervals: interval index " + std::to_string(idx) +
                " >= num_intervals " + std::to_string(base_mesh.num_intervals()));
        should_bisect[idx] = true;
    }

    NonUniformMesh result;
    result.node_times.reserve(base_mesh.num_nodes() + intervals_to_refine.size());
    result.node_times.push_back(base_mesh.node_times[0]);
    for (std::size_t k = 0; k < base_mesh.num_intervals(); ++k) {
        if (should_bisect[k]) {
            const double midpoint =
                0.5 * (base_mesh.node_times[k] + base_mesh.node_times[k + 1]);
            result.node_times.push_back(midpoint);
        }
        result.node_times.push_back(base_mesh.node_times[k + 1]);
    }
    return result;
}

}  // namespace goss::transcription
