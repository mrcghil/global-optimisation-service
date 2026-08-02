// src/transcription/hp_mesh.cpp
#include "goss/transcription/hp_mesh.hpp"
#include <string>

namespace goss::transcription {

void HpMesh::validate() const {
    if (segment_boundary_times.empty() || segment_boundary_times.size() < 2)
        throw TranscriptionError(
            "HpMesh::validate: segment_boundary_times must have at least 2 entries (t0 and tf)");

    const std::size_t num_seg = segment_boundary_times.size() - 1;
    if (per_segment_node_count.size() != num_seg)
        throw TranscriptionError(
            "HpMesh::validate: per_segment_node_count.size() must equal "
            "segment_boundary_times.size() - 1 (number of segments S)");

    // Strictly increasing boundary times.
    for (std::size_t seg_idx = 0; seg_idx < num_seg; ++seg_idx) {
        if (segment_boundary_times[seg_idx + 1] <= segment_boundary_times[seg_idx])
            throw TranscriptionError(
                "HpMesh::validate: segment_boundary_times must be strictly increasing "
                "(segment " + std::to_string(seg_idx) + " has zero or negative width)");
    }

    // Each segment needs at least 2 LGL nodes (to have at least 1 LGL interval).
    for (std::size_t seg_idx = 0; seg_idx < num_seg; ++seg_idx) {
        if (per_segment_node_count[seg_idx] < 2)
            throw TranscriptionError(
                "HpMesh::validate: per_segment_node_count[" + std::to_string(seg_idx) +
                "] must be >= 2 (need at least 2 LGL nodes per segment)");
    }
}

HpMesh to_single_segment_hp_mesh(const Mesh& uniform_mesh) {
    uniform_mesh.validate();
    HpMesh hp_mesh;
    hp_mesh.segment_boundary_times = {uniform_mesh.t_initial, uniform_mesh.t_final};
    // WHY num_nodes(): the LGL scheme interprets num_intervals+1 as the node count
    // (lgl_nodes_and_weights is called with nn = mesh.num_nodes()), so the single-segment
    // hp mesh must have per_segment_node_count[0] = num_nodes() to match compile()'s behavior.
    hp_mesh.per_segment_node_count = {uniform_mesh.num_nodes()};
    return hp_mesh;
}

}  // namespace goss::transcription
