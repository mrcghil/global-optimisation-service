// include/goss/transcription/hp_mesh.hpp
#pragma once
#include <cstddef>
#include <vector>
#include "goss/transcription/errors.hpp"
#include "goss/transcription/ocp_problem.hpp"  // for Mesh (uniform)

namespace goss::transcription {

/// Mesh specification for hp-pseudospectral collocation.
///
/// Partitions [t0, tf] into S segments, each with an independent set of
/// Legendre-Gauss-Lobatto (LGL) nodes. Boundary nodes are DUPLICATED (not shared):
/// segment s owns all n_s LGL nodes in [t_a^s, t_b^s] including its endpoints.
/// State continuity at segment boundaries is enforced via explicit equality constraints
/// in the NLP (not by sharing a variable). This keeps per-segment assembly uniform.
///
/// Controls are DISCONTINUOUS across segment boundaries (standard hp-OC convention).
struct HpMesh {
    /// Segment boundary times: size S+1.
    /// segment_boundary_times[0] = t_initial (overall),
    /// segment_boundary_times[S] = t_final (overall).
    /// Must be strictly increasing.
    std::vector<double> segment_boundary_times;

    /// Number of LGL nodes per segment: size S.
    /// per_segment_node_count[s] >= 2 for each s
    /// (need at least 2 LGL nodes, i.e. 1 LGL interval, per segment).
    std::vector<std::size_t> per_segment_node_count;

    std::size_t num_segments() const {
        return per_segment_node_count.size();
    }

    double t_initial() const {
        if (segment_boundary_times.empty())
            throw TranscriptionError("HpMesh::t_initial: mesh is empty");
        return segment_boundary_times.front();
    }

    double t_final() const {
        if (segment_boundary_times.empty())
            throw TranscriptionError("HpMesh::t_final: mesh is empty");
        return segment_boundary_times.back();
    }

    /// Total number of global decision-variable nodes (sum of per-segment counts).
    /// WHY sum (not sum - (S-1)): boundary nodes are duplicated; segment s occupies
    /// contiguous global nodes [offset_s, offset_s + n_s), simplifying assembly.
    std::size_t total_nodes() const {
        std::size_t total = 0;
        for (const std::size_t node_count : per_segment_node_count)
            total += node_count;
        return total;
    }

    /// Validate the HpMesh.
    /// Throws TranscriptionError if:
    ///   - segment_boundary_times is empty or has fewer than 2 entries
    ///   - per_segment_node_count is empty or has size != segment_boundary_times.size() - 1
    ///   - segment_boundary_times is not strictly increasing
    ///   - any per_segment_node_count[s] < 2
    void validate() const;
};

/// Build a single-segment HpMesh from a uniform Mesh.
/// The resulting HpMesh has S=1, t_initial = uniform_mesh.t_initial,
/// t_final = uniform_mesh.t_final, and per_segment_node_count[0] = uniform_mesh.num_nodes().
/// WHY: allows compile_hp with S=1 to be a drop-in for compile() on the same OcpProblem.
HpMesh to_single_segment_hp_mesh(const Mesh& uniform_mesh);

}  // namespace goss::transcription
