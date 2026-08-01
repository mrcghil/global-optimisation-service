// include/goss/transcription/mesh.hpp
#pragma once
#include <cstddef>
#include <vector>
#include "goss/transcription/errors.hpp"
#include "goss/transcription/ocp_problem.hpp"  // for Mesh (uniform)

namespace goss::transcription {

/// A mesh with explicitly stored, arbitrarily spaced node times.
/// The uniform Mesh is a special case; use to_nonuniform() to convert.
/// Invariant: node_times is strictly increasing with at least 2 entries.
struct NonUniformMesh {
    std::vector<double> node_times;

    std::size_t num_nodes()     const { return node_times.size(); }
    // Guard against unsigned underflow: an empty or single-node vector has no intervals.
    std::size_t num_intervals() const { return node_times.size() < 2 ? 0 : node_times.size() - 1; }

    double t_initial() const {
        if (node_times.empty()) throw TranscriptionError("NonUniformMesh::t_initial: mesh is empty");
        return node_times.front();
    }
    double t_final() const {
        if (node_times.empty()) throw TranscriptionError("NonUniformMesh::t_final: mesh is empty");
        return node_times.back();
    }

    /// Width of interval k = t[k+1] - t[k]. Throws if k is out of range.
    double interval_width(std::size_t k) const {
        if (k >= num_intervals())
            throw TranscriptionError("NonUniformMesh::interval_width: k out of range");
        return node_times[k + 1] - node_times[k];
    }

    /// Throws TranscriptionError if the mesh is malformed (fewer than 2 nodes
    /// or node_times not strictly increasing).
    void validate() const;
};

/// Convert a uniform Mesh into a NonUniformMesh with evenly spaced node_times.
NonUniformMesh to_nonuniform(const Mesh& uniform_mesh);

/// Return a new NonUniformMesh with each interval whose index is in
/// intervals_to_refine bisected (midpoint inserted).
/// intervals_to_refine must contain valid interval indices (< base.num_intervals()).
NonUniformMesh bisect_intervals(const NonUniformMesh& base_mesh,
                                const std::vector<std::size_t>& intervals_to_refine);

}  // namespace goss::transcription
