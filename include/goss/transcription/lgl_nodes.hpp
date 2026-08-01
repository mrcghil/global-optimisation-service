#pragma once
#include <cstddef>
#include <vector>
#include "goss/transcription/errors.hpp"

namespace goss::transcription {

/// Compute the n Legendre-Gauss-Lobatto (LGL) nodes on [-1, 1] and the
/// corresponding quadrature weights. Nodes are returned in ascending order.
/// Throws TranscriptionError if n < 2.
void lgl_nodes_and_weights(std::size_t n,
                           std::vector<double>& nodes_out,
                           std::vector<double>& weights_out);

/// Compute the n×n LGL differentiation matrix D such that
///   (D @ f)[i] ≈ df/dxi at node i,
/// where the nodes are the LGL nodes on [-1, 1].
/// The matrix is stored row-major: D[i*n + j] is D_{ij}.
/// nodes must be the LGL nodes from lgl_nodes_and_weights (size n).
/// Throws TranscriptionError if nodes.size() < 2.
std::vector<double> lgl_differentiation_matrix(const std::vector<double>& nodes);

}  // namespace goss::transcription
