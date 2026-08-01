#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include "goss/transcription/lgl_nodes.hpp"

TEST(LglNodes, TwoPointNodesAreEndpoints) {
    std::vector<double> nodes, weights;
    goss::transcription::lgl_nodes_and_weights(2, nodes, weights);
    ASSERT_EQ(nodes.size(), 2u);
    EXPECT_DOUBLE_EQ(nodes[0], -1.0);
    EXPECT_DOUBLE_EQ(nodes[1],  1.0);
    EXPECT_DOUBLE_EQ(weights[0], 1.0);
    EXPECT_DOUBLE_EQ(weights[1], 1.0);
}

TEST(LglNodes, FivePointNodesAreSymmetric) {
    std::vector<double> nodes, weights;
    goss::transcription::lgl_nodes_and_weights(5, nodes, weights);
    ASSERT_EQ(nodes.size(), 5u);
    // Symmetry about 0: nodes[i] == -nodes[4-i]
    for (std::size_t i = 0; i < 5u; ++i)
        EXPECT_NEAR(nodes[i], -nodes[4 - i], 1e-14);
    // Middle node should be 0
    EXPECT_NEAR(nodes[2], 0.0, 1e-14);
}

TEST(LglNodes, WeightsSumToTwo) {
    for (std::size_t n : {2u, 3u, 4u, 5u, 6u, 8u}) {
        std::vector<double> nodes, weights;
        goss::transcription::lgl_nodes_and_weights(n, nodes, weights);
        const double weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
        EXPECT_NEAR(weight_sum, 2.0, 1e-12) << "LGL weights must sum to 2 (integral of 1 on [-1,1])";
    }
}

TEST(LglNodes, DifferentiationMatrixOfConstantIsZero) {
    std::vector<double> nodes, weights;
    goss::transcription::lgl_nodes_and_weights(5, nodes, weights);
    auto D = goss::transcription::lgl_differentiation_matrix(nodes);
    const std::size_t n = nodes.size();
    // D @ [1,1,1,...] should be all zeros (derivative of constant).
    for (std::size_t i = 0; i < n; ++i) {
        double row_sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) row_sum += D[i * n + j];
        EXPECT_NEAR(row_sum, 0.0, 1e-12) << "D @ const must be 0 at row " << i;
    }
}

TEST(LglNodes, DifferentiationMatrixOfLinearIsOne) {
    // D @ x_j should be 1 at every node (derivative of identity is 1).
    std::vector<double> nodes, weights;
    goss::transcription::lgl_nodes_and_weights(5, nodes, weights);
    auto D = goss::transcription::lgl_differentiation_matrix(nodes);
    const std::size_t n = nodes.size();
    for (std::size_t i = 0; i < n; ++i) {
        double deriv = 0.0;
        for (std::size_t j = 0; j < n; ++j) deriv += D[i * n + j] * nodes[j];
        EXPECT_NEAR(deriv, 1.0, 1e-12) << "D @ x must be 1 at row " << i;
    }
}

TEST(LglNodes, ThreePointKnownValues) {
    std::vector<double> nodes, weights;
    goss::transcription::lgl_nodes_and_weights(3, nodes, weights);
    ASSERT_EQ(nodes.size(), 3u);
    EXPECT_NEAR(nodes[0], -1.0, 1e-14);
    EXPECT_NEAR(nodes[1],  0.0, 1e-14);
    EXPECT_NEAR(nodes[2],  1.0, 1e-14);
    EXPECT_NEAR(weights[0], 1.0/3.0, 1e-14);
    EXPECT_NEAR(weights[1], 4.0/3.0, 1e-14);
    EXPECT_NEAR(weights[2], 1.0/3.0, 1e-14);
}

TEST(LglNodes, DifferentiationMatrixExactUpToDegreeN) {
    // For n LGL nodes (degree N=n-1), D differentiates x^k exactly for k=0..N.
    for (std::size_t n : {4u, 5u, 6u}) {
        std::vector<double> nodes, weights;
        goss::transcription::lgl_nodes_and_weights(n, nodes, weights);
        auto D = goss::transcription::lgl_differentiation_matrix(nodes);
        const std::size_t N = n - 1;
        for (std::size_t k = 2; k <= N; ++k) {
            for (std::size_t i = 0; i < n; ++i) {
                double computed = 0.0;
                for (std::size_t j = 0; j < n; ++j)
                    computed += D[i * n + j] * std::pow(nodes[j], static_cast<double>(k));
                const double expected = static_cast<double>(k) * std::pow(nodes[i], static_cast<double>(k - 1));
                EXPECT_NEAR(computed, expected, 1e-9) << "n=" << n << " k=" << k << " row=" << i;
            }
        }
    }
}

TEST(LglNodes, RejectsTooFewPoints) {
    std::vector<double> nodes, weights;
    EXPECT_THROW(goss::transcription::lgl_nodes_and_weights(1, nodes, weights),
                 goss::transcription::TranscriptionError);
}
