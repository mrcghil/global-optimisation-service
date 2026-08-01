// tests/transcription/test_mesh.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "goss/transcription/mesh.hpp"
#include "goss/transcription/ocp_problem.hpp"  // for Mesh

TEST(NonUniformMesh, FromUniformMeshProducesCorrectNodeTimes) {
    goss::transcription::Mesh uniform_mesh{0.0, 2.0, 4};
    auto nonuniform = goss::transcription::to_nonuniform(uniform_mesh);
    ASSERT_EQ(nonuniform.num_nodes(), 5u);
    ASSERT_EQ(nonuniform.num_intervals(), 4u);
    EXPECT_DOUBLE_EQ(nonuniform.t_initial(), 0.0);
    EXPECT_DOUBLE_EQ(nonuniform.t_final(), 2.0);
    EXPECT_DOUBLE_EQ(nonuniform.interval_width(0), 0.5);
    EXPECT_DOUBLE_EQ(nonuniform.interval_width(3), 0.5);
}

TEST(NonUniformMesh, NonUniformWidthsReportedCorrectly) {
    goss::transcription::NonUniformMesh mesh;
    mesh.node_times = {0.0, 0.1, 0.5, 1.0};
    EXPECT_EQ(mesh.num_intervals(), 3u);
    EXPECT_DOUBLE_EQ(mesh.interval_width(0), 0.1);
    EXPECT_DOUBLE_EQ(mesh.interval_width(1), 0.4);
    EXPECT_DOUBLE_EQ(mesh.interval_width(2), 0.5);
}

TEST(NonUniformMesh, ValidateRejectsNonMonotonicTimes) {
    goss::transcription::NonUniformMesh mesh;
    mesh.node_times = {0.0, 0.5, 0.3, 1.0};  // not strictly increasing
    EXPECT_THROW(mesh.validate(), goss::transcription::TranscriptionError);
}

TEST(NonUniformMesh, ValidateRejectsTooFewNodes) {
    goss::transcription::NonUniformMesh mesh;
    mesh.node_times = {0.5};
    EXPECT_THROW(mesh.validate(), goss::transcription::TranscriptionError);
}

TEST(NonUniformMesh, BisectIntervalsInsertsCorrectMidpoints) {
    goss::transcription::NonUniformMesh base;
    base.node_times = {0.0, 1.0, 2.0, 3.0};
    // Bisect intervals 0 and 2 (indices 0 and 2).
    auto refined = goss::transcription::bisect_intervals(base, {0u, 2u});
    // Original 4 nodes + 2 midpoints = 6 nodes (order: 0.0, 0.5, 1.0, 2.0, 2.5, 3.0)
    ASSERT_EQ(refined.num_nodes(), 6u);
    EXPECT_DOUBLE_EQ(refined.node_times[0], 0.0);
    EXPECT_DOUBLE_EQ(refined.node_times[1], 0.5);  // midpoint of [0,1]
    EXPECT_DOUBLE_EQ(refined.node_times[2], 1.0);
    EXPECT_DOUBLE_EQ(refined.node_times[3], 2.0);
    EXPECT_DOUBLE_EQ(refined.node_times[4], 2.5);  // midpoint of [2,3]
    EXPECT_DOUBLE_EQ(refined.node_times[5], 3.0);
}
