// tests/transcription/test_hp_mesh.cpp
//
// Tests for HpMesh: construction, validation, and the to_single_segment_hp_mesh factory.
// HpMesh represents the segment-boundary and per-segment-order specification for
// hp-pseudospectral collocation; it is separate from OcpProblem so that transcription
// parameters do not bleed into the problem description.
#include <gtest/gtest.h>
#include <cstddef>
#include <vector>
#include "goss/transcription/hp_mesh.hpp"
#include "goss/transcription/errors.hpp"
#include "goss/transcription/ocp_problem.hpp"  // for Mesh (uniform)

TEST(HpMesh, ConstructAndQueryThreeSegments) {
    goss::transcription::HpMesh hp_mesh;
    hp_mesh.segment_boundary_times = {0.0, 1.0, 2.5, 4.0};  // S=3
    hp_mesh.per_segment_node_count = {4, 5, 3};

    EXPECT_EQ(hp_mesh.num_segments(), 3u);
    EXPECT_DOUBLE_EQ(hp_mesh.t_initial(), 0.0);
    EXPECT_DOUBLE_EQ(hp_mesh.t_final(),   4.0);
    // total_nodes = 4 + 5 + 3 = 12 (duplicated boundary nodes)
    EXPECT_EQ(hp_mesh.total_nodes(), 12u);
    EXPECT_NO_THROW(hp_mesh.validate());
}

TEST(HpMesh, SingleSegmentIsValid) {
    goss::transcription::HpMesh hp_mesh;
    hp_mesh.segment_boundary_times = {0.0, 1.0};  // S=1
    hp_mesh.per_segment_node_count = {8};

    EXPECT_EQ(hp_mesh.num_segments(), 1u);
    EXPECT_EQ(hp_mesh.total_nodes(), 8u);
    EXPECT_NO_THROW(hp_mesh.validate());
}

TEST(HpMesh, ValidateRejectsNonIncreasingBoundaries) {
    goss::transcription::HpMesh hp_mesh;
    hp_mesh.segment_boundary_times = {0.0, 2.0, 1.0};  // not strictly increasing
    hp_mesh.per_segment_node_count = {4, 4};
    EXPECT_THROW(hp_mesh.validate(), goss::transcription::TranscriptionError);
}

TEST(HpMesh, ValidateRejectsEqualBoundaries) {
    goss::transcription::HpMesh hp_mesh;
    hp_mesh.segment_boundary_times = {0.0, 1.0, 1.0, 2.0};  // equal boundaries
    hp_mesh.per_segment_node_count = {4, 4, 4};
    EXPECT_THROW(hp_mesh.validate(), goss::transcription::TranscriptionError);
}

TEST(HpMesh, ValidateRejectsNodeCountLessThanTwo) {
    goss::transcription::HpMesh hp_mesh;
    hp_mesh.segment_boundary_times = {0.0, 1.0, 2.0};
    hp_mesh.per_segment_node_count = {4, 1};  // segment 1 has only 1 node — invalid
    EXPECT_THROW(hp_mesh.validate(), goss::transcription::TranscriptionError);
}

TEST(HpMesh, ValidateRejectsMismatchedSizes) {
    goss::transcription::HpMesh hp_mesh;
    hp_mesh.segment_boundary_times = {0.0, 1.0, 2.0};  // S=2
    hp_mesh.per_segment_node_count = {4, 4, 4};           // 3 entries — mismatch
    EXPECT_THROW(hp_mesh.validate(), goss::transcription::TranscriptionError);
}

TEST(HpMesh, ValidateRejectsEmptyMesh) {
    goss::transcription::HpMesh hp_mesh;
    EXPECT_THROW(hp_mesh.validate(), goss::transcription::TranscriptionError);
}

TEST(HpMesh, ToSingleSegmentMatchesUniformMesh) {
    goss::transcription::Mesh uniform_mesh{0.0, 2.0, /*num_intervals=*/7};
    // num_nodes = 8 for uniform mesh; to_single_segment_hp_mesh should give 8 LGL nodes.
    const goss::transcription::HpMesh hp_mesh =
        goss::transcription::to_single_segment_hp_mesh(uniform_mesh);

    EXPECT_EQ(hp_mesh.num_segments(), 1u);
    EXPECT_DOUBLE_EQ(hp_mesh.t_initial(), 0.0);
    EXPECT_DOUBLE_EQ(hp_mesh.t_final(),   2.0);
    // WHY 8: num_nodes = num_intervals + 1 = 8; single segment has 8 LGL nodes.
    EXPECT_EQ(hp_mesh.per_segment_node_count[0], uniform_mesh.num_nodes());
    EXPECT_EQ(hp_mesh.total_nodes(), uniform_mesh.num_nodes());
    EXPECT_NO_THROW(hp_mesh.validate());
}
