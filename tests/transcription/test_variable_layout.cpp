#include <gtest/gtest.h>
#include "goss/transcription/errors.hpp"
#include "goss/transcription/variable_layout.hpp"

TEST(TranscriptionError, IsThrowable) {
    EXPECT_THROW(throw goss::transcription::TranscriptionError("boom"),
                 goss::transcription::TranscriptionError);
}

TEST(VariableLayout, ComputesTotalVariables) {
    goss::transcription::VariableLayout layout(2, 1, 3);  // 2 states, 1 control, 3 nodes
    EXPECT_EQ(layout.total_variables(), 3u * (2u + 1u));   // 9
}

TEST(VariableLayout, StateAndControlIndicesAreNodeGrouped) {
    goss::transcription::VariableLayout layout(2, 1, 3);
    // node 0: x0=0, x1=1, u0=2 ; node 1: x0=3, x1=4, u0=5 ; node 2: 6,7,8
    EXPECT_EQ(layout.state_index(0, 0), 0u);
    EXPECT_EQ(layout.state_index(0, 1), 1u);
    EXPECT_EQ(layout.control_index(0, 0), 2u);
    EXPECT_EQ(layout.state_index(1, 0), 3u);
    EXPECT_EQ(layout.control_index(1, 0), 5u);
    EXPECT_EQ(layout.state_index(2, 1), 7u);
}

TEST(VariableLayout, RejectsBadDimensions) {
    EXPECT_THROW(goss::transcription::VariableLayout(0, 1, 3), goss::transcription::TranscriptionError);
    EXPECT_THROW(goss::transcription::VariableLayout(2, 1, 1), goss::transcription::TranscriptionError);
}

TEST(VariableLayout, ZeroControlsIsAllowed) {
    goss::transcription::VariableLayout layout(2, 0, 4);  // states only
    EXPECT_EQ(layout.total_variables(), 8u);
    EXPECT_EQ(layout.state_index(3, 1), 7u);
}

#include "goss/transcription/ocp_problem.hpp"
#include "transcription/ocp_fixtures.hpp"

TEST(OcpProblem, MeshComputesNodesAndWidth) {
    goss::transcription::Mesh mesh{0.0, 2.0, 4};
    EXPECT_EQ(mesh.num_nodes(), 5u);
    EXPECT_DOUBLE_EQ(mesh.interval_width(), 0.5);
}

TEST(OcpProblem, ExponentialDecayFixtureEvaluates) {
    auto ocp = goss::transcription::test::make_exponential_decay(/*x0=*/1.0, /*tf=*/1.0, /*intervals=*/10);
    EXPECT_EQ(ocp.num_states, 1u);
    EXPECT_EQ(ocp.num_controls, 0u);
    std::vector<double> x{2.0}, u{};
    auto dx = ocp.dynamics(x, u, 0.0);
    ASSERT_EQ(dx.size(), 1u);
    EXPECT_DOUBLE_EQ(dx[0], -2.0);  // dx/dt = -x
}

TEST(VariableLayout, AlgebraicIndexAppendsAfterControls) {
    // 2 states, 1 control, 1 algebraic, 3 nodes.
    // Per-node stride = 2+1+1 = 4.
    // node 0: x0=0, x1=1, u0=2, alg0=3
    // node 1: x0=4, x1=5, u0=6, alg0=7
    // node 2: x0=8, x1=9, u0=10, alg0=11
    goss::transcription::VariableLayout layout(2, 1, 3, /*num_algebraic=*/1);
    EXPECT_EQ(layout.total_variables(), 12u);
    EXPECT_EQ(layout.variables_per_node(), 4u);
    EXPECT_EQ(layout.state_index(0, 0), 0u);
    EXPECT_EQ(layout.state_index(0, 1), 1u);
    EXPECT_EQ(layout.control_index(0, 0), 2u);
    EXPECT_EQ(layout.algebraic_index(0, 0), 3u);
    EXPECT_EQ(layout.state_index(1, 0), 4u);
    EXPECT_EQ(layout.control_index(1, 0), 6u);
    EXPECT_EQ(layout.algebraic_index(1, 0), 7u);
    EXPECT_EQ(layout.algebraic_index(2, 0), 11u);
}

TEST(VariableLayout, ZeroAlgebraicPreservesExistingIndices) {
    // With num_algebraic=0 (explicit), indices must be identical to the 3-arg constructor.
    goss::transcription::VariableLayout layout_old(2, 1, 3);
    goss::transcription::VariableLayout layout_new(2, 1, 3, 0);
    EXPECT_EQ(layout_old.total_variables(), layout_new.total_variables());
    EXPECT_EQ(layout_old.state_index(1, 0), layout_new.state_index(1, 0));
    EXPECT_EQ(layout_old.control_index(2, 0), layout_new.control_index(2, 0));
}

TEST(VariableLayout, AlgebraicIndexOutOfRangeThrows) {
    goss::transcription::VariableLayout layout(2, 1, 3, 1);
    EXPECT_THROW(layout.algebraic_index(3, 0), goss::transcription::TranscriptionError);  // node OOB
    EXPECT_THROW(layout.algebraic_index(0, 1), goss::transcription::TranscriptionError);  // alg_var OOB
}

TEST(VariableLayout, AlgebraicIndexWithZeroAlgebraicThrows) {
    // Calling algebraic_index when num_algebraic==0 is always a programming error.
    goss::transcription::VariableLayout layout(2, 1, 3, 0);
    EXPECT_THROW(layout.algebraic_index(0, 0), goss::transcription::TranscriptionError);
}

TEST(OcpProblem, NewAlgebraicFieldsDefaultToEmpty) {
    // Existing two-param OcpProblem must compile unchanged and have zero algebraics.
    auto ocp = goss::transcription::test::make_exponential_decay(1.0, 1.0, 5);
    EXPECT_EQ(ocp.num_algebraic, 0u);
    EXPECT_TRUE(ocp.algebraic_lower_bounds.empty());
    EXPECT_TRUE(ocp.algebraic_upper_bounds.empty());
}
