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
