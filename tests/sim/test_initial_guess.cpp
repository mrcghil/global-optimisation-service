// tests/sim/test_initial_guess.cpp
#include <gtest/gtest.h>
#include "goss/sim/errors.hpp"
#include "goss/sim/initial_guess.hpp"
#include "goss/model/model.hpp"
#include "goss/transcription/variable_layout.hpp"

TEST(SimError, IsThrowable) {
    EXPECT_THROW(throw goss::sim::SimError("boom"), goss::sim::SimError);
}

TEST(LinearGuess, InterpolatesPinnedStateEndpoints) {
    goss::model::Model model;
    auto x = model.add_state("x");
    model.set_initial_state(x, 0.0);
    model.set_final_state(x, 10.0);
    model.set_mesh(0.0, 1.0, 4);  // 5 nodes
    goss::transcription::VariableLayout layout(1, 0, 5);
    auto z0 = goss::sim::linear_guess(model, layout);
    ASSERT_EQ(z0.size(), 5u);
    EXPECT_DOUBLE_EQ(z0[layout.state_index(0, 0)], 0.0);
    EXPECT_DOUBLE_EQ(z0[layout.state_index(2, 0)], 5.0);   // midpoint
    EXPECT_DOUBLE_EQ(z0[layout.state_index(4, 0)], 10.0);
}

TEST(LinearGuess, HoldsUnpinnedStateAtInitial) {
    goss::model::Model model;
    auto x = model.add_state("x");
    model.set_initial_state(x, 3.0);   // no final → hold constant
    model.set_mesh(0.0, 1.0, 3);
    goss::transcription::VariableLayout layout(1, 0, 4);
    auto z0 = goss::sim::linear_guess(model, layout);
    for (std::size_t k = 0; k < 4; ++k)
        EXPECT_DOUBLE_EQ(z0[layout.state_index(k, 0)], 3.0);
}

TEST(LinearGuess, ControlAtBoundMidpoint) {
    goss::model::Model model;
    model.add_state("x");
    auto u = model.add_control("u");
    model.set_control_bounds(u, -2.0, 6.0);   // midpoint 2.0
    model.set_mesh(0.0, 1.0, 2);
    goss::transcription::VariableLayout layout(1, 1, 3);
    auto z0 = goss::sim::linear_guess(model, layout);
    for (std::size_t k = 0; k < 3; ++k)
        EXPECT_DOUBLE_EQ(z0[layout.control_index(k, 0)], 2.0);
}

TEST(LinearGuess, RejectsDimensionMismatch) {
    goss::model::Model model;
    model.add_state("x");
    model.set_mesh(0.0, 1.0, 2);
    goss::transcription::VariableLayout layout(2, 0, 3);  // 2 states != model's 1
    EXPECT_THROW(goss::sim::linear_guess(model, layout), goss::sim::SimError);
}
