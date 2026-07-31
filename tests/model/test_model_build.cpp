// tests/model/test_model_build.cpp
#include <gtest/gtest.h>
#include "goss/model/model.hpp"
#include "goss/transcription/transcription.hpp"

TEST(ModelSetters, RecordsStateAndControlBounds) {
    goss::model::Model model;
    auto q = model.add_state("q");
    auto r = model.add_control("r");
    model.set_state_bounds(q, 0.0, goss::transcription::kInf);
    model.set_control_bounds(r, -2.0, 2.0);
    EXPECT_DOUBLE_EQ(model.state_lower(0), 0.0);
    EXPECT_DOUBLE_EQ(model.state_upper(0), goss::transcription::kInf);
    EXPECT_DOUBLE_EQ(model.control_lower(0), -2.0);
    EXPECT_DOUBLE_EQ(model.control_upper(0), 2.0);
}

TEST(ModelSetters, RecordsBoundaryConditions) {
    goss::model::Model model;
    auto q = model.add_state("q");
    model.set_initial_state(q, 10.0);
    EXPECT_TRUE(model.initial_fixed(0));
    EXPECT_DOUBLE_EQ(model.initial_value(0), 10.0);
    EXPECT_FALSE(model.final_fixed(0));   // not set → free
}

TEST(ModelSetters, RejectsInvertedBounds) {
    goss::model::Model model;
    auto q = model.add_state("q");
    EXPECT_THROW(model.set_state_bounds(q, 5.0, -5.0), goss::model::ModelError);
}

TEST(ModelSetters, RejectsOutOfRangeHandle) {
    goss::model::Model model;
    model.add_state("q");
    goss::model::StateHandle bogus{7};
    EXPECT_THROW(model.set_state_bounds(bogus, 0.0, 1.0), goss::model::ModelError);
}
