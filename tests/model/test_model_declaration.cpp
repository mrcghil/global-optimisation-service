#include <gtest/gtest.h>
#include <cstddef>
#include <vector>
#include "goss/model/errors.hpp"
#include "goss/model/handles.hpp"
#include "goss/model/model.hpp"

TEST(ModelError, IsThrowable) {
    EXPECT_THROW(throw goss::model::ModelError("boom"), goss::model::ModelError);
}

TEST(Handles, ImplicitlyIndexVectors) {
    goss::model::StateHandle q{2};
    goss::model::ControlHandle r{0};
    std::vector<double> x{10.0, 20.0, 30.0};
    std::vector<double> u{5.0};
    EXPECT_DOUBLE_EQ(x[q], 30.0);   // x[2]
    EXPECT_DOUBLE_EQ(u[r], 5.0);    // u[0]
}

TEST(ModelDeclaration, AssignsSequentialIndices) {
    goss::model::Model model;
    auto q = model.add_state("queue_length");
    auto x2 = model.add_state("second");
    auto rate = model.add_control("service_rate");
    EXPECT_EQ(q.index, 0u);
    EXPECT_EQ(x2.index, 1u);
    EXPECT_EQ(rate.index, 0u);          // controls indexed independently
    EXPECT_EQ(model.num_states(), 2u);
    EXPECT_EQ(model.num_controls(), 1u);
    EXPECT_EQ(model.state_name(0), "queue_length");
    EXPECT_EQ(model.control_name(0), "service_rate");
}

TEST(ModelDeclaration, RejectsDuplicateNames) {
    goss::model::Model model;
    model.add_state("x");
    EXPECT_THROW(model.add_state("x"), goss::model::ModelError);
    EXPECT_THROW(model.add_control("x"), goss::model::ModelError);  // clash across kinds
}
