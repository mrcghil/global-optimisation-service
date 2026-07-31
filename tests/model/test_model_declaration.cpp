#include <gtest/gtest.h>
#include <cstddef>
#include <vector>
#include "goss/model/errors.hpp"
#include "goss/model/handles.hpp"

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
