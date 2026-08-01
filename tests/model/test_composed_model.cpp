#include <gtest/gtest.h>
#include "goss/model/errors.hpp"

TEST(ComponentErrorInComposed, IsAlsoRuntimeError) {
    try {
        throw goss::model::ComponentError("test");
    } catch (const std::runtime_error& error) {
        EXPECT_EQ(std::string(error.what()), "test");
    }
}
