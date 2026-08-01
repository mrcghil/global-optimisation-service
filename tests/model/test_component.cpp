#include <gtest/gtest.h>
#include "goss/model/errors.hpp"

TEST(ComponentError, IsThrowable) {
    EXPECT_THROW(
        throw goss::model::ComponentError("boom"),
        goss::model::ComponentError);
}
