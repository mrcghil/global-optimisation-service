// tests/model/test_expr_nodes.cpp
#include <gtest/gtest.h>
#include "goss/model/expr/errors.hpp"

TEST(ExprError, IsThrowableAndCarriesMessage) {
    try {
        throw goss::model::expr::ExprError("test error");
    } catch (const goss::model::expr::ExprError& caught_error) {
        EXPECT_STREQ(caught_error.what(), "test error");
    }
}
