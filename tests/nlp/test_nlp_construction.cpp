#include <gtest/gtest.h>
#include "goss/nlp/errors.hpp"

TEST(NLPError, IsThrowable) {
    EXPECT_THROW(throw goss::nlp::NLPError("boom"), goss::nlp::NLPError);
}
