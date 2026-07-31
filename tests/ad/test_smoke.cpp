// tests/ad/test_smoke.cpp
#include <gtest/gtest.h>
#include "goss/ad/types.hpp"

TEST(Smoke, TypesCompileAndDefault) {
    goss::ad::SparsityPattern pattern;
    goss::ad::SparseTriplets triplets;
    EXPECT_TRUE(pattern.empty());
    EXPECT_TRUE(triplets.empty());
}
