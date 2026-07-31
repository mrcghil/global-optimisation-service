// tests/ad/test_fixtures.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "ad/fixtures.hpp"  // fixtures are test-only; included relative to tests/ dir

TEST(Fixtures, QuadraticValueMatchesClosedForm) {
    goss::ad::test::Quadratic f{3};
    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = f(x);
    ASSERT_EQ(y.size(), 1u);
    EXPECT_DOUBLE_EQ(y[0], 0.5 * (1.0 + 4.0 + 9.0));
}

TEST(Fixtures, BandedOutputDependsOnNeighbors) {
    goss::ad::test::Banded f{4};
    EXPECT_EQ(f.input_size(), 4u);
    EXPECT_EQ(f.output_size(), 4u);
}
