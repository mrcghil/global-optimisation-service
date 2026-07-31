// tests/ad/test_verification.cpp
#include <gtest/gtest.h>
#include <complex>
#include "ad/fixtures.hpp"
#include "ad/verification.hpp"

TEST(FiniteDifference, MatchesQuadraticGradient) {
    goss::ad::test::Quadratic f{3};
    std::vector<double> x{1.0, 2.0, 3.0};
    auto jac = goss::ad::test::finite_difference_jacobian(f, x);
    ASSERT_EQ(jac.size(), 1u);
    ASSERT_EQ(jac[0].size(), 3u);
    // grad of 0.5 xᵀx is x
    EXPECT_NEAR(jac[0][0], 1.0, 1e-6);
    EXPECT_NEAR(jac[0][1], 2.0, 1e-6);
    EXPECT_NEAR(jac[0][2], 3.0, 1e-6);
}

TEST(ComplexStep, MatchesTrigJacobianToHighPrecision) {
    goss::ad::test::Trig f;
    std::vector<double> x{0.5, 1.2};
    auto jac = goss::ad::test::complex_step_jacobian(f, x);
    // f = [sin(x0)*x1, cos(x1)]
    EXPECT_NEAR(jac[0][0], std::cos(0.5) * 1.2, 1e-12);
    EXPECT_NEAR(jac[0][1], std::sin(0.5), 1e-12);
    EXPECT_NEAR(jac[1][0], 0.0, 1e-12);
    EXPECT_NEAR(jac[1][1], -std::sin(1.2), 1e-12);
}
