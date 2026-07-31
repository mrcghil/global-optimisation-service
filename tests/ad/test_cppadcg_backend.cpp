// tests/ad/test_cppadcg_backend.cpp
#include <gtest/gtest.h>
#include "goss/ad/cppadcg_backend.hpp"
#include "ad/fixtures.hpp"
#include "ad/verification.hpp"
#include <algorithm>

TEST(CppADCGBackend, EvalMatchesQuadratic) {
    goss::ad::test::Quadratic f{3};
    goss::ad::CppADCGBackend backend(f, f.input_size(), "quad_eval");
    EXPECT_EQ(backend.input_size(), 3u);
    EXPECT_EQ(backend.output_size(), 1u);
    auto y = backend.eval({1.0, 2.0, 3.0});
    ASSERT_EQ(y.size(), 1u);
    EXPECT_DOUBLE_EQ(y[0], 7.0);  // 0.5*(1+4+9)
}

TEST(CppADCGBackend, JacobianValuesMatchComplexStep) {
    goss::ad::test::Trig f;
    goss::ad::CppADCGBackend backend(f, f.input_size(), "trig_jac");
    std::vector<double> x{0.5, 1.2};
    auto pattern = backend.jacobian_sparsity();
    auto values = backend.eval_jacobian(x);
    ASSERT_EQ(pattern.size(), values.size());

    auto reference = goss::ad::test::complex_step_jacobian(f, x);
    for (std::size_t k = 0; k < pattern.size(); ++k) {
        auto [row, col] = pattern[k];
        EXPECT_NEAR(values[k], reference[row][col], 1e-9);
    }
}

TEST(CppADCGBackend, BandedJacobianSparsityIsTridiagonal) {
    goss::ad::test::Banded f{5};
    goss::ad::CppADCGBackend backend(f, f.input_size(), "banded_jac");
    for (auto [row, col] : backend.jacobian_sparsity()) {
        // output i depends only on x[i] and x[i+1]
        EXPECT_TRUE(col == row || col == row + 1)
            << "unexpected nonzero at (" << row << "," << col << ")";
    }
}
