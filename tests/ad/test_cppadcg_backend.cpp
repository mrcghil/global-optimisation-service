// tests/ad/test_cppadcg_backend.cpp
#include <gtest/gtest.h>
#include "goss/ad/cppadcg_backend.hpp"
#include "ad/fixtures.hpp"
#include "ad/verification.hpp"
#include <algorithm>
#include <random>

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

TEST(CppADCGBackend, QuadraticHessianIsIdentity) {
    goss::ad::test::Quadratic f{3};
    goss::ad::CppADCGBackend backend(f, f.input_size(), "quad_hess");
    std::vector<double> x{1.0, 2.0, 3.0};
    std::vector<double> weights{1.0};  // single output
    auto pattern = backend.hessian_sparsity();
    auto values = backend.eval_hessian(x, weights);
    ASSERT_EQ(pattern.size(), values.size());
    // Hessian of 0.5 xᵀx is I: diagonal entries = 1, off-diagonal = 0 (absent)
    for (std::size_t k = 0; k < pattern.size(); ++k) {
        auto [row, col] = pattern[k];
        EXPECT_DOUBLE_EQ(values[k], row == col ? 1.0 : 0.0);
    }
}

TEST(CppADCGBackend, HessianSparsityIsLowerTriangle) {
    goss::ad::test::Quadratic f{4};
    goss::ad::CppADCGBackend backend(f, f.input_size(), "quad_hess_tri");
    for (auto [row, col] : backend.hessian_sparsity()) {
        EXPECT_GE(row, col) << "hessian should be lower triangular";
    }
}

TEST(CppADCGBackend, EvalHessianRejectsWrongWeightsSize) {
    // Quadratic{3} has output_size == 1; passing 2 weights must throw ADError.
    goss::ad::test::Quadratic f{3};
    goss::ad::CppADCGBackend backend(f, f.input_size(), "quad_hess_weights_err");
    EXPECT_THROW(backend.eval_hessian({1.0, 2.0, 3.0}, {1.0, 2.0}),
                 goss::ad::ADError);
}

TEST(CppADCGBackend, RosenbrockJacobianMatchesBothOracles) {
    goss::ad::test::Rosenbrock f;
    goss::ad::CppADCGBackend backend(f, f.input_size(), "rosen_all");
    auto pattern = backend.jacobian_sparsity();

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    for (int trial = 0; trial < 20; ++trial) {
        std::vector<double> x{dist(rng), dist(rng)};
        auto values = backend.eval_jacobian(x);
        auto fd = goss::ad::test::finite_difference_jacobian(f, x);
        auto cs = goss::ad::test::complex_step_jacobian(f, x);
        for (std::size_t k = 0; k < pattern.size(); ++k) {
            auto [row, col] = pattern[k];
            EXPECT_NEAR(values[k], cs[row][col], 1e-8);
            EXPECT_NEAR(values[k], fd[row][col], 1e-4);
        }
    }
}
