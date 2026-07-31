// tests/ad/test_cppadcg_backend.cpp
#include <gtest/gtest.h>
#include "goss/ad/cppadcg_backend.hpp"
#include "ad/fixtures.hpp"

TEST(CppADCGBackend, EvalMatchesQuadratic) {
    goss::ad::test::Quadratic f{3};
    goss::ad::CppADCGBackend backend(f, f.input_size(), "quad_eval");
    EXPECT_EQ(backend.input_size(), 3u);
    EXPECT_EQ(backend.output_size(), 1u);
    auto y = backend.eval({1.0, 2.0, 3.0});
    ASSERT_EQ(y.size(), 1u);
    EXPECT_DOUBLE_EQ(y[0], 7.0);  // 0.5*(1+4+9)
}
