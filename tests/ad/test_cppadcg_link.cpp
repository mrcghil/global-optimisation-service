// tests/ad/test_cppadcg_link.cpp
#include <gtest/gtest.h>
// Use the umbrella header that includes all forward declarations before cg/cg.hpp
#include <cppad/cg.hpp>
#include <vector>

TEST(CppADCGLink, CanRecordTrivialFunction) {
    using CGD = CppAD::cg::CG<double>;
    using ADCG = CppAD::AD<CGD>;
    std::vector<ADCG> x(1);
    CppAD::Independent(x);
    std::vector<ADCG> y(1);
    y[0] = x[0] * x[0];
    CppAD::ADFun<CGD> fun(x, y);
    EXPECT_EQ(fun.Domain(), 1u);
    EXPECT_EQ(fun.Range(), 1u);
}
