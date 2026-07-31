// tests/solver/test_solver_link.cpp
// Link test: verifies that IPOPT and NLopt headers are reachable and both
// libraries link correctly.  No solver math here — construction only.
//
// Note: on Ubuntu 24.04 the coinor-libipopt-dev package (IPOPT 3.11.9) installs
// headers under /usr/include/coin/ and pkg-config reports -I/usr/include/coin,
// so the correct include form is <IpIpoptApplication.hpp> (no subdir prefix).
#include <gtest/gtest.h>
#include <IpIpoptApplication.hpp>
#include <IpTNLP.hpp>
#include <nlopt.hpp>

TEST(SolverLink, IpoptApplicationConstructs) {
    Ipopt::SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();
    ASSERT_TRUE(Ipopt::IsValid(app));
}

TEST(SolverLink, NloptOptConstructs) {
    nlopt::opt opt(nlopt::LN_COBYLA, 2);
    EXPECT_EQ(opt.get_dimension(), 2u);
}
