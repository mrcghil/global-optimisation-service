// tests/solver/test_ipopt_solver.cpp
// Integration test: verify that IpoptSolver correctly solves a small
// equality-constrained QP with a known closed-form optimum.
//
// Problem: min x0^2 + x1^2  s.t.  x0 + x1 = 1
// Optimum: x* = (0.5, 0.5),  f* = 0.5
#include <gtest/gtest.h>
#include <memory>
#include "goss/solver/ipopt_solver.hpp"
#include "goss/nlp/nlp_problem.hpp"
#include "goss/ad/cppadcg_backend.hpp"

namespace {

/// Packed functor: output[0] = x0^2 + x1^2 (objective),
///                 output[1] = x0 + x1 - 1  (constraint, bound to [0, 0]).
struct EqQP {
    std::size_t input_size() const { return 2; }
    std::size_t output_size() const { return 2; }
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x) const {
        return {x[0] * x[0] + x[1] * x[1], x[0] + x[1] - T(1)};
    }
};

/// Build the NLPProblem for the equality-constrained QP described above.
goss::nlp::NLPProblem make_eq_qp(const std::string& name) {
    EqQP f;
    auto backend = std::make_unique<goss::ad::CppADCGBackend>(f, f.input_size(), name);
    // variables in [-10, 10]; single equality constraint bound to [0, 0]
    return goss::nlp::NLPProblem(std::move(backend),
        {-10.0, -10.0}, {10.0, 10.0}, {0.0}, {0.0});
}

}  // namespace

TEST(IpoptSolver, SolvesEqualityConstrainedQP) {
    auto problem = make_eq_qp("ipopt_qp");
    goss::solver::IpoptSolver solver;
    auto result = solver.solve(problem, {2.0, -1.0});
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    ASSERT_EQ(result.x.size(), 2u);
    EXPECT_NEAR(result.x[0], 0.5, 1e-5);
    EXPECT_NEAR(result.x[1], 0.5, 1e-5);
    EXPECT_NEAR(result.objective_value, 0.5, 1e-6);
}
