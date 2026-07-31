// tests/solver/test_ipopt_solver.cpp
// Integration tests: verify that IpoptSolver correctly solves small
// constrained NLPs with known optima.
//
// Problem EqQP: min x0^2 + x1^2  s.t.  x0 + x1 = 1
// Optimum: x* = (0.5, 0.5),  f* = 0.5
//
// HS71, HS28, HS35 — Hock-Schittkowski benchmark problems with known optima.
#include <gtest/gtest.h>
#include <memory>
#include "goss/solver/ipopt_solver.hpp"
#include "goss/nlp/nlp_problem.hpp"
#include "goss/ad/cppadcg_backend.hpp"
#include "solver/hs_fixtures.hpp"

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

// ---------------------------------------------------------------------------
// Hock-Schittkowski benchmark problems
// ---------------------------------------------------------------------------

/// HS71: canonical IPOPT example — mixed equality/inequality NLP.
/// x* ≈ (1.0, 4.743, 3.821, 1.379),  f* ≈ 17.0140173
TEST(IpoptSolver, SolvesHS71) {
    goss::solver::hs::HS71 f;
    auto backend = std::make_unique<goss::ad::CppADCGBackend>(
        f, f.input_size(), "ipopt_hs71");
    // Variable bounds: 1 <= xi <= 5
    // Inequality g >= 25 -> constraint bound [25, kInf]
    // Equality   h == 40 -> constraint bound [40, 40]
    goss::nlp::NLPProblem problem(std::move(backend),
        {1.0, 1.0, 1.0, 1.0}, {5.0, 5.0, 5.0, 5.0},
        {25.0, 40.0}, {goss::solver::hs::kInf, 40.0});
    goss::solver::IpoptSolver solver;
    auto result = solver.solve(problem, {1.0, 5.0, 5.0, 1.0});
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    EXPECT_NEAR(result.objective_value, 17.0140173, 1e-4);
    EXPECT_NEAR(result.x[0], 1.0,   1e-3);
    EXPECT_NEAR(result.x[1], 4.743, 1e-3);
    EXPECT_NEAR(result.x[2], 3.821, 1e-3);
    EXPECT_NEAR(result.x[3], 1.379, 1e-3);
}

/// HS28: equality-only QP.
/// x* = (0.5, -0.5, 0.5),  f* = 0
TEST(IpoptSolver, SolvesHS28) {
    goss::solver::hs::HS28 f;
    auto backend = std::make_unique<goss::ad::CppADCGBackend>(
        f, f.input_size(), "ipopt_hs28");
    // Variables are free: [-kInf, kInf]
    // Equality residual h == 0 -> constraint bound [0, 0]
    const double inf = goss::solver::hs::kInf;
    goss::nlp::NLPProblem problem(std::move(backend),
        {-inf, -inf, -inf}, {inf, inf, inf}, {0.0}, {0.0});
    goss::solver::IpoptSolver solver;
    auto result = solver.solve(problem, {-4.0, 1.0, 1.0});
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    EXPECT_NEAR(result.objective_value, 0.0, 1e-6);
}

/// HS35: inequality-only QP.
/// f* = 1/9 ≈ 0.11111
TEST(IpoptSolver, SolvesHS35) {
    goss::solver::hs::HS35 f;
    auto backend = std::make_unique<goss::ad::CppADCGBackend>(
        f, f.input_size(), "ipopt_hs35");
    // Variable bounds: xi >= 0  -> [0, kInf]
    // Inequality g <= 3 -> constraint bound [-kInf, 3]
    const double inf = goss::solver::hs::kInf;
    goss::nlp::NLPProblem problem(std::move(backend),
        {0.0, 0.0, 0.0}, {inf, inf, inf}, {-inf}, {3.0});
    goss::solver::IpoptSolver solver;
    auto result = solver.solve(problem, {0.5, 0.5, 0.5});
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    EXPECT_NEAR(result.objective_value, 1.0 / 9.0, 1e-5);
}
