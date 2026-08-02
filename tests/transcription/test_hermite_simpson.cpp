// tests/transcription/test_hermite_simpson.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/transcription/mesh.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "transcription/ocp_fixtures.hpp"

TEST(HermiteSimpson, SolvesExponentialDecay) {
    const double x0 = 1.0, tf = 1.0;
    auto ocp = goss::transcription::test::make_exponential_decay(x0, tf, 20);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "hs_expdecay");
    goss::solver::IpoptSolver solver;
    std::vector<double> z0(compiled.problem->num_variables(), x0);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    std::size_t last = compiled.layout.num_nodes() - 1;
    double x_final = result.x[compiled.layout.state_index(last, 0)];
    EXPECT_NEAR(x_final, goss::transcription::test::exp_decay_solution(x0, tf), 1e-5);
}

TEST(HermiteSimpson, SolvesHarmonicOscillator) {
    // x0(0)=1, x1(0)=0 -> x0(t)=cos t. Check x0(tf).
    const double tf = 1.0;
    auto ocp = goss::transcription::test::make_harmonic(1.0, 0.0, tf, 20);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "hs_harmonic");
    goss::solver::IpoptSolver solver;
    std::vector<double> z0(compiled.problem->num_variables(), 0.5);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    std::size_t last = compiled.layout.num_nodes() - 1;
    double x0_final = result.x[compiled.layout.state_index(last, 0)];
    EXPECT_NEAR(x0_final, goss::transcription::test::harmonic_x0_solution(1.0, 0.0, tf), 1e-4);
}

TEST(HermiteSimpson, PinsInitialState) {
    auto ocp = goss::transcription::test::make_exponential_decay(2.0, 1.0, 20);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "hs_pin");
    std::size_t idx = compiled.layout.state_index(0, 0);
    EXPECT_DOUBLE_EQ(compiled.problem->variable_lower_bounds()[idx], 2.0);
    EXPECT_DOUBLE_EQ(compiled.problem->variable_upper_bounds()[idx], 2.0);
}

TEST(HermiteSimpson, FinalStateFreeWhenNotFixed) {
    auto ocp = goss::transcription::test::make_exponential_decay(1.0, 1.0, 20);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "hs_freefinal");
    std::size_t last = compiled.layout.num_nodes() - 1;
    std::size_t idx = compiled.layout.state_index(last, 0);
    EXPECT_LT(compiled.problem->variable_lower_bounds()[idx],
              compiled.problem->variable_upper_bounds()[idx]);
}

namespace {
double hs_max_error(std::size_t intervals) {
    const double x0 = 1.0, tf = 1.0;
    auto ocp = goss::transcription::test::make_exponential_decay(x0, tf, intervals);
    auto compiled = goss::transcription::HermiteSimpson::compile(
        ocp, "hs_conv_" + std::to_string(intervals));
    goss::solver::IpoptSolver solver;
    solver.set_tolerance(1e-11);  // discretization error must dominate, not solver tol
    std::vector<double> z0(compiled.problem->num_variables(), x0);
    auto result = solver.solve(*compiled.problem, z0);
    if (result.status != goss::solver::SolverStatus::Success) return 1e9;
    const auto& layout = compiled.layout;
    const double h = tf / static_cast<double>(intervals);
    double max_err = 0.0;
    for (std::size_t k = 0; k < layout.num_nodes(); ++k) {
        double xk = result.x[layout.state_index(k, 0)];
        double exact = goss::transcription::test::exp_decay_solution(x0, k * h);
        max_err = std::max(max_err, std::abs(xk - exact));
    }
    return max_err;
}
}  // namespace

TEST(HermiteSimpson, ConvergesAtFourthOrder) {
    // Use coarse meshes so O(h^4) error stays above solver tolerance floor.
    double e1 = hs_max_error(5);
    double e2 = hs_max_error(10);
    double e3 = hs_max_error(20);
    ASSERT_LT(e2, e1);
    ASSERT_LT(e3, e2);
    double order1 = std::log(e1 / e2) / std::log(2.0);
    double order2 = std::log(e2 / e3) / std::log(2.0);
    // HS is O(h^4). Allow slack (3.5) for solver-tolerance contamination at fine meshes.
    EXPECT_GE(order1, 3.5) << "Hermite-Simpson should be ~4th order";
    EXPECT_GE(order2, 3.5) << "Hermite-Simpson should be ~4th order";
}

TEST(HermiteSimpson, NonUniformMeshSolvesExponentialDecay) {
    const double x0 = 1.0;
    auto ocp = goss::transcription::test::make_exponential_decay(x0, 1.0, 1);
    goss::transcription::NonUniformMesh nu_mesh;
    nu_mesh.node_times = {0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0};
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, nu_mesh, "hs_nu_expdecay");
    goss::solver::IpoptSolver solver;
    solver.set_tolerance(1e-10);
    std::vector<double> z0(compiled.problem->num_variables(), x0);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    std::size_t last = compiled.layout.num_nodes() - 1;
    double x_final = result.x[compiled.layout.state_index(last, 0)];
    EXPECT_NEAR(x_final, goss::transcription::test::exp_decay_solution(x0, 1.0), 1e-4);
}

TEST(HermiteSimpson, UniformOverloadStillPassesAfterRefactor) {
    const double x0 = 1.0;
    auto ocp = goss::transcription::test::make_exponential_decay(x0, 1.0, 20);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "hs_uniform_regression");
    goss::solver::IpoptSolver solver;
    solver.set_tolerance(1e-11);
    std::vector<double> z0(compiled.problem->num_variables(), x0);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    std::size_t last = compiled.layout.num_nodes() - 1;
    double x_final = result.x[compiled.layout.state_index(last, 0)];
    EXPECT_NEAR(x_final, goss::transcription::test::exp_decay_solution(x0, 1.0), 1e-5);
}

namespace {

/// Minimal OcpProblem with 1 state, 0 controls, 1 algebraic variable.
/// Dynamics: dx/dt = 0 (trivial, no movement).
/// Algebraic residual: g(x, u, z_alg, t) = z_alg[0] - 2.0*x[0]
/// This enforces z_alg[0] = 2*x[0] at every node.
/// With x(0) = 3.0 fixed and dx/dt = 0, the solution is x(t) = 3.0 everywhere,
/// so z_alg(t) = 6.0 everywhere.
struct TrivialAlgebraicDynamics {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& /*x*/,
                               const std::vector<T>& /*u*/,
                               T /*t*/) const {
        return { T(0.0) };  // dx/dt = 0
    }
};
struct ZeroCostAlg {
    template <typename T>
    T operator()(const std::vector<T>& /*x*/,
                 const std::vector<T>& /*u*/,
                 T /*t*/) const {
        return T(0);
    }
};
struct TwiceXResidual {
    // g(x, u, z_alg, t) = z_alg[0] - 2*x[0]
    // Enforces z_alg[0] = 2*x[0] when driven to zero by the solver.
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x,
                               const std::vector<T>& /*u*/,
                               const std::vector<T>& alg_vars,
                               T /*t*/) const {
        return { alg_vars[0] - T(2.0) * x[0] };
    }
};

goss::transcription::OcpProblem<TrivialAlgebraicDynamics, ZeroCostAlg, TwiceXResidual>
make_twice_x_algebraic_ocp(double x0_val, std::size_t num_intervals) {
    goss::transcription::OcpProblem<TrivialAlgebraicDynamics, ZeroCostAlg, TwiceXResidual> ocp;
    ocp.num_states = 1;
    ocp.num_controls = 0;
    ocp.dynamics = TrivialAlgebraicDynamics{};
    ocp.cost = ZeroCostAlg{};
    ocp.mesh = goss::transcription::Mesh{0.0, 1.0, num_intervals};
    ocp.state_lower = { -1e19 };
    ocp.state_upper = { 1e19 };
    ocp.control_lower = {};
    ocp.control_upper = {};
    ocp.initial_state = { x0_val };
    ocp.initial_state_fixed = { 1.0 };
    ocp.final_state = { 0.0 };
    ocp.final_state_fixed = { 0.0 };
    ocp.num_algebraic = 1;
    ocp.algebraic_residuals_functor = TwiceXResidual{};
    ocp.algebraic_lower_bounds = { -1e19 };
    ocp.algebraic_upper_bounds = { 1e19 };
    return ocp;
}

}  // namespace

TEST(HermiteSimpsonAlgebraic, NumConstraintsIncludesAlgebraicResiduals) {
    // 5 intervals, 6 nodes (nn=6).
    // Defect constraints: ni*ns = 5*1 = 5.
    // Algebraic residual constraints: nn*na = 6*1 = 6.
    // Total constraints = 5 + 6 = 11.
    const std::size_t num_intervals = 5;
    auto ocp = make_twice_x_algebraic_ocp(3.0, num_intervals);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "hs_alg_count");
    const std::size_t nn = num_intervals + 1;
    const std::size_t expected_defects = num_intervals * 1;  // ni * ns
    const std::size_t expected_alg_residuals = nn * 1;       // nn * na
    EXPECT_EQ(compiled.problem->num_constraints(),
              expected_defects + expected_alg_residuals);
}

TEST(HermiteSimpsonAlgebraic, AlgebraicConstraintBoundsAreEqualityZero) {
    const std::size_t num_intervals = 4;
    auto ocp = make_twice_x_algebraic_ocp(3.0, num_intervals);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "hs_alg_bounds");
    const auto& gl = compiled.problem->constraint_lower_bounds();
    const auto& gu = compiled.problem->constraint_upper_bounds();
    // All constraints (defects + algebraic) must be equality [0, 0].
    for (std::size_t i = 0; i < gl.size(); ++i) {
        EXPECT_DOUBLE_EQ(gl[i], 0.0) << "constraint lower bound at index " << i;
        EXPECT_DOUBLE_EQ(gu[i], 0.0) << "constraint upper bound at index " << i;
    }
}

TEST(HermiteSimpsonAlgebraic, LayoutHasCorrectAlgebraicStride) {
    const std::size_t num_intervals = 3;
    auto ocp = make_twice_x_algebraic_ocp(3.0, num_intervals);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "hs_alg_stride");
    // ns=1, nc=0, na=1, nn=4. stride=2. total_variables=8.
    EXPECT_EQ(compiled.layout.num_algebraic(), 1u);
    EXPECT_EQ(compiled.layout.variables_per_node(), 2u);
    EXPECT_EQ(compiled.layout.total_variables(), 8u);
    // Verify algebraic_index positioning.
    EXPECT_EQ(compiled.layout.algebraic_index(0, 0), 1u);  // node 0: x=0, alg=1
    EXPECT_EQ(compiled.layout.algebraic_index(1, 0), 3u);  // node 1: x=2, alg=3
}

TEST(HermiteSimpsonAlgebraic, AlgebraicBoundsSatisfied) {
    // Bounds test: algebraic variable bounds [-1e19, 1e19] should be set on alg slots.
    const std::size_t num_intervals = 3;
    auto ocp = make_twice_x_algebraic_ocp(3.0, num_intervals);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "hs_alg_varbounds");
    const auto& zl = compiled.problem->variable_lower_bounds();
    const auto& zu = compiled.problem->variable_upper_bounds();
    const std::size_t nn = num_intervals + 1;
    for (std::size_t k = 0; k < nn; ++k) {
        const std::size_t alg_idx = compiled.layout.algebraic_index(k, 0);
        EXPECT_DOUBLE_EQ(zl[alg_idx], -1e19);
        EXPECT_DOUBLE_EQ(zu[alg_idx],  1e19);
    }
}

TEST(HermiteSimpsonAlgebraic, ZeroAlgebraicPreservesExistingBehavior) {
    // Passing a two-template-param OcpProblem must still work exactly as before.
    const double x0 = 1.0, tf = 1.0;
    auto ocp = goss::transcription::test::make_exponential_decay(x0, tf, 10);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "hs_alg_compat");
    EXPECT_EQ(compiled.layout.num_algebraic(), 0u);
    // num_constraints must still be ni * ns = 10 * 1 = 10.
    EXPECT_EQ(compiled.problem->num_constraints(), 10u);
}
