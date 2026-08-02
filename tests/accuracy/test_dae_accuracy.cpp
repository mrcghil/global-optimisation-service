// tests/accuracy/test_dae_accuracy.cpp
//
// End-to-end accuracy test for DAE Flavor 2 (algebraic-variable) derived quantities.
//
// Problem: semi-explicit index-1 DAE on [0, 0.5].
//   dx/dt = -x + z_alg
//   g(x, z_alg) = z_alg - 3*x = 0   =>  z_alg = 3*x
//
// Substituted ODE (for the dynamics functor, since alg_vars are not threaded
// into dynamics in v1): dx/dt = 2*x
//
// Analytic solution:
//   x(t)     = exp(2*t)
//   z_alg(t) = 3*exp(2*t)
//
// Assertions:
//   1. x(tf) is within 1e-6 of exp(1.0).
//   2. z_alg(tf) is within 1e-6 of 3*exp(1.0).
//   3. At every node k: |z_alg_k / x_k - 3.0| < 1e-8
//      (algebraic constraint g == 0 is satisfied to tight tolerance at every node).
//
// Pre-condition: accuracy validation suite (goss_accuracy_tests) must be merged before
// this file is compiled. It provides accuracy::solve_and_extract_trajectory.
#include <gtest/gtest.h>
#include <cmath>
#include <cstddef>
#include <vector>
#include "goss/model/component.hpp"
#include "goss/model/composed_model.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "accuracy/accuracy_helpers.hpp"

namespace {

// The algebraic residual g(x, u, z_alg, t) = z_alg[0] - 3.0 * x[0].
// Enforces z_alg[0] = 3 * x[0] at every collocation node when driven to zero.
struct ThreeTimesXResidual {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x,
                               const std::vector<T>& /*u*/,
                               const std::vector<T>& alg_vars,
                               T /*t*/) const {
        return { alg_vars[0] - T(3.0) * x[0] };
    }
};

}  // namespace

TEST(DaeAccuracy, SemiExplicitIndex1DaeTracksAnalyticTrajectory) {
    // ---- Problem parameters ----
    constexpr double x_initial        = 1.0;
    constexpr double time_horizon     = 0.5;
    constexpr double algebraic_coeff  = 3.0;  // z_alg = algebraic_coeff * x
    constexpr std::size_t num_intervals = 40;

    // ---- Analytic reference values ----
    // x(t) = exp(2*t), z_alg(t) = 3*exp(2*t)
    const double x_final_reference      = std::exp(2.0 * time_horizon);  // exp(1.0)
    const double z_alg_final_reference  = algebraic_coeff * x_final_reference;

    // ---- Model assembly ----
    goss::model::ComposedModel composed;

    goss::model::Component dae_component("dae_system");
    const auto state_handle = dae_component.add_state("x");
    dae_component.set_initial_state(state_handle, x_initial);
    dae_component.set_state_bounds(state_handle, -1e19, 1e19);

    // Algebraic variable z_alg with bounds [-1e19, 1e19] (unconstrained in box sense;
    // the residual constraint z_alg = 3*x will fully determine its value).
    auto algebraic_validation_fn = [algebraic_coeff](
            const std::vector<double>& x,
            const std::vector<double>& /*u*/,
            const std::vector<double>& alg_vars,
            double /*t*/) -> double {
        return alg_vars[0] - algebraic_coeff * x[0];
    };
    dae_component.add_algebraic("z_alg", algebraic_validation_fn, -1e19, 1e19);

    // Dynamics: dx/dt = 2*x (substituted form; z_alg = 3*x enforced separately by residual).
    // Note: the dynamics functor does not receive alg_vars in v1; the caller must write
    // the dynamics in the substituted form for a semi-explicit DAE.
    dae_component.set_dynamics(
        [](const std::vector<double>& x,
           const std::vector<double>& /*u*/,
           const std::vector<double>& /*deriveds*/,
           double /*t*/) -> std::vector<double> {
            // dx/dt = (c-1)*x where c=3 => 2*x
            return { 2.0 * x[0] };
        });
    composed.add_component(std::move(dae_component));
    composed.set_mesh(0.0, time_horizon, num_intervals);

    // ---- AD-safe generic lambdas for build_with_algebraic ----

    // Algebraic residuals functor (AD-safe, concrete type ThreeTimesXResidual).
    // This is the SAME computation as the validation lambda above but as a generic functor
    // (template operator()) so it can be called under CppAD AD scalar types during recording.
    auto algebraic_residuals = ThreeTimesXResidual{};

    // Component dynamics (generic lambda, AD-safe).
    // dx/dt = 2*x (substituted DAE — z_alg = 3*x enforced by algebraic constraint).
    auto dae_dynamics = [](const auto& x, const auto& /*u*/, const auto& /*deriveds*/, auto /*t*/) {
        using ScalarT = typename std::decay_t<decltype(x)>::value_type;
        return std::vector<ScalarT>{ ScalarT(2.0) * x[0] };
    };

    // Zero cost (feasibility problem — unique trajectory determined by IC + DAE).
    // HAZARD A fix: deduce ScalarT from x (not from a captured double variable)
    // so this lambda compiles and records correctly under CppAD AD scalar types.
    auto zero_cost = [](const auto& x, const auto& /*u*/,
                        const auto& /*deriveds*/, auto /*t*/) {
        using ScalarT = typename std::decay_t<decltype(x)>::value_type;
        return ScalarT(0);
    };

    // ---- Solve ----
    auto ocp = composed.build_with_algebraic(algebraic_residuals, dae_dynamics, zero_cost);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "dae_accuracy_test");

    // Initial guess: flat z0 = 1.0 for all variables (states and algebraic variables).
    // IPOPT will satisfy the algebraic constraints from this guess.
    std::vector<double> initial_guess(compiled.problem->num_variables(), 1.0);

    goss::solver::IpoptSolver solver;
    solver.set_tolerance(1e-10);  // tight solver tolerance so algebraic constraint error is below 1e-8
    const auto result = solver.solve(*compiled.problem, initial_guess);

    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success)
        << "IPOPT failed to converge on the semi-explicit index-1 DAE OCP";

    const auto& layout = compiled.layout;
    const std::size_t nn = layout.num_nodes();

    // ---- Assertion 1: x(tf) matches analytic reference ----
    const double x_final_computed =
        result.x[layout.state_index(nn - 1, 0)];
    EXPECT_NEAR(x_final_computed, x_final_reference, 1e-6)
        << "x(tf) = " << x_final_computed
        << " but analytic reference is exp(1.0) = " << x_final_reference;

    // ---- Assertion 2: z_alg(tf) matches analytic reference ----
    const double z_alg_final_computed =
        result.x[layout.algebraic_index(nn - 1, 0)];
    EXPECT_NEAR(z_alg_final_computed, z_alg_final_reference, 1e-6)
        << "z_alg(tf) = " << z_alg_final_computed
        << " but analytic reference is 3*exp(1.0) = " << z_alg_final_reference;

    // ---- Assertion 3: algebraic constraint z_alg = 3*x satisfied at every node ----
    // At every collocation node k, the residual g(x_k, z_alg_k) = z_alg_k - 3*x_k
    // must be zero to within 1e-8 (solver tolerance is 1e-10; discretization error
    // is O(h^4) for HermiteSimpson with 40 intervals over [0,0.5], well below 1e-8).
    double max_per_node_residual = 0.0;
    for (std::size_t k = 0; k < nn; ++k) {
        const double x_k     = result.x[layout.state_index(k, 0)];
        const double z_alg_k = result.x[layout.algebraic_index(k, 0)];
        const double residual_k = z_alg_k - algebraic_coeff * x_k;
        if (std::abs(residual_k) > max_per_node_residual) {
            max_per_node_residual = std::abs(residual_k);
        }
        EXPECT_NEAR(residual_k, 0.0, 1e-8)
            << "Algebraic constraint g = z_alg - 3*x violated at node " << k
            << ": z_alg=" << z_alg_k << ", x=" << x_k
            << ", residual=" << residual_k;
    }

    // ---- Assertion 4: x is non-decreasing (monotone for dx/dt = 2*x with x(0)>0) ----
    for (std::size_t k = 1; k < nn; ++k) {
        EXPECT_GE(result.x[layout.state_index(k, 0)],
                  result.x[layout.state_index(k - 1, 0)] - 1e-6)
            << "x should be non-decreasing (dx/dt=2x, x(0)=1 > 0) at node " << k;
    }

    // Emit calibration evidence to stdout so the test report can record it.
    std::cout << "[DaeAccuracy] x(tf)=" << x_final_computed
              << " ref=" << x_final_reference
              << " err=" << std::abs(x_final_computed - x_final_reference) << "\n";
    std::cout << "[DaeAccuracy] z_alg(tf)=" << z_alg_final_computed
              << " ref=" << z_alg_final_reference
              << " err=" << std::abs(z_alg_final_computed - z_alg_final_reference) << "\n";
    std::cout << "[DaeAccuracy] max per-node |z_alg-3x|=" << max_per_node_residual << "\n";
}
