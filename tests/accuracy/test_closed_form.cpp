// tests/accuracy/test_closed_form.cpp
//
// Class 1: Closed-form OCPs.
// These tests assert that the numeric solver matches known analytic optima.
// Problems chosen because they have exact closed-form solutions derivable
// from Pontryagin's minimum principle, not just plausibility checks.
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "goss/model/model.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/transcription/trapezoidal.hpp"
#include "goss/transcription/legendre_gauss_lobatto.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "accuracy/accuracy_helpers.hpp"

// ---------------------------------------------------------------------------
// Problem parameters: double integrator minimum-energy
// dx/dt = u,  x(0)=0,  x(T)=1,  min ∫₀ᵀ u² dt
// Analytic solution: u*(t)=1/T (constant), J*=1/T, x*(t)=t/T.
// Reference: Bryson & Ho "Applied Optimal Control" (1975), Example 1.4-1.
// ---------------------------------------------------------------------------
namespace {
constexpr double kTimeHorizon          = 2.0;   // T = 2 seconds
constexpr double kAnalyticObjective    = 1.0 / kTimeHorizon;  // J* = 1/T = 0.5
constexpr double kAnalyticControl      = 1.0 / kTimeHorizon;  // u* = 0.5 (constant)
constexpr double kObjectiveTolerance   = 1e-4;  // WHY 1e-4: HermiteSimpson O(h^4) with 40 intervals
                                                 //   h=0.05, error O(0.05^4)~6e-6 << 1e-4.
constexpr double kFinalStateTolerance  = 1e-6;  // final state is pinned by a hard variable bound
constexpr double kControlTolerance     = 5e-3;  // control tolerance looser: u* is recovered
                                                 //   indirectly; nodal control fluctuates slightly
                                                 //   around the constant optimum near boundaries.
constexpr std::size_t kNumIntervals    = 40;
}  // namespace

// Shared model factory to avoid duplication across transcription-scheme tests.
namespace {
template <typename CompileFn>
goss::accuracy::SolutionTrajectory build_and_solve_double_integrator_min_energy(
        CompileFn compile_fn, const std::string& model_name) {
    goss::model::Model model;
    const auto position_handle = model.add_state("position");
    const auto force_handle    = model.add_control("force");
    model.set_control_bounds(force_handle, -10.0, 10.0);
    model.set_initial_state(position_handle, 0.0);
    model.set_final_state(position_handle, 1.0);
    model.set_mesh(0.0, kTimeHorizon, kNumIntervals);

    // Dynamics: dx/dt = u (trivial integrator)
    auto dynamics = [](const auto& state_vec, const auto& control_vec, auto /*time*/) {
        using ScalarT = typename std::decay_t<decltype(state_vec)>::value_type;
        return std::vector<ScalarT>{ control_vec[0] };
    };
    // Running cost: L = u² (energy)
    auto running_cost = [](const auto& /*state_vec*/, const auto& control_vec, auto /*time*/) {
        return control_vec[0] * control_vec[0];
    };

    auto ocp      = model.build(dynamics, running_cost);
    auto compiled = compile_fn(ocp, model_name);
    return goss::accuracy::solve_and_extract_trajectory(compiled, /*initial_guess_value=*/0.5);
}
}  // namespace

// --- Test 1a: Hermite-Simpson matches analytic objective ---
// WHY HermiteSimpson as primary: O(h^4) achieves 1e-4 tolerance at 40 intervals,
// making the test informative (not trivially satisfied by any scheme).
TEST(ClosedForm, DoubleIntegratorMinEnergyObjectiveMatchesAnalyticHermiteSimpson) {
    const auto trajectory = build_and_solve_double_integrator_min_energy(
        [](const auto& ocp, const std::string& name) {
            return goss::transcription::HermiteSimpson::compile(ocp, name);
        },
        "di_minenergy_hs");
    ASSERT_FALSE(trajectory.states.empty()) << "Solver failed; see earlier ADD_FAILURE message";
    EXPECT_NEAR(trajectory.objective_value, kAnalyticObjective, kObjectiveTolerance)
        << "HermiteSimpson: J_numeric=" << trajectory.objective_value
        << ", J_analytic=" << kAnalyticObjective;
}

// --- Test 1b: Final state is pinned to 1.0 ---
TEST(ClosedForm, DoubleIntegratorMinEnergyFinalStateExact) {
    const auto trajectory = build_and_solve_double_integrator_min_energy(
        [](const auto& ocp, const std::string& name) {
            return goss::transcription::HermiteSimpson::compile(ocp, name);
        },
        "di_minenergy_finalstate");
    ASSERT_FALSE(trajectory.states.empty());
    const double x_final = trajectory.states.back()[0];
    EXPECT_NEAR(x_final, 1.0, kFinalStateTolerance)
        << "Final state should be pinned to 1.0 by variable bound";
}

// --- Test 1c: Optimal control is nearly constant = 1/T ---
// WHY: verifies the optimal control profile, not just the objective scalar.
TEST(ClosedForm, DoubleIntegratorMinEnergyControlIsNearlyConstant) {
    const auto trajectory = build_and_solve_double_integrator_min_energy(
        [](const auto& ocp, const std::string& name) {
            return goss::transcription::HermiteSimpson::compile(ocp, name);
        },
        "di_minenergy_control");
    ASSERT_FALSE(trajectory.states.empty());
    for (std::size_t node_index = 0; node_index < trajectory.controls.size(); ++node_index) {
        EXPECT_NEAR(trajectory.controls[node_index][0], kAnalyticControl, kControlTolerance)
            << "Control deviates from u*=1/T at node " << node_index;
    }
}

// --- Test 1d: Trapezoidal also recovers J* (looser tolerance: O(h^2)) ---
// WHY: confirms the closed-form yardstick works across schemes.
// Tolerance 1e-2: h=0.05, error O(h^2)~2.5e-3, so 1e-2 is safe with margin.
TEST(ClosedForm, DoubleIntegratorMinEnergyObjectiveMatchesAnalyticTrapezoidal) {
    const auto trajectory = build_and_solve_double_integrator_min_energy(
        [](const auto& ocp, const std::string& name) {
            return goss::transcription::Trapezoidal::compile(ocp, name);
        },
        "di_minenergy_trap");
    ASSERT_FALSE(trajectory.states.empty());
    EXPECT_NEAR(trajectory.objective_value, kAnalyticObjective, 1e-2)
        << "Trapezoidal: J_numeric=" << trajectory.objective_value;
}

// --- Test 1e: LGL (spectral) matches J* to 1e-8 with 20 nodes ---
// WHY: LGL is spectrally accurate on smooth problems; tighter tolerance validates this.
TEST(ClosedForm, DoubleIntegratorMinEnergyObjectiveMatchesAnalyticLGL) {
    goss::model::Model model;
    const auto position_handle = model.add_state("position");
    const auto force_handle    = model.add_control("force");
    model.set_control_bounds(force_handle, -10.0, 10.0);
    model.set_initial_state(position_handle, 0.0);
    model.set_final_state(position_handle, 1.0);
    // LGL requires all initial states pinned; set_initial_state handles this.
    // Use 19 intervals => 20 LGL nodes for spectral accuracy.
    model.set_mesh(0.0, kTimeHorizon, /*num_intervals=*/19);

    auto dynamics = [](const auto& state_vec, const auto& control_vec, auto /*time*/) {
        using ScalarT = typename std::decay_t<decltype(state_vec)>::value_type;
        return std::vector<ScalarT>{ control_vec[0] };
    };
    auto running_cost = [](const auto& /*state_vec*/, const auto& control_vec, auto /*time*/) {
        return control_vec[0] * control_vec[0];
    };

    auto ocp      = model.build(dynamics, running_cost);
    auto compiled = goss::transcription::LegendreGaussLobatto::compile(ocp, "di_minenergy_lgl");
    // WHY initial guess 0.5: mid-range for both x ∈ [0,1] and u* ≈ 0.5.
    const auto trajectory = goss::accuracy::solve_and_extract_trajectory(
        compiled, /*initial_guess_value=*/0.5, /*solver_tolerance=*/1e-11);
    ASSERT_FALSE(trajectory.states.empty());
    EXPECT_NEAR(trajectory.objective_value, kAnalyticObjective, 1e-8)
        << "LGL (20 nodes): J_numeric=" << trajectory.objective_value;
}

// ---------------------------------------------------------------------------
// Problem 2: Double integrator (2nd order) minimum-energy
// State: [position p, velocity v], dynamics: dp/dt=v, dv/dt=u
// Boundary: p(0)=0, v(0)=0, p(T)=1, v(T)=0.  min ∫₀ᵀ u² dt
// Analytic solution (Bryson & Ho, §2.3):
//   u*(t) = 6/T² - 12t/T³  (linear ramp from 6/T² to -6/T²)
//   J* = 12/T³
// For T=1: u*(t)=6-12t, J*=12.
// Reference: Bryson & Ho (1975), problem 2.3-1.
// ---------------------------------------------------------------------------
namespace {
constexpr double kT2 = 1.0;  // time horizon for 2nd-order problem
constexpr double kJ2 = 12.0 / (kT2 * kT2 * kT2);  // J* = 12/T^3 = 12
constexpr std::size_t kN2 = 40;  // mesh intervals
}  // namespace

TEST(ClosedForm, SecondOrderDoubleIntegratorMinEnergyMatchesAnalytic) {
    goss::model::Model model;
    const auto position_handle = model.add_state("position");
    const auto velocity_handle = model.add_state("velocity");
    const auto thrust_handle   = model.add_control("thrust");
    model.set_control_bounds(thrust_handle, -50.0, 50.0);
    // Pinned boundary conditions: p(0)=0, v(0)=0, p(T)=1, v(T)=0.
    model.set_initial_state(position_handle, 0.0);
    model.set_initial_state(velocity_handle, 0.0);
    model.set_final_state(position_handle, 1.0);
    model.set_final_state(velocity_handle, 0.0);
    model.set_mesh(0.0, kT2, kN2);

    // Dynamics: [dp/dt, dv/dt] = [v, u]
    auto dynamics = [](const auto& state_vec, const auto& control_vec, auto /*time*/) {
        using ScalarT = typename std::decay_t<decltype(state_vec)>::value_type;
        return std::vector<ScalarT>{ state_vec[1], control_vec[0] };
    };
    auto running_cost = [](const auto& /*state_vec*/, const auto& control_vec, auto /*time*/) {
        return control_vec[0] * control_vec[0];
    };

    auto ocp      = model.build(dynamics, running_cost);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "di2_minenergy_hs");
    // WHY initial guess 0.5: non-zero to avoid degenerate starting point; control
    // of magnitude ~6 is feasible and the solver converges easily from 0.5.
    const auto trajectory = goss::accuracy::solve_and_extract_trajectory(
        compiled, /*initial_guess_value=*/0.5);
    ASSERT_FALSE(trajectory.states.empty());
    // WHY tolerance 0.1: J*=12 is large; relative error ~1e-3 corresponds to 0.012.
    // 0.1 is safe with margin for the O(h^4) HermiteSimpson at 40 intervals.
    EXPECT_NEAR(trajectory.objective_value, kJ2, 0.1)
        << "2nd-order double integrator: J_numeric=" << trajectory.objective_value
        << ", J_analytic=" << kJ2;
    // Final boundary conditions must be satisfied.
    EXPECT_NEAR(trajectory.states.back()[0], 1.0, 1e-5) << "p(T) must equal 1";
    EXPECT_NEAR(trajectory.states.back()[1], 0.0, 1e-5) << "v(T) must equal 0";
}

// ---------------------------------------------------------------------------
// Smoke test (retained from Task 1): confirms the scaffold compiles and
// solve_and_extract_trajectory returns sane node counts.
// ---------------------------------------------------------------------------
TEST(ClosedFormSmoke, TrajectoryExtractorReturnsCorrectNodeCount) {
    // Simplest possible OCP: dx/dt = u, x(0)=0, x(1)=1, min integral(u^2).
    // Used only to confirm the scaffold compiles and the helper returns sane data.
    const std::size_t num_intervals = 10;
    goss::model::Model model;
    const auto position_handle = model.add_state("position");
    const auto force_handle    = model.add_control("force");
    model.set_initial_state(position_handle, 0.0);
    model.set_final_state(position_handle, 1.0);
    model.set_mesh(0.0, 1.0, num_intervals);

    auto dynamics = [](const auto& state_vec, const auto& control_vec, auto /*time*/) {
        using ScalarT = typename std::decay_t<decltype(state_vec)>::value_type;
        return std::vector<ScalarT>{ control_vec[0] };
    };
    auto running_cost = [](const auto& /*state_vec*/, const auto& control_vec, auto /*time*/) {
        return control_vec[0] * control_vec[0];
    };

    auto ocp      = model.build(dynamics, running_cost);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "smoke_scaffold");
    const goss::accuracy::SolutionTrajectory trajectory =
        goss::accuracy::solve_and_extract_trajectory(compiled, /*initial_guess_value=*/0.5);

    // num_nodes = num_intervals + 1
    EXPECT_EQ(trajectory.states.size(), num_intervals + 1);
    EXPECT_EQ(trajectory.controls.size(), num_intervals + 1);
    EXPECT_EQ(trajectory.times.size(), num_intervals + 1);
}
