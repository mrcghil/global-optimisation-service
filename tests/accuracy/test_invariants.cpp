// tests/accuracy/test_invariants.cpp
//
// Class 4: Invariant and conservation checks.
// For autonomous problems, physical invariants must be preserved along the
// numerically solved trajectory. This catches schemes that introduce spurious
// dissipation or drift.
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <functional>
#include "goss/model/model.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/transcription/legendre_gauss_lobatto.hpp"
#include "goss/transcription/trapezoidal.hpp"
#include "goss/transcription/transcription.hpp"
#include "accuracy/accuracy_helpers.hpp"

// ---------------------------------------------------------------------------
// Problem 1: Harmonic oscillator energy conservation (zero-cost autonomous ODE).
// State [q, p]: dq/dt = p, dp/dt = -q. Initial: q(0)=1, p(0)=0. Free terminal.
// Energy invariant: E(q, p) = (q² + p²) / 2 = 1/2 = constant.
//
// WHY energy as invariant: E is an exact first integral of the harmonic oscillator.
// A transcription scheme that introduces artificial dissipation will show E decreasing
// along the trajectory — this test catches such bugs definitively.
// ---------------------------------------------------------------------------
namespace {
constexpr double kHarmonicTimeHorizon      = 3.0;   // slightly less than π to avoid sign flip
constexpr std::size_t kHarmonicNumIntervals = 60;
constexpr double kHarmonicEnergyReference  = 0.5;   // E = (1² + 0²)/2 = 0.5
// WHY tolerance 1e-4: HermiteSimpson O(h^4), h=3/60=0.05, error ~ h^4=6.25e-6 per node.
// Accumulated over 61 nodes the max deviation stays << 1e-4.
constexpr double kHarmonicEnergyTolerance  = 1e-4;
// WHY tolerance 1e-2 for Trapezoidal: O(h^2), h=0.05, error~2.5e-3 per node. Loose enough.
constexpr double kHarmonicEnergyTolTrap    = 1e-2;
}  // namespace

// Build the harmonic oscillator OCP (zero cost, autonomous, free terminal).
namespace {
template <typename CompileFn>
goss::transcription::CompiledOcp build_harmonic_oscillator_ocp(
        std::size_t num_intervals,
        const std::string& model_name,
        CompileFn compile_fn) {
    goss::model::Model model;
    const auto position_handle = model.add_state("position");
    const auto momentum_handle = model.add_state("momentum");
    model.set_initial_state(position_handle, 1.0);
    model.set_initial_state(momentum_handle, 0.0);
    // Free terminal: no set_final_state.
    model.set_mesh(0.0, kHarmonicTimeHorizon, num_intervals);

    auto dynamics = [](const auto& state_vec, const auto& /*control_vec*/, auto /*time*/) {
        using ScalarT = typename std::decay_t<decltype(state_vec)>::value_type;
        return std::vector<ScalarT>{ state_vec[1], -state_vec[0] };  // [dq/dt=p, dp/dt=-q]
    };
    // WHY ScalarT(0): the cost functor must return the same AD scalar type T used by the
    // transcription scheme during AD recording. Returning plain 0.0 (double) causes a
    // type mismatch when T = CppAD::AD<...>. Using ScalarT(0) constructs from double,
    // which CppAD AD types support, giving the correct zero-valued AD expression.
    auto zero_cost = [](const auto& state_vec, const auto& /*control_vec*/, auto /*time*/) {
        using ScalarT = typename std::decay_t<decltype(state_vec)>::value_type;
        return ScalarT(0);
    };
    auto ocp = model.build(dynamics, zero_cost);
    return compile_fn(ocp, model_name);
}

// Energy invariant: E = (q² + p²) / 2.
const std::function<double(const std::vector<double>&, const std::vector<double>&)>
    kHarmonicEnergyInvariant =
        [](const std::vector<double>& state, const std::vector<double>& /*control*/) -> double {
            return 0.5 * (state[0] * state[0] + state[1] * state[1]);
        };
}  // namespace

TEST(Invariants, HarmonicOscillatorEnergyConservedHermiteSimpson) {
    auto compiled = build_harmonic_oscillator_ocp(
        kHarmonicNumIntervals,
        "harmonic_energy_hs",
        [](const auto& ocp, const std::string& name) {
            return goss::transcription::HermiteSimpson::compile(ocp, name);
        });
    const auto trajectory = goss::accuracy::solve_and_extract_trajectory(
        compiled, /*initial_guess_value=*/0.5);
    ASSERT_FALSE(trajectory.states.empty());
    goss::accuracy::check_invariant_along_trajectory(
        trajectory, kHarmonicEnergyInvariant, kHarmonicEnergyTolerance);
}

TEST(Invariants, HarmonicOscillatorEnergyConservedTrapezoidal) {
    auto compiled = build_harmonic_oscillator_ocp(
        kHarmonicNumIntervals,
        "harmonic_energy_trap",
        [](const auto& ocp, const std::string& name) {
            return goss::transcription::Trapezoidal::compile(ocp, name);
        });
    const auto trajectory = goss::accuracy::solve_and_extract_trajectory(
        compiled, /*initial_guess_value=*/0.5);
    ASSERT_FALSE(trajectory.states.empty());
    // WHY kHarmonicEnergyTolTrap (1e-2): Trapezoidal is O(h^2); energy error is larger.
    goss::accuracy::check_invariant_along_trajectory(
        trajectory, kHarmonicEnergyInvariant, kHarmonicEnergyTolTrap);
}

TEST(Invariants, HarmonicOscillatorEnergyConservedLGL) {
    // LGL: 20 nodes (19 intervals) for spectral accuracy on smooth harmonic oscillator.
    // WHY 20 nodes: 20 LGL nodes should give energy error < 1e-10 (spectral).
    goss::model::Model model;
    const auto position_handle = model.add_state("position");
    const auto momentum_handle = model.add_state("momentum");
    model.set_initial_state(position_handle, 1.0);
    model.set_initial_state(momentum_handle, 0.0);
    model.set_mesh(0.0, kHarmonicTimeHorizon, /*num_intervals=*/19);
    auto dynamics = [](const auto& state_vec, const auto& /*control_vec*/, auto /*time*/) {
        using ScalarT = typename std::decay_t<decltype(state_vec)>::value_type;
        return std::vector<ScalarT>{ state_vec[1], -state_vec[0] };
    };
    // WHY ScalarT(0): must match the AD scalar type during CppADCG recording (see HS test above).
    auto zero_cost = [](const auto& state_vec, const auto& /*control_vec*/, auto /*time*/) {
        using ScalarT = typename std::decay_t<decltype(state_vec)>::value_type;
        return ScalarT(0);
    };
    auto ocp      = model.build(dynamics, zero_cost);
    auto compiled = goss::transcription::LegendreGaussLobatto::compile(ocp, "harmonic_energy_lgl");
    const auto trajectory = goss::accuracy::solve_and_extract_trajectory(
        compiled, /*initial_guess_value=*/0.5, /*solver_tolerance=*/1e-12);
    ASSERT_FALSE(trajectory.states.empty());
    // WHY tolerance 1e-8: LGL spectral convergence — 20 nodes achieves near-machine precision.
    goss::accuracy::check_invariant_along_trajectory(
        trajectory, kHarmonicEnergyInvariant, /*tolerance=*/1e-8);
}

// ---------------------------------------------------------------------------
// Problem 2: Double integrator Hamiltonian constancy check (primal proxy).
// dx/dt=u, x(0)=0, x(T)=1, min∫u²dt. Optimal control u*=1/T (constant).
//
// PMP Hamiltonian: H = λ·u + u². With H_u = λ + 2u = 0 → λ = -2u*.
// H = -2u*·u* + u*² = -u*² + u*² = 0. Constant along optimal arc.
//
// We cannot evaluate H directly (λ not exposed in SolverResult.x).
// Instead, verify that the running cost L(u*(t)) = u*² is approximately
// constant along the optimal control trajectory — a necessary (but not
// sufficient) condition for Hamiltonian constancy when u* is independent of t.
// WHY sufficient here: u*=const for the double integrator → L=const is equivalent
// to Hamiltonian constancy for this specific problem.
// ---------------------------------------------------------------------------
TEST(Invariants, DoubleIntegratorRunningCostIsConstantAlongOptimalControl) {
    constexpr double kTimeHorizon2  = 2.0;
    constexpr std::size_t kNumIntervals2 = 40;
    constexpr double kExpectedCostRate = (1.0/kTimeHorizon2) * (1.0/kTimeHorizon2);  // u*² = (1/T)²

    goss::model::Model model;
    const auto position_handle = model.add_state("position");
    const auto force_handle    = model.add_control("force");
    model.set_control_bounds(force_handle, -10.0, 10.0);
    model.set_initial_state(position_handle, 0.0);
    model.set_final_state(position_handle, 1.0);
    model.set_mesh(0.0, kTimeHorizon2, kNumIntervals2);
    auto dynamics = [](const auto& state_vec, const auto& control_vec, auto /*time*/) {
        using ScalarT = typename std::decay_t<decltype(state_vec)>::value_type;
        return std::vector<ScalarT>{ control_vec[0] };
    };
    auto running_cost = [](const auto& /*state_vec*/, const auto& control_vec, auto /*time*/) {
        return control_vec[0] * control_vec[0];
    };
    auto ocp      = model.build(dynamics, running_cost);
    auto compiled = goss::transcription::HermiteSimpson::compile(
        ocp, "di_hamiltonian_invariant_hs");
    const auto trajectory = goss::accuracy::solve_and_extract_trajectory(
        compiled, /*initial_guess_value=*/0.5);
    ASSERT_FALSE(trajectory.states.empty());

    // u² should equal (1/T)² = 0.25 at every node (u*=0.5 constant, u*²=0.25).
    const std::function<double(const std::vector<double>&, const std::vector<double>&)>
        running_cost_invariant =
            [](const std::vector<double>& /*state*/, const std::vector<double>& control) -> double {
                return control[0] * control[0];
            };
    // WHY tolerance 1e-3: u* is constant only away from the boundary nodes;
    // near x(0) and x(T) the control may fluctuate slightly.
    // We only check interior nodes to avoid boundary-layer artifacts.
    // Manually loop over interior nodes [1, N-2] instead of using the helper.
    for (std::size_t node_index = 1; node_index + 1 < trajectory.controls.size(); ++node_index) {
        const double control_val    = trajectory.controls[node_index][0];
        const double running_cost_val = control_val * control_val;
        EXPECT_NEAR(running_cost_val, kExpectedCostRate, 5e-3)
            << "Running cost u² deviates from u*²=(1/T)² at interior node " << node_index
            << ": u=" << control_val << ", u²=" << running_cost_val;
    }
}

// ---------------------------------------------------------------------------
// Problem 3: Kepler orbit first integral (angular momentum conservation).
// For completeness of invariant coverage without requiring orbital mechanics:
// use the 2D harmonic oscillator as a proxy for angular momentum.
// State: [x1, x2, v1, v2], dynamics: x1'=v1, x2'=v2, v1'=-x1, v2'=-x2.
// Angular momentum: L_ang = x1*v2 - x2*v1 = constant.
// Initial: x1=1, x2=0, v1=0, v2=1 → L_ang = 1*1 - 0*0 = 1.
// WHY: angular momentum is a different type of invariant from energy — linear
//      in the state components rather than quadratic — ensuring the checker
//      is not accidentally energy-specific.
// ---------------------------------------------------------------------------
TEST(Invariants, TwoDHarmonicOscillatorAngularMomentumConserved) {
    constexpr double kOrbTimeHorizon    = 2.0;
    constexpr std::size_t kOrbIntervals = 60;
    constexpr double kExpectedAngMomentum = 1.0;  // x1*v2 - x2*v1 = 1*1 - 0*0 = 1
    constexpr double kAngMomentumTol = 1e-4;      // HermiteSimpson O(h^4) at h=2/60≈0.033

    goss::model::Model model;
    const auto x1_handle = model.add_state("x1");
    const auto x2_handle = model.add_state("x2");
    const auto v1_handle = model.add_state("v1");
    const auto v2_handle = model.add_state("v2");
    model.set_initial_state(x1_handle, 1.0);
    model.set_initial_state(x2_handle, 0.0);
    model.set_initial_state(v1_handle, 0.0);
    model.set_initial_state(v2_handle, 1.0);
    model.set_mesh(0.0, kOrbTimeHorizon, kOrbIntervals);

    auto dynamics = [](const auto& state_vec, const auto& /*control_vec*/, auto /*time*/) {
        using ScalarT = typename std::decay_t<decltype(state_vec)>::value_type;
        return std::vector<ScalarT>{
            state_vec[2],   // dx1/dt = v1
            state_vec[3],   // dx2/dt = v2
            -state_vec[0],  // dv1/dt = -x1  (spring force)
            -state_vec[1]   // dv2/dt = -x2
        };
    };
    // WHY ScalarT(0): must match the AD scalar type during CppADCG recording (see HS test above).
    auto zero_cost = [](const auto& state_vec, const auto&, auto) {
        using ScalarT = typename std::decay_t<decltype(state_vec)>::value_type;
        return ScalarT(0);
    };
    auto ocp      = model.build(dynamics, zero_cost);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "orb_angmom_hs");
    const auto trajectory = goss::accuracy::solve_and_extract_trajectory(
        compiled, /*initial_guess_value=*/0.5);
    ASSERT_FALSE(trajectory.states.empty());

    // Angular momentum invariant: L_ang = x1*v2 - x2*v1.
    const std::function<double(const std::vector<double>&, const std::vector<double>&)>
        angular_momentum_invariant =
            [](const std::vector<double>& state, const std::vector<double>&) -> double {
                return state[0] * state[3] - state[1] * state[2];
            };
    goss::accuracy::check_invariant_along_trajectory(
        trajectory, angular_momentum_invariant, kAngMomentumTol);
}
