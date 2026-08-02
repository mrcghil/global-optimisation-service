// tests/accuracy/test_convergence_order.cpp
//
// Class 3: Convergence-order validation.
// For each transcription scheme, solve the same smooth OCP at a sequence of
// mesh sizes and verify the empirical error order matches theory:
//   Trapezoidal:     O(h²)  → slope ≈ 2
//   Hermite-Simpson: O(h⁴)  → slope ≈ 4
//   LGL:             spectral (exponential in N)
//
// Reference problem: 1D LQR double integrator, dx/dt=u, x(0)=0, x(T)=1,
//   min ∫₀ᵀ (x² + u²) dt.
// Analytic solution: x*(t) = sinh(t)/sinh(T), u*(t) = cosh(t)/sinh(T).
// J* = sinh(2T)/(2·sinh²(T)).
// WHY this problem: smooth analytic solution (hyperbolic — infinitely differentiable)
// with non-trivial higher derivatives. The running cost x²+u² ensures the
// discretization truncation error is NONZERO at every finite mesh, which is
// required for convergence-order testing.
//
// NOTE on problem selection: the simpler choice min∫u²dt has optimal control
// u*(t)=1/T (constant) and optimal state x*(t)=t/T (linear). Both are in the
// null space of Trapezoidal/HS truncation errors (linear states are integrated
// exactly by any consistent method), so that problem gives zero position error
// at all mesh sizes — making slope estimation meaningless. The x²+u² cost
// avoids this degenerate case while keeping the same model structure.
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <string>
#include "goss/model/model.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/transcription/trapezoidal.hpp"
#include "goss/transcription/legendre_gauss_lobatto.hpp"
#include "goss/transcription/transcription.hpp"
#include "accuracy/accuracy_helpers.hpp"

namespace {
// Time horizon for all convergence tests.
constexpr double kTimeHorizon = 1.0;

// Analytic optimal state: x*(t) = sinh(t)/sinh(T).
// For T=1: x*(t) = sinh(t)/sinh(1).
// WHY: analytic solution to dx/dt=u, x(0)=0, x(T)=1, min∫(x²+u²)dt.
//   From the PMP: u* = -λ/2, λ' = -2x*, x* = u* = -λ'/2.
//   => x*'' = x* (hyperbolic oscillator BVP). With x*(0)=0: x*(t) = A·sinh(t).
//   With x*(T)=1: A=1/sinh(T). So x*(t) = sinh(t)/sinh(T).
inline double analytic_position(double time, double T) {
    return std::sinh(time) / std::sinh(T);
}

// Analytic optimal objective: J* = sinh(2T)/(2·sinh²(T)).
// WHY: ∫₀ᵀ (x*²+u*²) dt = ∫₀ᵀ (sinh²t+cosh²t)/sinh²T dt
//   = ∫₀ᵀ cosh(2t)/sinh²T dt = sinh(2T)/(2·sinh²T).
inline double analytic_objective(double T) {
    return std::sinh(2.0 * T) / (2.0 * std::sinh(T) * std::sinh(T));
}

// Build and compile the LQR double integrator OCP at a given resolution.
// Returns a CompiledOcp ready for solve_and_extract_trajectory.
// model_name must be unique per call (CppADCG shared-library naming).
template <typename CompileFn>
goss::transcription::CompiledOcp build_double_integrator_ocp(
        std::size_t num_intervals,
        const std::string& model_name,
        CompileFn compile_fn) {
    goss::model::Model model;
    const auto position_handle = model.add_state("position");
    const auto force_handle    = model.add_control("force");
    model.set_control_bounds(force_handle, -10.0, 10.0);
    model.set_initial_state(position_handle, 0.0);
    model.set_final_state(position_handle, 1.0);
    model.set_mesh(0.0, kTimeHorizon, num_intervals);
    auto dynamics = [](const auto& state_vec, const auto& control_vec, auto /*time*/) {
        using ScalarT = typename std::decay_t<decltype(state_vec)>::value_type;
        return std::vector<ScalarT>{ control_vec[0] };
    };
    // WHY x²+u² cost (not just u²): with only u², the optimal control u*(t)=1/T is
    // constant and the state x*(t)=t/T is linear. Both are in the null space of
    // Trapezoidal and Hermite-Simpson truncation errors, giving zero discretization
    // error at every mesh size. With x²+u², the optimal trajectory x*(t)=sinh(t)/sinh(T)
    // has nonzero higher-order derivatives at every order, producing the expected
    // O(h²) and O(h⁴) discretization errors.
    auto running_cost = [](const auto& state_vec, const auto& control_vec, auto /*time*/) {
        return state_vec[0] * state_vec[0] + control_vec[0] * control_vec[0];
    };
    auto ocp = model.build(dynamics, running_cost);
    return compile_fn(ocp, model_name);
}

// Error metric: max nodal absolute error in position vs analytic x*(t_k)=sinh(t_k)/sinh(T).
// For uniform mesh with T=1: t_k = k * h = k / num_intervals.
// WHY: position error converges at the scheme's spatial order (O(h²) or O(h⁴)) because
// x*(t) has nonzero derivatives of all orders (hyperbolic sine is infinitely smooth).
double position_max_error(const goss::accuracy::SolutionTrajectory& trajectory,
                          std::size_t num_intervals) {
    const std::size_t num_nodes = trajectory.states.size();
    double max_error = 0.0;
    for (std::size_t node_index = 0; node_index < num_nodes; ++node_index) {
        // Uniform node time: t_k = k * h = k / num_intervals (T=1).
        const double time_at_node =
            static_cast<double>(node_index) / static_cast<double>(num_intervals);
        const double expected_position = analytic_position(time_at_node, kTimeHorizon);
        const double error = std::abs(trajectory.states[node_index][0] - expected_position);
        max_error = std::max(max_error, error);
    }
    return max_error;
}

}  // namespace

// --- Trapezoidal: empirical slope must be ≥ 1.8 (theoretical: 2) ---
// WHY 1.8 not 2.0: small tolerance for solver residual contaminating the finest mesh.
// WHY mesh_sizes {10, 20, 40, 80}: coarse enough that h^2 error dominates solver tolerance,
//   fine enough to give a reliable slope estimate.
TEST(ConvergenceOrder, TrapezoidalIsSecondOrder) {
    const std::vector<std::size_t> mesh_sizes = {10, 20, 40, 80};

    const double empirical_slope = goss::accuracy::estimate_convergence_slope(
        // problem_factory
        [](std::size_t num_intervals, const std::string& model_name) {
            return build_double_integrator_ocp(num_intervals, model_name,
                [](const auto& ocp, const std::string& name) {
                    return goss::transcription::Trapezoidal::compile(ocp, name);
                });
        },
        // error_at_mesh_size
        position_max_error,
        mesh_sizes,
        /*solver_tolerance=*/1e-10,
        /*model_name_prefix=*/"conv_trap_n");

    // Print the actual slope for calibration evidence in the task report.
    std::cout << "[CALIBRATION] Trapezoidal empirical slope = " << empirical_slope << std::endl;

    EXPECT_GE(empirical_slope, 1.8)
        << "Trapezoidal empirical convergence slope=" << empirical_slope
        << "; expected ≥ 1.8 (theoretical O(h²))";
    // Upper bound: O(h²) should not appear as O(h⁴) due to cancellation.
    EXPECT_LE(empirical_slope, 3.0)
        << "Trapezoidal slope too steep — likely a sign error in the error metric";
}

// --- Hermite-Simpson: empirical slope must be ≥ 3.5 (theoretical: 4) ---
// WHY 3.5: HermiteSimpson ConvergesAtFourthOrder in test_hermite_simpson.cpp uses 3.5.
// WHY mesh_sizes {5, 10, 20, 40}: at 5 intervals h=0.2, error~0.2^4=1.6e-3 >> solver tol.
TEST(ConvergenceOrder, HermiteSimpsonIsFourthOrder) {
    const std::vector<std::size_t> mesh_sizes = {5, 10, 20, 40};

    const double empirical_slope = goss::accuracy::estimate_convergence_slope(
        [](std::size_t num_intervals, const std::string& model_name) {
            return build_double_integrator_ocp(num_intervals, model_name,
                [](const auto& ocp, const std::string& name) {
                    return goss::transcription::HermiteSimpson::compile(ocp, name);
                });
        },
        position_max_error,
        mesh_sizes,
        /*solver_tolerance=*/1e-11,
        /*model_name_prefix=*/"conv_hs_n");

    // Print the actual slope for calibration evidence in the task report.
    std::cout << "[CALIBRATION] HermiteSimpson empirical slope = " << empirical_slope << std::endl;

    EXPECT_GE(empirical_slope, 3.5)
        << "HermiteSimpson empirical slope=" << empirical_slope
        << "; expected ≥ 3.5 (theoretical O(h⁴))";
    EXPECT_LE(empirical_slope, 6.0)
        << "HermiteSimpson slope unexpectedly steep — check error metric";
}

// --- LGL: errors at increasing node counts must decay faster than O(h⁴) ---
// WHY separate test from estimate_convergence_slope: LGL uses num_nodes (not num_intervals)
// as the resolution parameter, and the error-vs-N relationship is exponential (spectral),
// not a clean polynomial. We verify: (a) monotone decrease, (b) error at 16 nodes < 1e-8
// (spectral convergence threshold).
TEST(ConvergenceOrder, LGLConvergesSpectrally) {
    // WHY node counts {5, 8, 12, 16}: small enough that the AD recording is fast,
    // large enough to clearly distinguish O(h^4) from spectral convergence.
    const std::vector<std::size_t> node_counts = {5, 8, 12, 16};

    // Analytic optimal objective for x²+u² cost: J* = sinh(2T)/(2·sinh²(T)).
    // WHY use objective as proxy: the objective error converges spectrally (exponentially)
    // on smooth problems — it is O(spectral accuracy) and serves as a scalar summary of
    // the whole-trajectory approximation quality.
    const double kAnalyticObjective = analytic_objective(kTimeHorizon);

    // Solve at each node count, collect errors.
    std::vector<double> errors;
    errors.reserve(node_counts.size());
    for (std::size_t n_nodes : node_counts) {
        const std::size_t num_intervals = n_nodes - 1;  // LGL: num_nodes = num_intervals + 1
        const std::string model_name = "conv_lgl_n" + std::to_string(n_nodes);

        goss::model::Model model;
        const auto position_handle = model.add_state("position");
        const auto force_handle    = model.add_control("force");
        model.set_control_bounds(force_handle, -10.0, 10.0);
        model.set_initial_state(position_handle, 0.0);
        model.set_final_state(position_handle, 1.0);
        // LGL: num_intervals → num_nodes = num_intervals + 1 LGL nodes.
        model.set_mesh(0.0, kTimeHorizon, num_intervals);
        auto dynamics = [](const auto& state_vec, const auto& control_vec, auto /*time*/) {
            using ScalarT = typename std::decay_t<decltype(state_vec)>::value_type;
            return std::vector<ScalarT>{ control_vec[0] };
        };
        auto running_cost = [](const auto& state_vec, const auto& control_vec, auto /*time*/) {
            return state_vec[0] * state_vec[0] + control_vec[0] * control_vec[0];
        };
        auto ocp      = model.build(dynamics, running_cost);
        auto compiled = goss::transcription::LegendreGaussLobatto::compile(ocp, model_name);
        const auto trajectory = goss::accuracy::solve_and_extract_trajectory(
            compiled, /*initial_guess_value=*/0.5, /*solver_tolerance=*/1e-12);
        ASSERT_FALSE(trajectory.states.empty())
            << "LGL n_nodes=" << n_nodes << " solver failed";

        // Error: |J_numeric - J*| where J* = sinh(2T)/(2·sinh²(T)).
        // WHY objective as proxy: J* = sinh(2)/(2·sinh²(1)) ≈ 1.3130; error in J
        // reflects spectral accuracy of the full trajectory approximation.
        const double objective_error = std::abs(trajectory.objective_value - kAnalyticObjective);
        errors.push_back(objective_error);
    }

    // Errors must decrease monotonically with node count, until they reach
    // near-machine-precision saturation. Once below 1e-10, further refinement may
    // produce apparent increase due to floating-point cancellation in the NLP solver —
    // that is NOT a convergence failure; spectral accuracy has already been achieved.
    // WHY 1e-10 threshold (tightened from 1e-12): still well above machine epsilon
    // (~1e-16) and 100x below the spectral target of 1e-8. Tightening from 1e-12
    // exercises one more monotonicity comparison step. If FP noise between 1e-12 and
    // 1e-10 causes a spurious failure, revert to 1e-12 and note it.
    for (std::size_t i = 0; i + 1 < errors.size(); ++i) {
        if (errors[i] < 1e-10) break;  // near spectral saturation — stop monotonicity check
        EXPECT_LT(errors[i + 1], errors[i])
            << "LGL error must decrease as node count increases: "
            << "errors[" << i << "]=" << errors[i]
            << ", errors[" << i+1 << "]=" << errors[i+1];
    }

    // Print the LGL error sequence for calibration evidence in the task report.
    for (std::size_t i = 0; i < node_counts.size(); ++i) {
        std::cout << "[CALIBRATION] LGL n_nodes=" << node_counts[i]
                  << " objective_error=" << errors[i] << std::endl;
    }

    // Spectral convergence: error at 16 nodes must be < 1e-8
    // (LGL is spectrally accurate on smooth problems).
    EXPECT_LT(errors.back(), 1e-8)
        << "LGL with 16 nodes should achieve < 1e-8 on smooth double integrator"
        << "; actual error=" << errors.back()
        << " (analytic J*=" << kAnalyticObjective << ")";
}
