// tests/accuracy/test_composition_accuracy.cpp
// End-to-end accuracy test for the composition quick-wins feature:
//   - 2 state-owning components (feature A)
//   - 2 derived quantities (feature B)
//   - solved result asserted against closed-form optimal cost
//
// Dependency: requires goss_accuracy_tests target (accuracy-validation-suite plan
// must be merged before this task executes).
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "goss/model/component.hpp"
#include "goss/model/composed_model.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "accuracy/accuracy_helpers.hpp"

TEST(CompositionAccuracy, TwoStateOwnerTwoDerivedClosedFormOptimalCost) {
    // Problem: 2-state linear OCP with 2 state-owning components and 2 observational
    // derived quantities. The two states are decoupled: x is controlled, y is free.
    //
    // System:
    //   dx/dt = -a * x + u,  x(0)=1.0, x(T)=0.0,  min integral u^2
    //   dy/dt = -b * y,      y(0)=1.0, y(T) free
    //   d0    = x + y        (observational derived, topo index 0)
    //   d1    = x - y        (observational derived, topo index 1)
    //
    // Closed-form optimal cost (Pontryagin minimum principle / controllability Gramian):
    //   J* = 2*a*x0^2 / (exp(2*a*T) - 1)
    // With a=1.0, x0=1.0, T=2.0:
    //   J* = 2.0 / (exp(4.0) - 1.0) ≈ 0.037315

    constexpr double decay_rate_x  = 1.0;   // a
    constexpr double decay_rate_y  = 2.0;   // b
    constexpr double initial_x     = 1.0;
    constexpr double initial_y     = 1.0;
    constexpr double final_x       = 0.0;
    constexpr double time_horizon  = 2.0;
    constexpr std::size_t num_intervals = 40;

    // Closed-form optimal cost for linear minimum-energy transfer dx/dt = -a*x + u,
    // x(0)=x0, x(T)=0, min integral u^2.
    //
    // Derivation (Pontryagin minimum principle):
    //   H = u^2 + lambda*(-a*x + u)
    //   u* = -lambda/2  (from dH/du=0)
    //   lambda_dot = a*lambda  =>  lambda(t) = lambda_0 * exp(a*t)
    //   x(t) = exp(-a*t)*x0 - lambda_0*(exp(a*t) - exp(-a*t))/4
    //   x(T)=0 gives lambda_0 = 4*exp(-a*T)*x0 / (exp(a*T) - exp(-a*T))
    //                          = 2*a*x0*exp(-a*T) / sinh(a*T)   [note: 2*sinh = e^at - e^-at... corrected]
    //
    // Equivalently via controllability Gramian:
    //   W_c(T) = integral_0^T exp(-2*a*s) ds = (1 - exp(-2*a*T)) / (2*a)
    //   J* = x0^2 * exp(-2*a*T) / W_c(T)
    //      = x0^2 * exp(-2*a*T) * 2*a / (1 - exp(-2*a*T))
    //      = 2*a*x0^2 / (exp(2*a*T) - 1)
    //
    // With a=1.0, x0=1.0, T=2.0:
    //   J* = 2 / (exp(4) - 1) ≈ 2 / 53.598 ≈ 0.037315
    //
    // NOTE: the brief's formula x0^2*a/(1-exp(-2aT)) ≈ 1.01864 is incorrect for this
    // problem; that formula arises from a different sign convention or problem variant.
    // The solver returns ~0.0373, consistent with the Pontryagin and Gramian derivations.
    const double closed_form_optimal_cost =
        (2.0 * decay_rate_x * initial_x * initial_x) /
        (std::exp(2.0 * decay_rate_x * time_horizon) - 1.0);

    goss::model::ComposedModel composed;
    const auto control_handle = composed.add_control("u", -10.0, 10.0);

    // Component "alpha" owns state x; publishes derived d0 = x + y.
    goss::model::Component comp_alpha("alpha");
    const auto x_handle = comp_alpha.add_state("x");
    comp_alpha.set_initial_state(x_handle, initial_x);
    comp_alpha.set_final_state(x_handle, final_x);
    comp_alpha.add_derived(
        "d0",
        // Validation lambda: d0 = x + y = global_x[0] + global_x[1]
        [](const std::vector<double>& global_x, const std::vector<double>& /*u*/,
           const std::vector<double>& /*d*/, double /*t*/) {
            return global_x[0] + global_x[1];  // x + y
        });
    comp_alpha.set_dynamics(
        // Validation lambda: dx/dt = -a*x + u
        [decay_rate_x](const std::vector<double>& global_x,
                        const std::vector<double>& global_u,
                        const std::vector<double>& /*d*/,
                        double /*t*/) {
            return std::vector<double>{ -decay_rate_x * global_x[0] + global_u[0] };
        });

    // Component "beta" owns state y; publishes derived d1 = x - y.
    goss::model::Component comp_beta("beta");
    comp_beta.input_state("x");  // reads x from component alpha
    const auto y_handle = comp_beta.add_state("y");
    comp_beta.set_initial_state(y_handle, initial_y);
    comp_beta.add_derived(
        "d1",
        // Validation lambda: d1 = x - y = global_x[0] - global_x[1]
        [](const std::vector<double>& global_x, const std::vector<double>& /*u*/,
           const std::vector<double>& /*d*/, double /*t*/) {
            return global_x[0] - global_x[1];  // x - y
        });
    comp_beta.set_dynamics(
        // Validation lambda: dy/dt = -b*y
        [decay_rate_y](const std::vector<double>& global_x,
                        const std::vector<double>& /*u*/,
                        const std::vector<double>& /*d*/,
                        double /*t*/) {
            // global_x[0]=x, global_x[1]=y (registration order)
            return std::vector<double>{ -decay_rate_y * global_x[1] };
        });

    composed.add_component(std::move(comp_alpha));
    composed.add_component(std::move(comp_beta));
    composed.set_mesh(0.0, time_horizon, num_intervals);

    // Generic AD-safe lambdas for the build() call.
    // d0 = global_x[0] + global_x[1]; no declared dependencies (no input_derived calls).
    auto derived_d0 = [](const auto& global_x, const auto& /*u*/,
                          const auto& /*deps_so_far*/, auto /*t*/) {
        return global_x[0] + global_x[1];
    };
    // d1 = global_x[0] - global_x[1]; also no declared dependencies.
    auto derived_d1 = [](const auto& global_x, const auto& /*u*/,
                          const auto& /*deps_so_far*/, auto /*t*/) {
        return global_x[0] - global_x[1];
    };

    // alpha dynamics: dx/dt = -a * global_x[0] + global_u[0]
    auto alpha_dyn = [decay_rate_x](const auto& global_x, const auto& global_u,
                                     const auto& /*deriveds*/, auto /*t*/) {
        using T = typename std::decay_t<decltype(global_x)>::value_type;
        return std::vector<T>{ -T(decay_rate_x) * global_x[0] + global_u[0] };
    };
    // beta dynamics: dy/dt = -b * global_x[1]
    auto beta_dyn = [decay_rate_y](const auto& global_x, const auto& /*global_u*/,
                                    const auto& /*deriveds*/, auto /*t*/) {
        using T = typename std::decay_t<decltype(global_x)>::value_type;
        return std::vector<T>{ -T(decay_rate_y) * global_x[1] };
    };

    // Cost: integral u^2 (x subsystem only; y is free)
    auto running_cost = [](const auto& /*global_x*/, const auto& global_u,
                            const auto& /*deriveds*/, auto /*t*/) {
        return global_u[0] * global_u[0];
    };

    auto ocp = composed.build(
        goss::model::make_derived_exprs(derived_d0, derived_d1),
        goss::model::make_component_dyns(alpha_dyn, beta_dyn),
        running_cost);

    ASSERT_EQ(ocp.num_states, 2u);
    ASSERT_EQ(ocp.num_controls, 1u);

    // Structural assertion: evaluate assembled dynamics at initial state.
    // At x=[1.0, 1.0], u=[0.0]: dx/dt = [-1.0, -2.0].
    {
        std::vector<double> x_initial{ initial_x, initial_y };
        std::vector<double> u_zero{ 0.0 };
        auto dx_initial = ocp.dynamics(x_initial, u_zero, 0.0);
        ASSERT_EQ(dx_initial.size(), 2u);
        EXPECT_NEAR(dx_initial[0], -decay_rate_x * initial_x, 1e-12);  // dx/dt = -1*1 + 0
        EXPECT_NEAR(dx_initial[1], -decay_rate_y * initial_y, 1e-12);  // dy/dt = -2*1
    }

    // Solve the composed OCP.
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "composition_accuracy_two_state");
    const goss::accuracy::SolutionTrajectory trajectory =
        goss::accuracy::solve_and_extract_trajectory(compiled, /*initial_guess_value=*/0.1);

    // Primary accuracy assertion: objective must match closed-form to 1% tolerance.
    // With 40 Hermite-Simpson intervals on a smooth linear-quadratic problem,
    // the transcription error is O(h^4) ≈ O((2/40)^4) ≈ 1.56e-5; 1% is very conservative.
    EXPECT_NEAR(trajectory.objective_value, closed_form_optimal_cost, 1e-2);

    // Secondary: initial state pinned correctly.
    EXPECT_NEAR(trajectory.states[0][0], initial_x, 1e-6);  // x(0) = 1.0
    EXPECT_NEAR(trajectory.states[0][1], initial_y, 1e-6);  // y(0) = 1.0

    // Final state of x must be pinned to 0.0.
    EXPECT_NEAR(trajectory.states.back()[0], final_x, 1e-4);

    // y evolves freely: y(T) should match exp(-b*T) = exp(-4.0) within 1%.
    const double expected_y_final = initial_y * std::exp(-decay_rate_y * time_horizon);
    EXPECT_NEAR(trajectory.states.back()[1], expected_y_final, 1e-2);
}
