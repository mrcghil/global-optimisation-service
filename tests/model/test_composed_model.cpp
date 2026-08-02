// tests/model/test_composed_model.cpp
// Name-resolution and global-index tests for ComposedModel (Task 3).
// Task 5 build() tests appended at the bottom.
#include <gtest/gtest.h>
#include "goss/model/errors.hpp"
#include "goss/model/component.hpp"
#include "goss/model/composed_model.hpp"

// ---- Sanity test that was here before Task 3 ----
TEST(ComponentErrorInComposed, IsAlsoRuntimeError) {
    try {
        throw goss::model::ComponentError("test");
    } catch (const std::runtime_error& error) {
        EXPECT_EQ(std::string(error.what()), "test");
    }
}

// ---- Task 3 name-resolution tests ----

TEST(ComposedModel, AddComponentAndResolveNames) {
    goss::model::ComposedModel composed;
    goss::model::Component component_a("queue");
    component_a.add_state("queue_length");
    composed.add_component(std::move(component_a));
    EXPECT_NO_THROW(composed.resolve_names());
    EXPECT_EQ(composed.total_num_states(), 1u);
    EXPECT_EQ(composed.global_state_index("queue_length"), 0u);
}

TEST(ComposedModel, DuplicateStateNameAcrossComponentsThrows) {
    goss::model::ComposedModel composed;
    goss::model::Component component_a("a");
    component_a.add_state("x");
    goss::model::Component component_b("b");
    component_b.add_state("x");  // duplicate
    composed.add_component(std::move(component_a));
    composed.add_component(std::move(component_b));
    EXPECT_THROW(composed.resolve_names(), goss::model::ComponentError);
}

TEST(ComposedModel, UnresolvedInputStateThrows) {
    goss::model::ComposedModel composed;
    goss::model::Component component("service");
    component.input_state("nonexistent_state");
    composed.add_component(std::move(component));
    EXPECT_THROW(composed.resolve_names(), goss::model::ComponentError);
}

TEST(ComposedModel, ResolvedInputStateFromAnotherComponent) {
    goss::model::ComposedModel composed;
    goss::model::Component component_queue("queue");
    component_queue.add_state("queue_length");
    goss::model::Component component_service("service");
    component_service.input_state("queue_length");  // wired by name
    component_service.add_derived(
        "service_rate",
        [](const auto& /*x*/, const auto& /*u*/,
           const auto& /*d*/, double /*t*/) { return 1.0; });
    composed.add_component(std::move(component_queue));
    composed.add_component(std::move(component_service));
    EXPECT_NO_THROW(composed.resolve_names());
    EXPECT_EQ(composed.total_num_derived(), 1u);
    EXPECT_EQ(composed.global_derived_index("service_rate"), 0u);
}

// ---- Task 3 regression test: duplicate derived name across components ----

TEST(ComposedModel, DuplicateDerivedNameAcrossComponentsThrows) {
    goss::model::ComposedModel composed;
    goss::model::Component component_a("a");
    component_a.add_state("x");
    component_a.add_derived(
        "rate",
        [](const auto& /*x*/, const auto& /*u*/,
           const auto& /*d*/, double /*t*/) { return 1.0; });
    goss::model::Component component_b("b");
    component_b.add_state("y");
    component_b.add_derived(
        "rate",  // duplicate derived name
        [](const auto& /*x*/, const auto& /*u*/,
           const auto& /*d*/, double /*t*/) { return 2.0; });
    composed.add_component(std::move(component_a));
    composed.add_component(std::move(component_b));
    EXPECT_THROW(composed.resolve_names(), goss::model::ComponentError);
}

// ---- Task 4: topological sort and cycle detection tests ----

TEST(ComposedModel, SingleDerivedHasNoTopoCycles) {
    goss::model::ComposedModel composed;
    goss::model::Component component("c");
    component.add_state("x");
    component.add_derived("a", [](const auto&, const auto&, const auto&, double) { return 1.0; });
    composed.add_component(std::move(component));
    EXPECT_NO_THROW(composed.resolve_names());
    EXPECT_EQ(composed.total_num_derived(), 1u);
}

TEST(ComposedModel, LinearDerivedDependencyOrderedCorrectly) {
    // b depends on a; expected order: a first (index 0), b second (index 1).
    goss::model::ComposedModel composed;
    goss::model::Component component("c");
    component.add_state("x");
    component.add_derived("a", [](const auto&, const auto&, const auto&, double) { return 1.0; });
    component.input_derived("a");  // declares that the next derived reads derived["a"]
    component.add_derived("b", [](const auto&, const auto&, const auto& d, double) { return d[0] * 2.0; });
    composed.add_component(std::move(component));
    EXPECT_NO_THROW(composed.resolve_names());
    EXPECT_EQ(composed.global_derived_index("a"), 0u);
    EXPECT_EQ(composed.global_derived_index("b"), 1u);
}

TEST(ComposedModel, CyclicDerivedDependencyThrows) {
    // a depends on b, b depends on a — cycle.
    goss::model::ComposedModel composed;
    goss::model::Component component("c");
    component.add_state("x");
    component.input_derived("b");  // a depends on b
    component.add_derived("a", [](const auto&, const auto&, const auto& d, double) { return d[0]; });
    component.input_derived("a");  // b depends on a
    component.add_derived("b", [](const auto&, const auto&, const auto& d, double) { return d[0]; });
    composed.add_component(std::move(component));
    EXPECT_THROW(composed.resolve_names(), goss::model::ComponentError);
}

// ---- Task 5: ComposedModel::build() tests ----

// build() produces an OcpProblem with correct dimension metadata and mesh forwarded.
TEST(ComposedModel, BuildProducesOcpProblemWithCorrectDimensions) {
    goss::model::ComposedModel composed;
    composed.add_control("service_rate", 0.0, 5.0);

    // "service" component: has a derived quantity "svc" (no owned states)
    goss::model::Component component_service("service");
    component_service.add_derived(
        "svc",
        [](const auto& x, const auto& /*u*/, const auto& /*d*/, double /*t*/) {
            return x[0];  // trivial: svc = x[0]
        });

    // "queue" component: owns state "q", has dynamics, reads derived[0] = "svc"
    goss::model::Component component_queue("queue");
    component_queue.add_state("q");
    component_queue.set_dynamics(
        [](const std::vector<double>& /*x*/,
           const std::vector<double>& u,
           const std::vector<double>& d,
           double /*t*/) {
            return std::vector<double>{ 3.0 - d[0] };
        });

    composed.add_component(std::move(component_service));
    composed.add_component(std::move(component_queue));
    composed.set_mesh(0.0, 1.0, 5);

    // Generic lambdas for the AD path (no std::function — captured by value in the combined functor).
    // Signature for derived exprs: (const auto& x, const auto& u, const auto& deriveds_so_far, auto t) -> T
    auto derived_lambda = [](const auto& x, const auto& /*u*/,
                              const auto& /*d*/, auto /*t*/) {
        return x[0];
    };
    // Signature for component dynamics: (const auto& x, const auto& u, const auto& deriveds, auto t) -> vector<T>
    auto dyn_lambda = [](const auto& /*x*/, const auto& u,
                          const auto& d, auto /*t*/) {
        using T = typename std::decay_t<decltype(u)>::value_type;
        return std::vector<T>{ T(3.0) - d[0] };
    };
    // Signature for cost: (const auto& x, const auto& u, const auto& deriveds, auto t) -> T
    auto cost_lambda = [](const auto& x, const auto& /*u*/,
                           const auto& /*d*/, auto /*t*/) {
        return x[0];
    };

    auto ocp = composed.build(
        goss::model::make_derived_exprs(derived_lambda),
        goss::model::make_component_dyns(dyn_lambda),
        cost_lambda);
    EXPECT_EQ(ocp.num_states, 1u);
    EXPECT_EQ(ocp.num_controls, 1u);
    EXPECT_EQ(ocp.mesh.num_intervals, 5u);
}

// build() forwards bounds from components to the OcpProblem.
TEST(ComposedModel, BuildForwardsBoundsFromComponents) {
    goss::model::ComposedModel composed;
    composed.add_control("u", 0.0, 10.0);

    goss::model::Component comp("c");
    auto q_handle = comp.add_state("q");
    comp.set_state_bounds(q_handle, 0.0, 100.0);
    comp.set_initial_state(q_handle, 5.0);
    comp.set_dynamics(
        [](const std::vector<double>& /*x*/,
           const std::vector<double>& u,
           const std::vector<double>& /*d*/,
           double /*t*/) {
            return std::vector<double>{ -u[0] };
        });

    composed.add_component(std::move(comp));
    composed.set_mesh(0.0, 2.0, 4);

    auto dyn_lambda = [](const auto& /*x*/, const auto& u,
                          const auto& /*d*/, auto /*t*/) {
        using T = typename std::decay_t<decltype(u)>::value_type;
        return std::vector<T>{ -u[0] };
    };
    auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                           const auto& /*d*/, auto /*t*/) {
        using T = double;
        return T(0.0);
    };

    // No derived quantities → build with 0 derived lambdas via variadic overload.
    auto ocp = composed.build(
        goss::model::make_derived_exprs(),
        goss::model::make_component_dyns(dyn_lambda),
        cost_lambda);
    EXPECT_EQ(ocp.num_states, 1u);
    EXPECT_EQ(ocp.num_controls, 1u);
    EXPECT_EQ(ocp.state_lower[0], 0.0);
    EXPECT_EQ(ocp.state_upper[0], 100.0);
    EXPECT_EQ(ocp.initial_state[0], 5.0);
    EXPECT_EQ(ocp.initial_state_fixed[0], 1.0);
    EXPECT_EQ(ocp.control_lower[0], 0.0);
    EXPECT_EQ(ocp.control_upper[0], 10.0);
    EXPECT_EQ(ocp.mesh.t_initial, 0.0);
    EXPECT_EQ(ocp.mesh.t_final, 2.0);
    EXPECT_EQ(ocp.mesh.num_intervals, 4u);
}

// build() combined dynamics functor correctly assembles dx using derived quantities.
TEST(ComposedModel, BuildCombinedDynamicsEvaluatesCorrectly) {
    goss::model::ComposedModel composed;
    composed.add_control("u", 0.0, 5.0);

    // "rate" component: publishes derived "rate" = x[0] * 2.0
    goss::model::Component comp_rate("rate_comp");
    // no owned states — rate is purely derived from global x[0]

    // "queue" component: owns "q"; dx/dt = 1.0 - deriveds[0]
    goss::model::Component comp_queue("queue");
    comp_queue.add_state("q");
    comp_rate.add_derived(
        "rate",
        [](const auto& x, const auto& /*u*/, const auto& /*d*/, double /*t*/) {
            return x[0] * 2.0;
        });
    comp_queue.set_dynamics(
        [](const std::vector<double>& /*x*/,
           const std::vector<double>& /*u*/,
           const std::vector<double>& d,
           double /*t*/) {
            return std::vector<double>{ 1.0 - d[0] };
        });

    composed.add_component(std::move(comp_rate));
    composed.add_component(std::move(comp_queue));
    composed.set_mesh(0.0, 1.0, 3);

    // Generic lambdas passed at build() — these run under double AND AD scalar T.
    auto derived_lambda = [](const auto& x, const auto& /*u*/,
                              const auto& /*d*/, auto /*t*/) {
        return x[0] * decltype(x[0])(2.0);
    };
    auto dyn_lambda = [](const auto& /*x*/, const auto& /*u*/,
                          const auto& d, auto /*t*/) {
        using T = typename std::decay_t<decltype(d)>::value_type;
        return std::vector<T>{ T(1.0) - d[0] };
    };
    auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                           const auto& /*d*/, auto /*t*/) {
        return double(0.0);
    };

    auto ocp = composed.build(
        goss::model::make_derived_exprs(derived_lambda),
        goss::model::make_component_dyns(dyn_lambda),
        cost_lambda);

    // Evaluate combined dynamics at x = [3.0], u = [0.0], t = 0.0.
    // Expected: rate = 3.0 * 2.0 = 6.0;  dx/dt[0] = 1.0 - 6.0 = -5.0.
    std::vector<double> x{3.0};
    std::vector<double> u{0.0};
    auto dx = ocp.dynamics(x, u, 0.0);
    ASSERT_EQ(dx.size(), 1u);
    EXPECT_DOUBLE_EQ(dx[0], -5.0);
}

// build() throws when no component owns any state (Important #1 guard).
TEST(ComposedModel, BuildWithNoStateOwnerThrows) {
    goss::model::ComposedModel composed;
    // Add a component that only has a derived quantity — no owned states.
    goss::model::Component comp("derived_only");
    comp.add_derived(
        "rate",
        [](const auto& /*x*/, const auto& /*u*/,
           const auto& /*d*/, double /*t*/) { return 1.0; });
    composed.add_component(std::move(comp));
    composed.set_mesh(0.0, 1.0, 4);

    auto derived_lambda = [](const auto& /*x*/, const auto& /*u*/,
                              const auto& /*d*/, auto /*t*/) { return 1.0; };
    // queue component dynamics — never reached but satisfies template signature
    auto dyn_lambda = [](const auto& /*x*/, const auto& /*u*/,
                          const auto& /*d*/, auto /*t*/) {
        using T = double;
        return std::vector<T>{};
    };
    auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                           const auto& /*d*/, auto /*t*/) { return double(0.0); };
    EXPECT_THROW(
        composed.build(
            goss::model::make_derived_exprs(derived_lambda),
            goss::model::make_component_dyns(dyn_lambda),
            cost_lambda),
        goss::model::ComponentError);
}

// build() cost lambda sees real derived values, not zero-filled placeholder (Important #2).
// Fixture: derived d = 2 * x[0]; cost = d; so ocp.cost({x0}, {}, t) should equal 2 * x0.
TEST(ComposedModel, BuildCostSeesDerivedValue) {
    goss::model::ComposedModel composed;

    // "queue" component: owns state "q" and registers one derived quantity "d_val".
    // The derived must be registered with add_derived() so that total_num_derived()==1,
    // matching the 3-arg (1-derived) build() overload's I2 guard.
    goss::model::Component comp_queue("queue");
    comp_queue.add_state("q");
    comp_queue.add_derived(
        "d_val",
        [](const std::vector<double>& x, const std::vector<double>& /*u*/,
           const std::vector<double>& /*d*/, double /*t*/) {
            return x[0] * 2.0;  // validation lambda: d_val = 2 * q
        });
    comp_queue.set_dynamics(
        [](const std::vector<double>& /*x*/,
           const std::vector<double>& /*u*/,
           const std::vector<double>& /*d*/,
           double /*t*/) {
            return std::vector<double>{0.0};
        });

    composed.add_component(std::move(comp_queue));
    composed.set_mesh(0.0, 1.0, 4);

    // Generic derived lambda for the AD path: d[0] = 2 * x[0]
    auto derived_lambda = [](const auto& x, const auto& /*u*/,
                              const auto& /*d*/, auto /*t*/) {
        return x[0] * decltype(x[0])(2.0);
    };
    auto dyn_lambda = [](const auto& /*x*/, const auto& /*u*/,
                          const auto& /*d*/, auto /*t*/) {
        return std::vector<double>{0.0};
    };
    // cost = d[0]  (so cost should equal 2 * x[0])
    auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                           const auto& d, auto /*t*/) {
        return d[0];
    };

    auto ocp = composed.build(
        goss::model::make_derived_exprs(derived_lambda),
        goss::model::make_component_dyns(dyn_lambda),
        cost_lambda);

    // Evaluate cost at x = [3.0], u = [], t = 0.0.
    // Expected: d[0] = 2 * 3.0 = 6.0.
    const double x0 = 3.0;
    std::vector<double> x{x0};
    std::vector<double> u{};
    double result = ocp.cost(x, u, 0.0);
    EXPECT_DOUBLE_EQ(result, 2.0 * x0);
}

// ---- Task 6: pre-build dynamics dimension validation tests ----

// A component whose dynamics lambda returns the wrong number of derivatives throws
// ComponentError at build() time (before any AD codegen).
TEST(ComposedModel, DynamicsDimensionMismatchThrows) {
    goss::model::ComposedModel composed;
    composed.add_control("u", 0.0, 1.0);
    goss::model::Component component("c");
    component.add_state("x");
    // Validation lambda returns 2 values but only 1 state owned — mismatch.
    component.set_dynamics(
        [](const std::vector<double>&, const std::vector<double>&,
           const std::vector<double>&, double) {
            return std::vector<double>{ 1.0, 2.0 };  // wrong size
        });
    composed.add_component(std::move(component));
    composed.set_mesh(0.0, 1.0, 3);

    auto dyn_lambda = [](const auto& /*x*/, const auto& /*u*/,
                          const auto& /*d*/, auto /*t*/) {
        using T = double;
        return std::vector<T>{ T(1.0), T(2.0) };  // still wrong size
    };
    auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                           const auto& /*d*/, auto /*t*/) { return 0.0; };
    EXPECT_THROW(
        composed.build(
            goss::model::make_derived_exprs(),
            goss::model::make_component_dyns(dyn_lambda),
            cost_lambda),
        goss::model::ComponentError);
}

// A composed model whose dynamics lambda returns the CORRECT number of derivatives does NOT throw.
TEST(ComposedModel, AssembledDynamicsEvaluatesCorrectlyUnderDouble) {
    // Build a simple 1-state composed model and verify the dynamics value.
    goss::model::ComposedModel composed;
    goss::model::Component component("c");
    component.add_state("x");
    component.add_derived(
        "d_val",
        [](const auto& x, const auto& /*u*/,
           const auto& /*prev_deriveds*/, double /*t*/) {
            return x[0] * 2.0;
        });
    component.set_dynamics(
        [](const std::vector<double>& /*x*/,
           const std::vector<double>& /*u*/,
           const std::vector<double>& d,
           double /*t*/) {
            return std::vector<double>{ -d[0] };  // dx/dt = -d_val = -2x
        });
    composed.add_component(std::move(component));
    composed.set_mesh(0.0, 1.0, 3);

    auto derived_lambda = [](const auto& x, const auto& /*u*/,
                              const auto& /*prev_d*/, auto /*t*/) {
        return x[0] * decltype(x[0]){2.0};
    };
    auto dyn_lambda = [](const auto& /*x*/, const auto& /*u*/,
                          const auto& d, auto /*t*/) {
        using T = typename std::decay_t<decltype(d)>::value_type;
        return std::vector<T>{ -d[0] };
    };
    auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                           const auto& /*d*/, auto /*t*/) { return 0.0; };

    auto ocp = composed.build(
        goss::model::make_derived_exprs(derived_lambda),
        goss::model::make_component_dyns(dyn_lambda),
        cost_lambda);
    // Evaluate assembled dynamics at x={3.0}, u={}, t=0.
    std::vector<double> x_test{3.0}, u_test{};
    auto dx = ocp.dynamics(x_test, u_test, 0.0);
    ASSERT_EQ(dx.size(), 1u);
    EXPECT_DOUBLE_EQ(dx[0], -6.0);  // -2 * 3.0
}

// ---- Final-review regression tests ----

// C1 regression: 0-derived build() with a STATELESS component registered BEFORE the
// state-owning component. Before the fix, comp0_offset/comp0_nstates were always taken
// from components_[0], which here is the stateless component → dx written to wrong slot
// (or incorrectly sized). After the fix, the find-first-state-owner loop picks the
// queue component at index 1, and dynamics must be non-zero / correct.
TEST(ComposedModel, ZeroDerivedStateOwnerNotFirst) {
    goss::model::ComposedModel composed;

    // Stateless component registered FIRST — no owned states, no dynamics.
    goss::model::Component stateless("stateless");
    // (no add_state, no set_dynamics)

    // State-owning component registered SECOND.
    goss::model::Component queue("queue");
    auto q_handle = queue.add_state("q");
    queue.set_initial_state(q_handle, 0.0);
    // dx/dt = 2.0 (non-zero constant — so any dynamics eval should return 2.0)
    queue.set_dynamics(
        [](const std::vector<double>& /*x*/,
           const std::vector<double>& /*u*/,
           const std::vector<double>& /*d*/,
           double /*t*/) {
            return std::vector<double>{ 2.0 };
        });

    composed.add_component(std::move(stateless));
    composed.add_component(std::move(queue));
    composed.set_mesh(0.0, 1.0, 4);

    // Generic AD-safe dynamics lambda.
    auto dyn_lambda = [](const auto& /*x*/, const auto& /*u*/,
                          const auto& /*d*/, auto /*t*/) {
        using T = double;
        return std::vector<T>{ T(2.0) };
    };
    auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                           const auto& /*d*/, auto /*t*/) {
        return double(0.0);
    };

    // build() must succeed and assembled dynamics must be non-zero.
    auto ocp = composed.build(
        goss::model::make_derived_exprs(),
        goss::model::make_component_dyns(dyn_lambda),
        cost_lambda);

    // Evaluate assembled dynamics at x={0.0}, u={}, t=0.0.
    // The state-owning component sits at global state offset 0 (the stateless
    // component contributes 0 states). dx[0] must equal 2.0.
    std::vector<double> x_test{0.0}, u_test{};
    auto dx = ocp.dynamics(x_test, u_test, 0.0);
    ASSERT_EQ(dx.size(), 1u);
    // C1 fix verification: dx[0] must be 2.0 (non-zero); before the fix it would be 0.0
    // because the stateless component has no dynamics output mapped to offset 0.
    EXPECT_DOUBLE_EQ(dx[0], 2.0);
}

// Task 2: the old I1 upper-bound guard (>1 state owner throws) is REMOVED.
// Two state-owning components now build SUCCESSFULLY when one dynamics lambda is provided
// per state-owning component (in component registration order).
TEST(ComposedModel, MultipleStateOwnersBuildSucceeds) {
    goss::model::ComposedModel composed;

    goss::model::Component comp_a("a");
    comp_a.add_state("x");
    comp_a.set_dynamics(
        [](const std::vector<double>&, const std::vector<double>&,
           const std::vector<double>&, double) {
            return std::vector<double>{ 1.0 };
        });

    goss::model::Component comp_b("b");
    comp_b.add_state("y");
    comp_b.set_dynamics(
        [](const std::vector<double>&, const std::vector<double>&,
           const std::vector<double>&, double) {
            return std::vector<double>{ 2.0 };
        });

    composed.add_component(std::move(comp_a));
    composed.add_component(std::move(comp_b));
    composed.set_mesh(0.0, 1.0, 3);

    // Provide one dynamics lambda per state-owning component (comp_a first, comp_b second).
    auto dyn_a = [](const auto& /*x*/, const auto& /*u*/,
                    const auto& /*d*/, auto /*t*/) {
        using T = double;
        return std::vector<T>{ T(1.0) };
    };
    auto dyn_b = [](const auto& /*x*/, const auto& /*u*/,
                    const auto& /*d*/, auto /*t*/) {
        using T = double;
        return std::vector<T>{ T(2.0) };
    };
    auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                           const auto& /*d*/, auto /*t*/) {
        return double(0.0);
    };
    // Two state-owning components are now VALID — build() must succeed.
    auto ocp = composed.build(
        goss::model::make_derived_exprs(),
        goss::model::make_component_dyns(dyn_a, dyn_b),
        cost_lambda);
    EXPECT_EQ(ocp.num_states, 2u);
}

// I2 regression: calling the 0-derived build() on a model that has 1 derived → throws.
TEST(ComposedModel, WrongDerivedCountForOverloadThrows) {
    goss::model::ComposedModel composed;

    goss::model::Component comp("c");
    comp.add_state("q");
    comp.add_derived(
        "rate",
        [](const auto& /*x*/, const auto& /*u*/,
           const auto& /*d*/, double /*t*/) { return 1.0; });
    comp.set_dynamics(
        [](const std::vector<double>&, const std::vector<double>&,
           const std::vector<double>&, double) {
            return std::vector<double>{ 0.0 };
        });

    composed.add_component(std::move(comp));
    composed.set_mesh(0.0, 1.0, 3);

    // Use variadic build() providing 0 derived-expr lambdas for a model with 1 derived
    // → I2 guard must throw ComponentError.
    auto dyn_lambda = [](const auto& /*x*/, const auto& /*u*/,
                          const auto& /*d*/, auto /*t*/) {
        using T = double;
        return std::vector<T>{ T(0.0) };
    };
    auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                           const auto& /*d*/, auto /*t*/) {
        return double(0.0);
    };
    EXPECT_THROW(
        composed.build(
            goss::model::make_derived_exprs(),      // wrong: 0 provided, 1 needed
            goss::model::make_component_dyns(dyn_lambda),
            cost_lambda),
        goss::model::ComponentError);
}

// I3 regression: a component that calls input_derived("x") but never calls add_derived →
// the pending name is never flushed, and build() must throw ComponentError.
TEST(ComposedModel, DanglingInputDerivedThrows) {
    goss::model::ComposedModel composed;

    goss::model::Component comp("c");
    comp.add_state("q");
    // input_derived declares a dependency for the NEXT add_derived() call, but we never
    // call add_derived() — so the name stays in the pending list (dangling).
    comp.input_derived("some_derived");
    comp.set_dynamics(
        [](const std::vector<double>&, const std::vector<double>&,
           const std::vector<double>&, double) {
            return std::vector<double>{ 0.0 };
        });

    composed.add_component(std::move(comp));
    composed.set_mesh(0.0, 1.0, 3);

    auto dyn_lambda = [](const auto& /*x*/, const auto& /*u*/,
                          const auto& /*d*/, auto /*t*/) {
        using T = double;
        return std::vector<T>{ T(0.0) };
    };
    auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                           const auto& /*d*/, auto /*t*/) {
        return double(0.0);
    };
    EXPECT_THROW(
        composed.build(
            goss::model::make_derived_exprs(),
            goss::model::make_component_dyns(dyn_lambda),
            cost_lambda),
        goss::model::ComponentError);
}

// ---- Task 1 (composition-quickwins): variadic helper compile+size tests ----

// Task 1 — helpers compile and produce tuples of correct size.
TEST(ComposedModelVariadic, MakeDerivedExprsProducesCorrectTupleSize) {
    auto derived_tuple = goss::model::make_derived_exprs(
        [](const auto& x, const auto& /*u*/, const auto& /*d*/, auto /*t*/) { return x[0]; },
        [](const auto& x, const auto& /*u*/, const auto& d, auto /*t*/) { return d[0] + x[0]; });
    constexpr std::size_t expected_size = 2u;
    EXPECT_EQ(std::tuple_size_v<decltype(derived_tuple)>, expected_size);
}

TEST(ComposedModelVariadic, MakeComponentDynsProducesCorrectTupleSize) {
    auto dyn_tuple = goss::model::make_component_dyns(
        [](const auto& /*x*/, const auto& /*u*/, const auto& /*d*/, auto /*t*/) {
            using T = double;
            return std::vector<T>{ T(0.0) };
        },
        [](const auto& /*x*/, const auto& /*u*/, const auto& /*d*/, auto /*t*/) {
            using T = double;
            return std::vector<T>{ T(0.0) };
        });
    constexpr std::size_t expected_size = 2u;
    EXPECT_EQ(std::tuple_size_v<decltype(dyn_tuple)>, expected_size);
}

// build() without mesh set throws.
TEST(ComposedModel, BuildWithoutMeshThrows) {
    goss::model::ComposedModel composed;
    goss::model::Component comp("c");
    comp.add_state("q");
    comp.set_dynamics(
        [](const std::vector<double>& /*x*/,
           const std::vector<double>& /*u*/,
           const std::vector<double>& /*d*/,
           double /*t*/) {
            return std::vector<double>{0.0};
        });
    composed.add_component(std::move(comp));
    // no set_mesh() call

    auto dyn_lambda = [](const auto& /*x*/, const auto& /*u*/,
                          const auto& /*d*/, auto /*t*/) {
        using T = double;
        return std::vector<T>{T(0.0)};
    };
    auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                           const auto& /*d*/, auto /*t*/) {
        return double(0.0);
    };
    EXPECT_THROW(
        composed.build(
            goss::model::make_derived_exprs(),
            goss::model::make_component_dyns(dyn_lambda),
            cost_lambda),
        goss::model::ModelError);
}

// ---- Task 2: variadic build() tests ----

// Task 2 — variadic build() with 0 derived and 1 state-owning component.
TEST(ComposedModelVariadic, ZeroDerivedOneDynBuildSucceeds) {
    goss::model::ComposedModel composed;
    composed.add_control("u", 0.0, 5.0);

    goss::model::Component comp("c");
    auto q_handle = comp.add_state("q");
    comp.set_initial_state(q_handle, 1.0);
    comp.set_dynamics(
        [](const std::vector<double>& /*x*/,
           const std::vector<double>& u,
           const std::vector<double>& /*d*/,
           double /*t*/) {
            return std::vector<double>{ u[0] };
        });
    composed.add_component(std::move(comp));
    composed.set_mesh(0.0, 1.0, 4);

    auto dyn_lambda = [](const auto& /*x*/, const auto& u,
                          const auto& /*d*/, auto /*t*/) {
        using T = typename std::decay_t<decltype(u)>::value_type;
        return std::vector<T>{ u[0] };
    };
    auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                           const auto& /*d*/, auto /*t*/) {
        return double(0.0);
    };
    auto ocp = composed.build(
        goss::model::make_derived_exprs(),
        goss::model::make_component_dyns(dyn_lambda),
        cost_lambda);
    EXPECT_EQ(ocp.num_states, 1u);
    EXPECT_EQ(ocp.num_controls, 1u);
}

// Task 2 — variadic build() with 1 derived and 1 state-owning component.
TEST(ComposedModelVariadic, OneDerivedOneDynBuildSucceeds) {
    goss::model::ComposedModel composed;
    composed.add_control("u", 0.0, 5.0);

    goss::model::Component comp("c");
    comp.add_state("q");
    comp.add_derived(
        "d_val",
        [](const std::vector<double>& x, const std::vector<double>& /*u*/,
           const std::vector<double>& /*d*/, double /*t*/) { return x[0] * 2.0; });
    comp.set_dynamics(
        [](const std::vector<double>& /*x*/, const std::vector<double>& /*u*/,
           const std::vector<double>& d, double /*t*/) {
            return std::vector<double>{ -d[0] };
        });
    composed.add_component(std::move(comp));
    composed.set_mesh(0.0, 1.0, 3);

    auto derived_lambda = [](const auto& x, const auto& /*u*/,
                              const auto& /*d*/, auto /*t*/) {
        return x[0] * decltype(x[0])(2.0);
    };
    auto dyn_lambda = [](const auto& /*x*/, const auto& /*u*/,
                          const auto& d, auto /*t*/) {
        using T = typename std::decay_t<decltype(d)>::value_type;
        return std::vector<T>{ -d[0] };
    };
    auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                           const auto& /*d*/, auto /*t*/) {
        return double(0.0);
    };
    auto ocp = composed.build(
        goss::model::make_derived_exprs(derived_lambda),
        goss::model::make_component_dyns(dyn_lambda),
        cost_lambda);
    EXPECT_EQ(ocp.num_states, 1u);

    std::vector<double> x_test{ 3.0 }, u_test{ 0.0 };
    auto dx = ocp.dynamics(x_test, u_test, 0.0);
    ASSERT_EQ(dx.size(), 1u);
    EXPECT_DOUBLE_EQ(dx[0], -6.0);  // -2 * 3.0
}

// Task 2 — guard: wrong number of derived-expr lambdas throws ComponentError.
TEST(ComposedModelVariadic, WrongDerivedExprCountThrows) {
    goss::model::ComposedModel composed;
    goss::model::Component comp("c");
    comp.add_state("q");
    comp.add_derived(
        "d_val",
        [](const std::vector<double>& x, const std::vector<double>& /*u*/,
           const std::vector<double>& /*d*/, double /*t*/) { return x[0]; });
    comp.set_dynamics(
        [](const std::vector<double>&, const std::vector<double>&,
           const std::vector<double>&, double) { return std::vector<double>{ 0.0 }; });
    composed.add_component(std::move(comp));
    composed.set_mesh(0.0, 1.0, 3);

    // Provide 0 derived-expr lambdas but model has 1 derived → guard must throw.
    auto dyn_lambda = [](const auto& /*x*/, const auto& /*u*/,
                          const auto& /*d*/, auto /*t*/) {
        using T = double;
        return std::vector<T>{ T(0.0) };
    };
    auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                           const auto& /*d*/, auto /*t*/) { return 0.0; };
    EXPECT_THROW(
        composed.build(
            goss::model::make_derived_exprs(),      // wrong: 0 provided, 1 needed
            goss::model::make_component_dyns(dyn_lambda),
            cost_lambda),
        goss::model::ComponentError);
}

// Task 2 — guard: wrong number of dynamics lambdas throws ComponentError.
TEST(ComposedModelVariadic, WrongDynLambdaCountThrows) {
    goss::model::ComposedModel composed;
    goss::model::Component comp("c");
    comp.add_state("q");
    comp.set_dynamics(
        [](const std::vector<double>&, const std::vector<double>&,
           const std::vector<double>&, double) { return std::vector<double>{ 0.0 }; });
    composed.add_component(std::move(comp));
    composed.set_mesh(0.0, 1.0, 3);

    // Provide 0 dyn lambdas but model has 1 state-owning component → guard must throw.
    auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                           const auto& /*d*/, auto /*t*/) { return 0.0; };
    EXPECT_THROW(
        composed.build(
            goss::model::make_derived_exprs(),
            goss::model::make_component_dyns(),     // wrong: 0 provided, 1 needed
            cost_lambda),
        goss::model::ComponentError);
}

// Task 3 — two state-owning components: dimensions and assembled dynamics correct.
TEST(ComposedModelMultiStateOwner, TwoStateOwnersProduceCorrectGlobalDimensions) {
    goss::model::ComposedModel composed;
    composed.add_control("u", 0.0, 1.0);

    goss::model::Component comp_a("a");
    comp_a.add_state("x_a");
    comp_a.set_dynamics(
        [](const std::vector<double>&, const std::vector<double>&,
           const std::vector<double>&, double) {
            return std::vector<double>{ 1.0 };
        });

    goss::model::Component comp_b("b");
    comp_b.add_state("x_b");
    comp_b.set_dynamics(
        [](const std::vector<double>&, const std::vector<double>&,
           const std::vector<double>&, double) {
            return std::vector<double>{ 2.0 };
        });

    composed.add_component(std::move(comp_a));
    composed.add_component(std::move(comp_b));
    composed.set_mesh(0.0, 1.0, 4);

    auto dyn_a = [](const auto& /*x*/, const auto& /*u*/,
                     const auto& /*d*/, auto /*t*/) {
        using T = double;
        return std::vector<T>{ T(1.0) };
    };
    auto dyn_b = [](const auto& /*x*/, const auto& /*u*/,
                     const auto& /*d*/, auto /*t*/) {
        using T = double;
        return std::vector<T>{ T(2.0) };
    };
    auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                           const auto& /*d*/, auto /*t*/) { return double(0.0); };

    auto ocp = composed.build(
        goss::model::make_derived_exprs(),
        goss::model::make_component_dyns(dyn_a, dyn_b),
        cost_lambda);

    EXPECT_EQ(ocp.num_states, 2u);
    EXPECT_EQ(ocp.num_controls, 1u);
}

TEST(ComposedModelMultiStateOwner, TwoStateOwnersAssembledDynamicsCorrect) {
    // Two components: a owns x_a (dx_a/dt = 3.0), b owns x_b (dx_b/dt = -2.0).
    // Global state vector: [x_a, x_b] (component registration order).
    // Expected assembled dx: [3.0, -2.0].
    goss::model::ComposedModel composed;

    goss::model::Component comp_a("a");
    comp_a.add_state("x_a");
    comp_a.set_dynamics(
        [](const std::vector<double>&, const std::vector<double>&,
           const std::vector<double>&, double) {
            return std::vector<double>{ 3.0 };
        });

    goss::model::Component comp_b("b");
    comp_b.add_state("x_b");
    comp_b.set_dynamics(
        [](const std::vector<double>&, const std::vector<double>&,
           const std::vector<double>&, double) {
            return std::vector<double>{ -2.0 };
        });

    composed.add_component(std::move(comp_a));
    composed.add_component(std::move(comp_b));
    composed.set_mesh(0.0, 1.0, 4);

    auto dyn_a = [](const auto& /*x*/, const auto& /*u*/,
                     const auto& /*d*/, auto /*t*/) {
        using T = double;
        return std::vector<T>{ T(3.0) };
    };
    auto dyn_b = [](const auto& /*x*/, const auto& /*u*/,
                     const auto& /*d*/, auto /*t*/) {
        using T = double;
        return std::vector<T>{ T(-2.0) };
    };
    auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                           const auto& /*d*/, auto /*t*/) { return double(0.0); };

    auto ocp = composed.build(
        goss::model::make_derived_exprs(),
        goss::model::make_component_dyns(dyn_a, dyn_b),
        cost_lambda);

    std::vector<double> x_test{ 0.0, 0.0 }, u_test{};
    auto dx = ocp.dynamics(x_test, u_test, 0.0);
    ASSERT_EQ(dx.size(), 2u);
    // x_a is global slot 0 (comp_a registered first), x_b is global slot 1.
    EXPECT_DOUBLE_EQ(dx[0], 3.0);
    EXPECT_DOUBLE_EQ(dx[1], -2.0);
}

TEST(ComposedModelMultiStateOwner, TwoStateOwnersCrossStateRead) {
    // Component b reads x_a (owned by component a) via input_state.
    // dx_a/dt = 1.0; dx_b/dt = x[0] * 2.0 = x_a * 2.0.
    goss::model::ComposedModel composed;

    goss::model::Component comp_a("a");
    comp_a.add_state("x_a");
    comp_a.set_dynamics(
        [](const std::vector<double>&, const std::vector<double>&,
           const std::vector<double>&, double) {
            return std::vector<double>{ 1.0 };
        });

    goss::model::Component comp_b("b");
    comp_b.input_state("x_a");  // declares dependency; global index resolved to 0
    comp_b.add_state("x_b");
    comp_b.set_dynamics(
        [](const std::vector<double>& x, const std::vector<double>&,
           const std::vector<double>&, double) {
            // x[0] is x_a (global state index 0 after resolve_names)
            return std::vector<double>{ x[0] * 2.0 };
        });

    composed.add_component(std::move(comp_a));
    composed.add_component(std::move(comp_b));
    composed.set_mesh(0.0, 1.0, 4);

    auto dyn_a = [](const auto& /*x*/, const auto& /*u*/,
                     const auto& /*d*/, auto /*t*/) {
        using T = double;
        return std::vector<T>{ T(1.0) };
    };
    auto dyn_b = [](const auto& x, const auto& /*u*/,
                     const auto& /*d*/, auto /*t*/) {
        using T = typename std::decay_t<decltype(x)>::value_type;
        return std::vector<T>{ x[0] * T(2.0) };
    };
    auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                           const auto& /*d*/, auto /*t*/) { return double(0.0); };

    auto ocp = composed.build(
        goss::model::make_derived_exprs(),
        goss::model::make_component_dyns(dyn_a, dyn_b),
        cost_lambda);

    // At x = [3.0, 5.0]: dx_a = 1.0, dx_b = 3.0 * 2.0 = 6.0.
    std::vector<double> x_test{ 3.0, 5.0 }, u_test{};
    auto dx = ocp.dynamics(x_test, u_test, 0.0);
    ASSERT_EQ(dx.size(), 2u);
    EXPECT_DOUBLE_EQ(dx[0], 1.0);
    EXPECT_DOUBLE_EQ(dx[1], 6.0);
}

// Guard: zero state-owning components still throws (I1 lower bound preserved).
TEST(ComposedModelMultiStateOwner, ZeroStateOwnersStillThrows) {
    goss::model::ComposedModel composed;
    goss::model::Component comp("derived_only");
    comp.add_derived(
        "rate",
        [](const auto& /*x*/, const auto& /*u*/,
           const auto& /*d*/, double /*t*/) { return 1.0; });
    composed.add_component(std::move(comp));
    composed.set_mesh(0.0, 1.0, 4);

    auto derived_lambda = [](const auto& /*x*/, const auto& /*u*/,
                              const auto& /*d*/, auto /*t*/) { return 1.0; };
    auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                           const auto& /*d*/, auto /*t*/) { return double(0.0); };
    EXPECT_THROW(
        composed.build(
            goss::model::make_derived_exprs(derived_lambda),
            goss::model::make_component_dyns(),
            cost_lambda),
        goss::model::ComponentError);
}

// ---- Task 4 (composition-quickwins): multi-derived topo-ordered evaluation tests ----

// Task 4 — two derived quantities with a dependency chain: b depends on a.
TEST(ComposedModelMultiDerived, TwoDerivedWithDependencyChainEvaluatesCorrectly) {
    // Derived a = x[0] * 3.0; derived b depends on a: b = a + 1.0.
    // Component dynamics: dx/dt = -b = -(a + 1.0) = -(3 * x[0] + 1).
    // At x = [2.0]: a = 6.0, b = 7.0, dx/dt = -7.0.
    goss::model::ComposedModel composed;

    goss::model::Component comp("c");
    comp.add_state("x");
    comp.add_derived(
        "a",
        [](const std::vector<double>& x, const std::vector<double>& /*u*/,
           const std::vector<double>& /*d*/, double /*t*/) { return x[0] * 3.0; });
    comp.input_derived("a");  // declares: b depends on a
    comp.add_derived(
        "b",
        [](const std::vector<double>& /*x*/, const std::vector<double>& /*u*/,
           const std::vector<double>& d, double /*t*/) {
            return d[0] + 1.0;  // d[0] is a (topo index 0)
        });
    comp.set_dynamics(
        [](const std::vector<double>& /*x*/, const std::vector<double>& /*u*/,
           const std::vector<double>& d, double /*t*/) {
            return std::vector<double>{ -d[1] };  // d[1] is b (topo index 1)
        });
    composed.add_component(std::move(comp));
    composed.set_mesh(0.0, 1.0, 4);

    auto derived_a_lambda = [](const auto& x, const auto& /*u*/,
                                const auto& /*d*/, auto /*t*/) {
        return x[0] * decltype(x[0])(3.0);
    };
    // derived_b_lambda: d is the deps_so_far slice — contains only d[0] = a.
    auto derived_b_lambda = [](const auto& /*x*/, const auto& /*u*/,
                                const auto& d, auto /*t*/) {
        using T = typename std::decay_t<decltype(d)>::value_type;
        return d[0] + T(1.0);
    };
    auto dyn_lambda = [](const auto& /*x*/, const auto& /*u*/,
                          const auto& d, auto /*t*/) {
        using T = typename std::decay_t<decltype(d)>::value_type;
        return std::vector<T>{ -d[1] };  // d[1] is b in the full deriveds vector
    };
    auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                           const auto& /*d*/, auto /*t*/) { return double(0.0); };

    auto ocp = composed.build(
        goss::model::make_derived_exprs(derived_a_lambda, derived_b_lambda),
        goss::model::make_component_dyns(dyn_lambda),
        cost_lambda);

    EXPECT_EQ(ocp.num_states, 1u);
    std::vector<double> x_test{ 2.0 }, u_test{};
    auto dx = ocp.dynamics(x_test, u_test, 0.0);
    ASSERT_EQ(dx.size(), 1u);
    // a = 2.0 * 3.0 = 6.0; b = 6.0 + 1.0 = 7.0; dx/dt = -7.0
    EXPECT_DOUBLE_EQ(dx[0], -7.0);
}

TEST(ComposedModelMultiDerived, TwoDerivedNoDependencyBothEvaluated) {
    // Two independent derived quantities (no dependency between them).
    // a = x[0]; b = x[0] * 2.0.
    // dx/dt = a + b = x[0] + 2 * x[0] = 3 * x[0].
    // At x = [4.0]: dx/dt = 12.0.
    goss::model::ComposedModel composed;

    goss::model::Component comp("c");
    comp.add_state("x");
    comp.add_derived(
        "a",
        [](const std::vector<double>& x, const std::vector<double>& /*u*/,
           const std::vector<double>& /*d*/, double /*t*/) { return x[0]; });
    comp.add_derived(
        "b",
        [](const std::vector<double>& x, const std::vector<double>& /*u*/,
           const std::vector<double>& /*d*/, double /*t*/) { return x[0] * 2.0; });
    comp.set_dynamics(
        [](const std::vector<double>& /*x*/, const std::vector<double>& /*u*/,
           const std::vector<double>& d, double /*t*/) {
            return std::vector<double>{ d[0] + d[1] };  // a + b
        });
    composed.add_component(std::move(comp));
    composed.set_mesh(0.0, 1.0, 4);

    auto derived_a = [](const auto& x, const auto& /*u*/,
                         const auto& /*d*/, auto /*t*/) { return x[0]; };
    auto derived_b = [](const auto& x, const auto& /*u*/,
                         const auto& /*d*/, auto /*t*/) {
        return x[0] * decltype(x[0])(2.0);
    };
    auto dyn_lambda = [](const auto& /*x*/, const auto& /*u*/,
                          const auto& d, auto /*t*/) {
        using T = typename std::decay_t<decltype(d)>::value_type;
        return std::vector<T>{ d[0] + d[1] };
    };
    auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                           const auto& /*d*/, auto /*t*/) { return double(0.0); };

    auto ocp = composed.build(
        goss::model::make_derived_exprs(derived_a, derived_b),
        goss::model::make_component_dyns(dyn_lambda),
        cost_lambda);

    std::vector<double> x_test{ 4.0 }, u_test{};
    auto dx = ocp.dynamics(x_test, u_test, 0.0);
    ASSERT_EQ(dx.size(), 1u);
    EXPECT_DOUBLE_EQ(dx[0], 12.0);  // a=4, b=8, dx=12
}

// ---- I-1 regression: topo_ordered_derived_names() reflects dependency order ----

// A model where insertion order of derived quantities in the component differs from
// topological (dependency-first) order. Specifically: component registers 'b' first,
// then 'a', but declares that 'b' depends on 'a'. Kahn's algorithm must produce the
// topo order [a, b], regardless of the insertion order [b, a].
//
// This test guards against the silent-wrong-result footgun described in I-1: if a
// caller passes make_derived_exprs(lambda_b, lambda_a) — matching insertion order —
// instead of make_derived_exprs(lambda_a, lambda_b) — matching topo order — the I2
// count guard passes but the wrong lambda is wired to each dependency set.
// topo_ordered_derived_names() lets the caller verify the required order before build().
TEST(ComposedModelVariadic, TopoOrderedDerivedNamesReflectsDependencyOrder) {
    // 'b' is declared first in the component, but 'b' depends on 'a', so 'a' must
    // appear first in the topological order returned by topo_ordered_derived_names().
    goss::model::ComposedModel composed;

    goss::model::Component comp("c");
    comp.add_state("x");
    // 'b' is registered before 'a' in insertion order, but declares a dependency on 'a'.
    // insert order: b=0, a=1; topo order must be: a=0, b=1.
    comp.input_derived("a");  // b depends on a — must be declared before add_derived("b")
    comp.add_derived(
        "b",
        [](const auto& /*x*/, const auto& /*u*/, const auto& d, double /*t*/) {
            return d[0] * 2.0;  // d[0] = a (dependency)
        });
    comp.add_derived(
        "a",
        [](const auto& x, const auto& /*u*/, const auto& /*d*/, double /*t*/) {
            return x[0];
        });
    comp.set_dynamics(
        [](const std::vector<double>& /*x*/, const std::vector<double>& /*u*/,
           const std::vector<double>& /*d*/, double /*t*/) {
            return std::vector<double>{ 0.0 };
        });
    composed.add_component(std::move(comp));

    const std::vector<std::string> topo_names = composed.topo_ordered_derived_names();

    // Dependency 'a' must precede dependent 'b' in topo order, regardless of
    // insertion order ('b' was inserted before 'a' in the component above).
    ASSERT_EQ(topo_names.size(), 2u);
    EXPECT_EQ(topo_names[0], "a");  // dependency first
    EXPECT_EQ(topo_names[1], "b");  // dependent second
}

// ---- Task 4 (dae-algebraic-variables): ComposedModel algebraic metadata collection tests ----

TEST(ComposedModelAlgebraic, TotalNumAlgebraicSumsComponents) {
    goss::model::ComposedModel composed;
    goss::model::Component component_a("comp_a");
    component_a.add_state("x");
    auto alg_fn_a = [](const std::vector<double>&, const std::vector<double>&,
                       const std::vector<double>&, double) -> double { return 0.0; };
    component_a.add_algebraic("z1", alg_fn_a, -1e19, 1e19);
    // Only component_a (with 1 algebraic variable) is registered.
    // total_num_algebraic() must return 1.
    composed.add_component(std::move(component_a));
    EXPECT_EQ(composed.total_num_algebraic(), 1u);
}

TEST(ComposedModelAlgebraic, TotalNumAlgebraicZeroWhenNoneRegistered) {
    goss::model::ComposedModel composed;
    goss::model::Component component("comp");
    component.add_state("position");
    composed.add_component(std::move(component));
    EXPECT_EQ(composed.total_num_algebraic(), 0u);
}

TEST(ComposedModelMultiDerived, ThreeDerivedLinearChainEvaluatesCorrectly) {
    // Chain: c depends on b which depends on a.
    // a = 1.0 (constant); b = a * 2.0 = 2.0; c = b + a = 3.0.
    // dx/dt = c = 3.0.
    goss::model::ComposedModel composed;

    goss::model::Component comp("c");
    comp.add_state("x");
    comp.add_derived(
        "a",
        [](const std::vector<double>& /*x*/, const std::vector<double>& /*u*/,
           const std::vector<double>& /*d*/, double /*t*/) { return 1.0; });
    comp.input_derived("a");
    comp.add_derived(
        "b",
        [](const std::vector<double>& /*x*/, const std::vector<double>& /*u*/,
           const std::vector<double>& d, double /*t*/) { return d[0] * 2.0; });
    comp.input_derived("a");
    comp.input_derived("b");
    comp.add_derived(
        "c",
        [](const std::vector<double>& /*x*/, const std::vector<double>& /*u*/,
           const std::vector<double>& d, double /*t*/) { return d[0] + d[1]; });
    comp.set_dynamics(
        [](const std::vector<double>& /*x*/, const std::vector<double>& /*u*/,
           const std::vector<double>& d, double /*t*/) {
            return std::vector<double>{ d[2] };  // dx/dt = c (topo index 2)
        });
    composed.add_component(std::move(comp));
    composed.set_mesh(0.0, 1.0, 3);

    auto derived_a = [](const auto& /*x*/, const auto& /*u*/,
                         const auto& /*d*/, auto /*t*/) { return double(1.0); };
    auto derived_b = [](const auto& /*x*/, const auto& /*u*/,
                         const auto& d, auto /*t*/) {
        return d[0] * decltype(d[0])(2.0);  // d[0]=a in deps_so_far
    };
    // c depends on a (topo 0) and b (topo 1); deps_so_far = [a, b]
    auto derived_c = [](const auto& /*x*/, const auto& /*u*/,
                         const auto& d, auto /*t*/) {
        using T = typename std::decay_t<decltype(d)>::value_type;
        return d[0] + d[1];  // d[0]=a, d[1]=b in deps_so_far
    };
    auto dyn_lambda = [](const auto& /*x*/, const auto& /*u*/,
                          const auto& d, auto /*t*/) {
        using T = typename std::decay_t<decltype(d)>::value_type;
        return std::vector<T>{ d[2] };  // d[2]=c in the full deriveds vector
    };
    auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                           const auto& /*d*/, auto /*t*/) { return double(0.0); };

    auto ocp = composed.build(
        goss::model::make_derived_exprs(derived_a, derived_b, derived_c),
        goss::model::make_component_dyns(dyn_lambda),
        cost_lambda);

    std::vector<double> x_test{ 0.0 }, u_test{};
    auto dx = ocp.dynamics(x_test, u_test, 0.0);
    ASSERT_EQ(dx.size(), 1u);
    EXPECT_DOUBLE_EQ(dx[0], 3.0);  // a=1, b=2, c=3
}
