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

    auto ocp = composed.build(derived_lambda, dyn_lambda, cost_lambda);
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

    // No derived quantities → build with 0 derived lambdas: use the 0-derived overload
    auto ocp = composed.build(dyn_lambda, cost_lambda);
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

    auto ocp = composed.build(derived_lambda, dyn_lambda, cost_lambda);

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
    EXPECT_THROW(composed.build(derived_lambda, dyn_lambda, cost_lambda),
                 goss::model::ComponentError);
}

// build() cost lambda sees real derived values, not zero-filled placeholder (Important #2).
// Fixture: derived d = 2 * x[0]; cost = d; so ocp.cost({x0}, {}, t) should equal 2 * x0.
TEST(ComposedModel, BuildCostSeesDerivedValue) {
    goss::model::ComposedModel composed;

    // "queue" component: owns state "q"
    goss::model::Component comp_queue("queue");
    comp_queue.add_state("q");
    comp_queue.set_dynamics(
        [](const std::vector<double>& /*x*/,
           const std::vector<double>& /*u*/,
           const std::vector<double>& /*d*/,
           double /*t*/) {
            return std::vector<double>{0.0};
        });

    composed.add_component(std::move(comp_queue));
    composed.set_mesh(0.0, 1.0, 4);

    // derived d[0] = 2 * x[0]
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

    auto ocp = composed.build(derived_lambda, dyn_lambda, cost_lambda);

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
        composed.build(dyn_lambda, cost_lambda),
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

    auto ocp = composed.build(derived_lambda, dyn_lambda, cost_lambda);
    // Evaluate assembled dynamics at x={3.0}, u={}, t=0.
    std::vector<double> x_test{3.0}, u_test{};
    auto dx = ocp.dynamics(x_test, u_test, 0.0);
    ASSERT_EQ(dx.size(), 1u);
    EXPECT_DOUBLE_EQ(dx[0], -6.0);  // -2 * 3.0
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
    EXPECT_THROW(composed.build(dyn_lambda, cost_lambda), goss::model::ModelError);
}
