// tests/model/test_composed_model.cpp
// Name-resolution and global-index tests for ComposedModel (Task 3).
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
