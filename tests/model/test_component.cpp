#include <gtest/gtest.h>
#include "goss/model/errors.hpp"
#include "goss/model/component.hpp"

TEST(ComponentError, IsThrowable) {
    EXPECT_THROW(
        throw goss::model::ComponentError("boom"),
        goss::model::ComponentError);
}

TEST(Component, ConstructionAndNameAccess) {
    goss::model::Component component("service");
    EXPECT_EQ(component.component_name(), "service");
    EXPECT_EQ(component.num_owned_states(), 0u);
    EXPECT_EQ(component.num_derived(), 0u);
}

TEST(Component, AddOwnedStateAssignsLocalIndex) {
    goss::model::Component component("c");
    auto state_handle = component.add_state("queue_length");
    EXPECT_EQ(state_handle.index, 0u);
    EXPECT_EQ(component.num_owned_states(), 1u);
}

TEST(Component, InputStateHasUnresolvedIndex) {
    goss::model::Component component("c");
    auto input_handle = component.input_state("some_state");
    EXPECT_EQ(input_handle.index, goss::model::kUnresolvedIndex);
    EXPECT_EQ(component.input_state_names().size(), 1u);
    EXPECT_EQ(component.input_state_names()[0], "some_state");
}

TEST(Component, AddDerivedReturnsLocalIndex) {
    goss::model::Component component("c");
    auto derived_handle = component.add_derived(
        "service_rate",
        [](const auto& /*x*/, const auto& /*u*/,
           const auto& /*d*/, double /*t*/) { return 0.0; });
    EXPECT_EQ(derived_handle.index, 0u);
    EXPECT_EQ(component.num_derived(), 1u);
}

TEST(Component, DuplicateNameWithinComponentThrows) {
    goss::model::Component component("c");
    component.add_state("x");
    EXPECT_THROW(component.add_state("x"), goss::model::ComponentError);
    EXPECT_THROW(component.add_derived("x", [](const auto&, const auto&, const auto&, double) { return 0.0; }),
                 goss::model::ComponentError);
}

TEST(Component, EvaluateDynamicsDoublePath) {
    goss::model::Component component("queue");
    component.add_state("q");
    component.set_dynamics(
        [](const std::vector<double>& x,
           const std::vector<double>& /*u*/,
           const std::vector<double>& /*d*/,
           double /*t*/) {
            return std::vector<double>{ 3.0 - x[0] };
        });
    std::vector<double> x{2.0}, u{}, d{};
    auto result = component.evaluate_dynamics(x, u, d, 0.0);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_DOUBLE_EQ(result[0], 1.0);  // 3 - 2 = 1
}
