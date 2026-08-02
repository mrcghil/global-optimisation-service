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

TEST(ComponentAlgebraic, AddAlgebraicStoresEntry) {
    goss::model::Component component("test_comp");
    // Validation lambda: the algebraic residual g(x, u, z_alg, t) = z_alg[0] - x[0]*x[0]
    // Enforces z_alg[0] = x[0]^2 when the solver drives g to zero.
    auto residual_validation_fn = [](
            const std::vector<double>& x,
            const std::vector<double>& /*u*/,
            const std::vector<double>& alg_vars,
            double /*t*/) -> double {
        return alg_vars[0] - x[0] * x[0];
    };
    auto handle = component.add_algebraic("x_squared", residual_validation_fn, 0.0, 1e19);
    EXPECT_EQ(component.num_algebraic(), 1u);
    EXPECT_EQ(static_cast<std::size_t>(handle), 0u);
    const auto& entries = component.algebraic_entries();
    ASSERT_EQ(entries.size(), 1u);
    EXPECT_EQ(entries[0].name, "x_squared");
    EXPECT_DOUBLE_EQ(entries[0].lower_bound, 0.0);
    EXPECT_DOUBLE_EQ(entries[0].upper_bound, 1e19);
}

TEST(ComponentAlgebraic, DuplicateAlgebraicNameThrows) {
    goss::model::Component component("test_comp");
    auto first_fn = [](const std::vector<double>&, const std::vector<double>&,
                       const std::vector<double>&, double) -> double { return 0.0; };
    auto second_fn = [](const std::vector<double>&, const std::vector<double>&,
                        const std::vector<double>&, double) -> double { return 0.0; };
    component.add_algebraic("my_alg", first_fn, -1e19, 1e19);
    EXPECT_THROW(component.add_algebraic("my_alg", second_fn, -1e19, 1e19),
                 goss::model::ComponentError);
}

TEST(ComponentAlgebraic, DuplicateNameWithStateThrows) {
    goss::model::Component component("test_comp");
    component.add_state("position");
    auto residual_fn = [](const std::vector<double>&, const std::vector<double>&,
                          const std::vector<double>&, double) -> double { return 0.0; };
    // "position" is already registered as a state — should throw.
    EXPECT_THROW(component.add_algebraic("position", residual_fn, -1e19, 1e19),
                 goss::model::ComponentError);
}

TEST(ComponentAlgebraic, EvaluateAlgebraicResidualInvokesValidationLambda) {
    goss::model::Component component("test_comp");
    // g(x, u, z_alg, t) = z_alg[0] - 2.0 * x[0]
    auto residual_fn = [](const std::vector<double>& x,
                          const std::vector<double>& /*u*/,
                          const std::vector<double>& alg_vars,
                          double /*t*/) -> double {
        return alg_vars[0] - 2.0 * x[0];
    };
    component.add_algebraic("twice_x", residual_fn, -1e19, 1e19);
    // At x[0]=3.0, alg_vars[0]=6.0: residual = 6.0 - 2*3.0 = 0.0 (satisfied)
    std::vector<double> x{3.0};
    std::vector<double> u{};
    std::vector<double> alg_vars{6.0};
    double residual = component.evaluate_algebraic_residual(0, x, u, alg_vars, 0.0);
    EXPECT_DOUBLE_EQ(residual, 0.0);
    // At alg_vars[0]=5.0: residual = 5.0 - 6.0 = -1.0 (not satisfied)
    alg_vars[0] = 5.0;
    residual = component.evaluate_algebraic_residual(0, x, u, alg_vars, 0.0);
    EXPECT_DOUBLE_EQ(residual, -1.0);
}

TEST(ComponentAlgebraic, NumAlgebraicIsZeroByDefault) {
    goss::model::Component component("test_comp");
    component.add_state("x");
    EXPECT_EQ(component.num_algebraic(), 0u);
}
