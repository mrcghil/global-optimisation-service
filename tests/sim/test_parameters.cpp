// tests/sim/test_parameters.cpp
#include <gtest/gtest.h>
#include <string>
#include "goss/model/model.hpp"
#include "goss/transcription/trapezoidal.hpp"
#include "goss/sim/parameters.hpp"
#include "goss/model/errors.hpp"

TEST(SimParameters, ValidateThenBindAcceptsInRange) {
    goss::model::Model model;
    auto x = model.add_state("x");
    model.add_parameter("arrival_rate", 1.0, 0.0, 10.0);
    model.set_initial_state(x, 0.0);
    model.set_mesh(0.0, 1.0, 4);
    auto ocp = model.build(
        [](const auto& xx, const auto&, const auto& p, auto){ using T=std::decay_t<decltype(xx[0])>; return std::vector<T>{p[0]-xx[0]}; },
        [](const auto&, const auto&, const auto&, auto t){ using T=std::decay_t<decltype(t)>; return T(0); });
    auto compiled = goss::transcription::Trapezoidal::compile(ocp, "sim_param_ok");

    ASSERT_EQ(compiled.validator.size(), 1u);
    EXPECT_NO_THROW(goss::sim::apply_parameters(*compiled.problem, compiled.validator, {3.0}));
}

TEST(SimParameters, ValidateThenBindRejectsOutOfRangeBeforeTouchingSolver) {
    goss::model::Model model;
    auto x = model.add_state("x");
    model.add_parameter("arrival_rate", 1.0, 0.0, 10.0);
    model.set_initial_state(x, 0.0);
    model.set_mesh(0.0, 1.0, 4);
    auto ocp = model.build(
        [](const auto& xx, const auto&, const auto& p, auto){ using T=std::decay_t<decltype(xx[0])>; return std::vector<T>{p[0]-xx[0]}; },
        [](const auto&, const auto&, const auto&, auto t){ using T=std::decay_t<decltype(t)>; return T(0); });
    auto compiled = goss::transcription::Trapezoidal::compile(ocp, "sim_param_bad");

    try {
        goss::sim::apply_parameters(*compiled.problem, compiled.validator, {50.0});
        FAIL() << "expected ModelError";
    } catch (const goss::model::ModelError& error) {
        EXPECT_NE(std::string(error.what()).find("arrival_rate"), std::string::npos);
    }
}
