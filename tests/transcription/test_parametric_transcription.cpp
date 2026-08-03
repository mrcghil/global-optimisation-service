// tests/transcription/test_parametric_transcription.cpp
// Unit tests for arity-dispatch helpers (Task 5: invoke.hpp) and
// end-to-end parametric transcription (Trapezoidal::compile with 4-arg dynamics).
#include <gtest/gtest.h>
#include <vector>
#include "goss/transcription/invoke.hpp"

namespace {
struct ThreeArgDyn {  // legacy: (x, u, t)
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x, const std::vector<T>&, T) const {
        return { -x[0] };
    }
};
struct FourArgDyn {   // parametric: (x, u, p, t)
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x, const std::vector<T>&,
                              const std::vector<T>& p, T) const {
        return { p[0] - x[0] };  // arrival_rate - x
    }
};
}  // namespace

TEST(TranscriptionInvoke, DispatchesToLegacyThreeArg) {
    std::vector<double> x{5.0}, u{}, p{2.0};
    auto out = goss::transcription::detail::call_dynamics(ThreeArgDyn{}, x, u, p, 0.0);
    EXPECT_DOUBLE_EQ(out[0], -5.0);   // ignores p
}

TEST(TranscriptionInvoke, DispatchesToParametricFourArg) {
    std::vector<double> x{5.0}, u{}, p{2.0};
    auto out = goss::transcription::detail::call_dynamics(FourArgDyn{}, x, u, p, 0.0);
    EXPECT_DOUBLE_EQ(out[0], 2.0 - 5.0);  // uses p
}

#include "goss/model/model.hpp"
#include "goss/transcription/trapezoidal.hpp"

TEST(ParametricTranscription, CompilesOnceAndTracksParameterAcrossEvals) {
    goss::model::Model model;
    auto x = model.add_state("x");
    auto rate = model.add_parameter("arrival_rate", /*default=*/1.0, 0.0, 10.0);
    (void)rate;
    model.set_initial_state(x, 0.0);
    model.set_mesh(0.0, 1.0, 5);

    // dx/dt = arrival_rate - x  (uses the 4-arg parametric overload)
    auto dynamics = [](const auto& xx, const auto&, const auto& p, auto t) {
        using T = std::decay_t<decltype(t)>;
        return std::vector<T>{ p[0] - xx[0] };
    };
    auto cost = [](const auto&, const auto&, const auto&, auto t) {
        using T = std::decay_t<decltype(t)>; return T(0);
    };
    auto ocp = model.build(dynamics, cost);
    auto compiled = goss::transcription::Trapezoidal::compile(ocp, "param_trap");

    ASSERT_EQ(compiled.problem->num_parameters(), 1u);
    // Objective is zero here; assert a defect constraint value shifts with p.
    std::vector<double> z(compiled.layout.total_variables(), 0.0);
    compiled.problem->set_parameters({1.0});
    auto g1 = compiled.problem->eval_constraints(z);
    compiled.problem->set_parameters({4.0});
    auto g2 = compiled.problem->eval_constraints(z);
    // First defect: x1 - x0 - (h/2)(f0+f1); at z=0, f = p - 0 = p, so g = -(h)*p.
    EXPECT_NE(g1.front(), g2.front());
}
