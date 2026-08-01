// tests/accuracy/test_closed_form.cpp
// Minimal smoke test: just checks the header compiles and solve_and_extract_trajectory
// returns a trajectory for a trivial problem. Full closed-form tests follow in Task 2.
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "goss/model/model.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "accuracy/accuracy_helpers.hpp"

TEST(ClosedFormSmoke, TrajectoryExtractorReturnsCorrectNodeCount) {
    // Simplest possible OCP: dx/dt = u, x(0)=0, x(1)=1, min integral(u^2).
    // Used only to confirm the scaffold compiles and the helper returns sane data.
    const std::size_t num_intervals = 10;
    goss::model::Model model;
    const auto position_handle = model.add_state("position");
    const auto force_handle    = model.add_control("force");
    model.set_initial_state(position_handle, 0.0);
    model.set_final_state(position_handle, 1.0);
    model.set_mesh(0.0, 1.0, num_intervals);

    auto dynamics = [](const auto& state_vec, const auto& control_vec, auto /*time*/) {
        using ScalarT = typename std::decay_t<decltype(state_vec)>::value_type;
        return std::vector<ScalarT>{ control_vec[0] };
    };
    auto running_cost = [](const auto& /*state_vec*/, const auto& control_vec, auto /*time*/) {
        return control_vec[0] * control_vec[0];
    };

    auto ocp      = model.build(dynamics, running_cost);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "smoke_scaffold");
    const goss::accuracy::SolutionTrajectory trajectory =
        goss::accuracy::solve_and_extract_trajectory(compiled, /*initial_guess_value=*/0.5);

    // num_nodes = num_intervals + 1
    EXPECT_EQ(trajectory.states.size(), num_intervals + 1);
    EXPECT_EQ(trajectory.controls.size(), num_intervals + 1);
    EXPECT_EQ(trajectory.times.size(), num_intervals + 1);
}
