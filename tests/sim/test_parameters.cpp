// tests/sim/test_parameters.cpp
#include <gtest/gtest.h>
#include <string>
#include "goss/model/errors.hpp"
#include "goss/model/model.hpp"
#include "goss/sim/initial_guess.hpp"
#include "goss/sim/parameters.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "goss/solver/solver_result.hpp"
#include "goss/transcription/trapezoidal.hpp"

// ---------------------------------------------------------------------------
// Test: valid parameters pass validation and the solver reaches a solution.
// ---------------------------------------------------------------------------
TEST(SimParameters, ValidateThenSolveAcceptsInRange) {
    goss::model::Model model;
    auto x = model.add_state("x");
    model.add_parameter("arrival_rate", 1.0, 0.0, 10.0);
    model.set_initial_state(x, 0.0);
    model.set_mesh(0.0, 1.0, 4);
    auto ocp = model.build(
        [](const auto& xx, const auto&, const auto& p, auto) {
            using T = std::decay_t<decltype(xx[0])>;
            return std::vector<T>{ p[0] - xx[0] };
        },
        [](const auto&, const auto&, const auto&, auto t) {
            using T = std::decay_t<decltype(t)>;
            return T(0);
        });
    auto compiled = goss::transcription::Trapezoidal::compile(ocp, "sim_param_ok");
    const auto z0 = goss::sim::linear_guess(model, compiled.layout);

    goss::solver::IpoptSolver solver;

    goss::solver::SolverResult result;
    // solve_with_parameters must not throw for a valid parameter value.
    ASSERT_NO_THROW(
        result = goss::sim::solve_with_parameters(
            solver, *compiled.problem, compiled.validator, z0, {3.0}));

    // The trivial model converges; assert actual solver success, not a tautology.
    EXPECT_EQ(result.status, goss::solver::SolverStatus::Success);
}

// ---------------------------------------------------------------------------
// Test: out-of-range parameters throw ModelError (naming "arrival_rate") before
// the solver is ever invoked — the throw comes from validator.validate().
// ---------------------------------------------------------------------------
TEST(SimParameters, ValidateThenSolveRejectsOutOfRangeBeforeTouchingSolver) {
    goss::model::Model model;
    auto x = model.add_state("x");
    model.add_parameter("arrival_rate", 1.0, 0.0, 10.0);
    model.set_initial_state(x, 0.0);
    model.set_mesh(0.0, 1.0, 4);
    auto ocp = model.build(
        [](const auto& xx, const auto&, const auto& p, auto) {
            using T = std::decay_t<decltype(xx[0])>;
            return std::vector<T>{ p[0] - xx[0] };
        },
        [](const auto&, const auto&, const auto&, auto t) {
            using T = std::decay_t<decltype(t)>;
            return T(0);
        });
    auto compiled = goss::transcription::Trapezoidal::compile(ocp, "sim_param_bad");
    const auto z0 = goss::sim::linear_guess(model, compiled.layout);

    goss::solver::IpoptSolver solver;

    // 50.0 is far outside [0,10] — validator.validate() must throw before
    // Solver::solve is ever called.
    try {
        goss::sim::solve_with_parameters(
            solver, *compiled.problem, compiled.validator, z0, {50.0});
        FAIL() << "expected ModelError for out-of-range arrival_rate";
    } catch (const goss::model::ModelError& error) {
        // The error message must explicitly name the offending parameter.
        EXPECT_NE(std::string(error.what()).find("arrival_rate"), std::string::npos);
    }
}
