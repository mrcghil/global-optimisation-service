// tests/solver/test_solver_parameters.cpp
// Integration tests: verify that solver parameters passed at solve-time are
// injected into the problem's AD backend before evaluation callbacks run.
//
// Model: parametric queue
//   state:   queue_length (q)
//   control: service_rate (u), bounds [0, 5]
//   parameter: arrival_rate (p), default 2.0, bounds [0, 10]
//   dynamics: dq/dt = p[0] - u[0]
//   running cost: q + 0.1 * u^2
//   initial queue: 10.0; mesh 0..5, 30 intervals
//
// The two tests share one compiled OCP (compile-once) to confirm that the
// same problem instance handles repeated solve-time parameter injections
// without recompilation.
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "goss/model/model.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "goss/sim/initial_guess.hpp"
#include "goss/ad/errors.hpp"

// ---------------------------------------------------------------------------
// Shared fixture: a GTest fixture that compiles the parametric-queue OCP once
// for all tests in the SolverParameters suite.
// ---------------------------------------------------------------------------

namespace {

/// Build a parametric-queue OCP model and return it together with a compiled
/// transcription.  Separated from the fixture so it can be called without
/// default-constructing CompiledOcp (which requires a default VariableLayout).
struct ParametricQueueFixture : public ::testing::Test {
    // Model object must outlive compiled because the OCP lambda captures nothing
    // heap-allocated from it, but keeping it alive is safer for future changes.
    goss::model::Model model;
    std::unique_ptr<goss::transcription::CompiledOcp> compiled_ptr;
    std::vector<double> z0;

    void SetUp() override {
        constexpr double kMaxRate = 5.0;

        const auto queue_state   = model.add_state("queue_length");
        const auto service_ctrl  = model.add_control("service_rate");
        const auto arrival_param = model.add_parameter(
            "arrival_rate", /*default_value=*/2.0, /*lb=*/0.0, /*ub=*/10.0);
        (void)arrival_param;

        model.set_state_bounds(queue_state, 0.0, 1e19);          // queue >= 0
        model.set_control_bounds(service_ctrl, 0.0, kMaxRate);
        model.set_initial_state(queue_state, 10.0);
        model.set_mesh(0.0, 5.0, 30);

        // dq/dt = arrival_rate - service_rate
        auto dynamics = [](const auto& x, const auto& u, const auto& p, auto) {
            using T = std::decay_t<decltype(x[0])>;
            return std::vector<T>{ p[0] - u[0] };
        };
        // running cost: queue_length + 0.1 * service_rate^2
        auto cost = [](const auto& x, const auto& u, const auto& /*p*/, auto) {
            using T = std::decay_t<decltype(x[0])>;
            return x[0] + T(0.1) * u[0] * u[0];
        };

        auto ocp = model.build(dynamics, cost);

        // COMPILE ONCE — both tests below reuse this compiled problem.
        compiled_ptr = std::make_unique<goss::transcription::CompiledOcp>(
            goss::transcription::HermiteSimpson::compile(
                ocp, "solver_param_test_queue"));

        z0 = goss::sim::linear_guess(model, compiled_ptr->layout);
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// A higher arrival rate makes the queue harder to drain, raising the cost.
/// Both solves use the SAME compiled problem via solve-time parameter injection.
TEST_F(ParametricQueueFixture, ParametersPassedAtSolveTimeChangeTheOptimum) {
    goss::solver::IpoptSolver solver;

    const auto low_arrival = solver.solve(
        *compiled_ptr->problem, z0, /*parameters=*/{1.0});
    ASSERT_EQ(low_arrival.status, goss::solver::SolverStatus::Success);

    const auto high_arrival = solver.solve(
        *compiled_ptr->problem, z0, /*parameters=*/{4.0});
    ASSERT_EQ(high_arrival.status, goss::solver::SolverStatus::Success);

    // Higher arrival rate => larger objective (queue harder to drain).
    EXPECT_GT(high_arrival.objective_value, low_arrival.objective_value);
}

/// Passing a parameter vector of the wrong length must propagate ADError
/// before any solver work starts (size mismatch detected at injection time).
TEST_F(ParametricQueueFixture, WrongParameterCountThrows) {
    goss::solver::IpoptSolver solver;

    // The model has exactly one parameter; passing two must throw ADError.
    EXPECT_THROW(
        solver.solve(*compiled_ptr->problem, z0, {1.0, 2.0}),
        goss::ad::ADError);
}
