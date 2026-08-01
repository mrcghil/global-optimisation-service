// tests/bench/test_harness_unit.cpp
//
// Tests run_scheme<Trapezoidal> on the analytic exp-decay problem.
// Asserts STRUCTURE (labels, counts, non-negative timing) — NOT exact timing values.
// Accuracy is separately asserted in the flagship test (test_bench_flagship.cpp).
#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include "goss/bench/harness.hpp"
#include "goss/model/model.hpp"
#include "goss/transcription/trapezoidal.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "goss/solver/solver.hpp"

namespace {
// dx/dt = -x, x(0) = 1, cost = integral(0). Analytic: x(t) = exp(-t).
// Used for structure tests — small mesh (10 intervals) to keep solve fast.
struct ExpDecayDyn {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x_vec, const std::vector<T>&, T) const {
        return { -x_vec[0] };
    }
};
struct ZeroCostFn {
    template <typename T>
    T operator()(const std::vector<T>&, const std::vector<T>&, T) const { return T(0); }
};
}  // namespace

TEST(RunScheme, RejectsEmptySolverList) {
    goss::model::Model model;
    auto x = model.add_state("x");
    model.set_initial_state(x, 1.0);
    model.set_mesh(0.0, 1.0, 5);
    auto ocp = model.build(ExpDecayDyn{}, ZeroCostFn{});

    std::vector<goss::solver::Solver*> empty_solvers;
    EXPECT_THROW(
        goss::bench::run_scheme<goss::transcription::Trapezoidal>(
            ocp, model, "Trapezoidal", "trap_empty", empty_solvers, {}),
        goss::bench::BenchError);
}

TEST(RunScheme, RejectsMismatchedNameVectorSize) {
    goss::model::Model model;
    auto x = model.add_state("x");
    model.set_initial_state(x, 1.0);
    model.set_mesh(0.0, 1.0, 5);
    auto ocp = model.build(ExpDecayDyn{}, ZeroCostFn{});

    goss::solver::IpoptSolver ipopt;
    std::vector<goss::solver::Solver*> solvers = {&ipopt};
    std::vector<std::string> wrong_names = {"A", "B"};  // 2 names for 1 solver
    EXPECT_THROW(
        goss::bench::run_scheme<goss::transcription::Trapezoidal>(
            ocp, model, "Trapezoidal", "trap_mismatch", solvers, wrong_names),
        goss::bench::BenchError);
}

TEST(RunScheme, ProducesOneResultPerSolver) {
    goss::model::Model model;
    auto x = model.add_state("x");
    model.set_initial_state(x, 1.0);
    model.set_mesh(0.0, 1.0, 10);
    auto ocp = model.build(ExpDecayDyn{}, ZeroCostFn{});

    goss::solver::IpoptSolver ipopt;
    std::vector<goss::solver::Solver*> solvers = {&ipopt};
    std::vector<std::string> names = {"IpoptSolver"};

    auto results = goss::bench::run_scheme<goss::transcription::Trapezoidal>(
        ocp, model, "Trapezoidal", "trap_unit", solvers, names);

    ASSERT_EQ(results.size(), std::size_t{1});
}

TEST(RunScheme, ResultLabelsMatchInputArguments) {
    goss::model::Model model;
    auto x = model.add_state("x");
    model.set_initial_state(x, 1.0);
    model.set_mesh(0.0, 1.0, 10);
    auto ocp = model.build(ExpDecayDyn{}, ZeroCostFn{});

    goss::solver::IpoptSolver ipopt;
    std::vector<goss::solver::Solver*> solvers = {&ipopt};

    auto results = goss::bench::run_scheme<goss::transcription::Trapezoidal>(
        ocp, model, "Trapezoidal", "trap_labels", solvers, {"IpoptSolver"});

    ASSERT_EQ(results.size(), std::size_t{1});
    EXPECT_EQ(results[0].scheme_name, "Trapezoidal");
    EXPECT_EQ(results[0].solver_name, "IpoptSolver");
}

TEST(RunScheme, ElapsedSecondsIsNonNegative) {
    // Wall-clock time is non-deterministic; we only assert it is non-negative.
    // Never assert an exact value or a tight upper bound here.
    goss::model::Model model;
    auto x = model.add_state("x");
    model.set_initial_state(x, 1.0);
    model.set_mesh(0.0, 1.0, 10);
    auto ocp = model.build(ExpDecayDyn{}, ZeroCostFn{});

    goss::solver::IpoptSolver ipopt;
    auto results = goss::bench::run_scheme<goss::transcription::Trapezoidal>(
        ocp, model, "Trapezoidal", "trap_timing", {&ipopt}, {"IpoptSolver"});

    ASSERT_EQ(results.size(), std::size_t{1});
    EXPECT_GE(results[0].elapsed_seconds, 0.0);
}

TEST(RunScheme, NumVariablesIsPositive) {
    goss::model::Model model;
    auto x = model.add_state("x");
    model.set_initial_state(x, 1.0);
    model.set_mesh(0.0, 1.0, 10);
    auto ocp = model.build(ExpDecayDyn{}, ZeroCostFn{});

    goss::solver::IpoptSolver ipopt;
    auto results = goss::bench::run_scheme<goss::transcription::Trapezoidal>(
        ocp, model, "Trapezoidal", "trap_nvars", {&ipopt}, {"IpoptSolver"});

    ASSERT_EQ(results.size(), std::size_t{1});
    EXPECT_GT(results[0].num_variables, std::size_t{0});
}
