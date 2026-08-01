// tests/bench/test_bench_flagship.cpp
//
// Flagship benchmark: exp-decay problem across {Trapezoidal, HermiteSimpson}
//   x {IpoptSolver, NloptSolver}.
// Asserts: correct labels, non-negative timing, non-zero NVars, and (for IPOPT)
// near-zero objective and small RK4 validation error.
// DOES NOT assert exact timing values — timing is non-deterministic.
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include "goss/bench/harness.hpp"
#include "goss/bench/report.hpp"
#include "goss/model/model.hpp"
#include "goss/transcription/trapezoidal.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "goss/solver/nlopt_solver.hpp"
#include "goss/solver/solver.hpp"

namespace {
// dx/dt = -x, x(0) = 1, cost = 0. Analytic: x(t) = exp(-t).
// Objective value is 0 (zero running cost); validation error for a correct
// collocation solution should be small relative to the scheme order.
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

TEST(BenchFlagship, TwoSchemesTwoSolversProduceFourResults) {
    constexpr double kTimeHorizon   = 1.0;
    constexpr std::size_t kIntervals = 20;  // enough for both schemes to converge cleanly
    constexpr double kValidationTol  = 1e-3; // max acceptable RK4 deviation for IPOPT results
    constexpr double kObjectiveTol   = 1e-6; // zero-cost problem; objective should be ~0

    goss::model::Model model;
    auto x_state = model.add_state("x");
    model.set_initial_state(x_state, 1.0);
    model.set_mesh(0.0, kTimeHorizon, kIntervals);
    auto ocp = model.build(ExpDecayDyn{}, ZeroCostFn{});

    // Build solver instances; keep them alive for the duration of the test.
    goss::solver::IpoptSolver ipopt_solver;
    goss::solver::NloptSolver nlopt_solver;
    std::vector<goss::solver::Solver*> solvers = {&ipopt_solver, &nlopt_solver};
    std::vector<std::string> solver_names      = {"IpoptSolver", "NloptSolver"};

    // Run both schemes. Scheme axis is unrolled at compile time (see harness.hpp).
    auto trapezoidal_results = goss::bench::run_scheme<goss::transcription::Trapezoidal>(
        ocp, model, "Trapezoidal", "flagship_trap", solvers, solver_names);
    auto hermite_simpson_results = goss::bench::run_scheme<goss::transcription::HermiteSimpson>(
        ocp, model, "HermiteSimpson", "flagship_hs", solvers, solver_names);

    // Combine into a single results table.
    std::vector<goss::bench::BenchmarkResult> all_results = trapezoidal_results;
    all_results.insert(all_results.end(),
                       hermite_simpson_results.begin(), hermite_simpson_results.end());

    // --- Structural assertions (scheme/solver count, labels, non-negative timing) ---
    ASSERT_EQ(all_results.size(), std::size_t{4})
        << "Expected 2 schemes x 2 solvers = 4 results";

    // Verify scheme labels.
    const auto count_scheme = [&](const std::string& name) {
        return std::count_if(all_results.begin(), all_results.end(),
                             [&](const goss::bench::BenchmarkResult& r) {
                                 return r.scheme_name == name; });
    };
    EXPECT_EQ(count_scheme("Trapezoidal"),   std::size_t{2});
    EXPECT_EQ(count_scheme("HermiteSimpson"), std::size_t{2});

    // Verify solver labels.
    const auto count_solver = [&](const std::string& name) {
        return std::count_if(all_results.begin(), all_results.end(),
                             [&](const goss::bench::BenchmarkResult& r) {
                                 return r.solver_name == name; });
    };
    EXPECT_EQ(count_solver("IpoptSolver"), std::size_t{2});
    EXPECT_EQ(count_solver("NloptSolver"), std::size_t{2});

    // All timing values must be non-negative (they are wall-clock durations >= 0).
    for (const auto& result : all_results) {
        EXPECT_GE(result.elapsed_seconds, 0.0)
            << "Negative elapsed_seconds for " << result.scheme_name
            << " + " << result.solver_name;
    }

    // All num_variables must be positive (any compiled OCP has at least 1 variable).
    for (const auto& result : all_results) {
        EXPECT_GT(result.num_variables, std::size_t{0})
            << "Zero num_variables for " << result.scheme_name
            << " + " << result.solver_name;
    }

    // --- Accuracy assertions for IPOPT results only ---
    // IPOPT is gradient-based and converges reliably on this well-posed problem.
    // NloptSolver (COBYLA, derivative-free) may not converge within default limits;
    // we do NOT require NloptSolver to succeed — the harness records the status gracefully.
    for (const auto& result : all_results) {
        if (result.solver_name == "IpoptSolver") {
            EXPECT_EQ(result.solve_status, goss::solver::SolverStatus::Success)
                << "IPOPT must converge on the exp-decay problem for scheme "
                << result.scheme_name;
            EXPECT_NEAR(result.objective_value, 0.0, kObjectiveTol)
                << "Zero-cost problem: objective must be ~0 for " << result.scheme_name;
            EXPECT_LT(result.validation_error, kValidationTol)
                << "RK4 validation error too large for IPOPT + " << result.scheme_name;
        }
    }

    // --- Reporting smoke-test (no solver calls; just serialize and check structure) ---
    const std::string table_text = goss::bench::to_table(all_results);
    EXPECT_NE(table_text.find("Trapezoidal"),    std::string::npos);
    EXPECT_NE(table_text.find("HermiteSimpson"), std::string::npos);
    EXPECT_NE(table_text.find("IpoptSolver"),    std::string::npos);

    const std::string csv_text = goss::bench::to_csv(all_results);
    // Header starts with "scheme".
    EXPECT_EQ(csv_text.find("scheme"), std::size_t{0});
    // 1 header + 4 data rows = 5 newlines.
    EXPECT_EQ(std::count(csv_text.begin(), csv_text.end(), '\n'), std::ptrdiff_t{5});
}
