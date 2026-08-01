// tests/bench/test_bench_workflow.cpp
//
// Integration test: run_scheme → to_table + to_csv → verify data flows end to end.
// Tests the reporting pipeline with real solver output, complementing the PURE
// reporting tests in test_report.cpp (which use synthetic data only).
#include <gtest/gtest.h>
#include <algorithm>
#include <string>
#include <vector>
#include "goss/bench/harness.hpp"
#include "goss/bench/report.hpp"
#include "goss/model/model.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "goss/solver/solver.hpp"

namespace {
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

TEST(BenchWorkflow, HarnessResultsFlowIntoReporter) {
    goss::model::Model model;
    auto x_state = model.add_state("x");
    model.set_initial_state(x_state, 1.0);
    model.set_mesh(0.0, 1.0, 10);
    auto ocp = model.build(ExpDecayDyn{}, ZeroCostFn{});

    goss::solver::IpoptSolver ipopt;
    std::vector<goss::solver::Solver*> solvers  = {&ipopt};
    std::vector<std::string> solver_names        = {"IpoptSolver"};

    // Run one scheme (HermiteSimpson chosen for its higher-order accuracy).
    auto results = goss::bench::run_scheme<goss::transcription::HermiteSimpson>(
        ocp, model, "HermiteSimpson", "workflow_hs", solvers, solver_names);
    ASSERT_EQ(results.size(), std::size_t{1});

    // Table serializes without throwing and contains the scheme label.
    std::string table_text = goss::bench::to_table(results);
    EXPECT_NE(table_text.find("HermiteSimpson"), std::string::npos);
    EXPECT_NE(table_text.find("IpoptSolver"),    std::string::npos);

    // CSV header is present; data row carries the correct scheme label.
    std::string csv_text = goss::bench::to_csv(results);
    EXPECT_EQ(csv_text.find("scheme"), std::size_t{0});
    EXPECT_NE(csv_text.find("HermiteSimpson"), std::string::npos);
    // 1 header + 1 data row = 2 newlines.
    EXPECT_EQ(std::count(csv_text.begin(), csv_text.end(), '\n'), std::ptrdiff_t{2});

    // Status name round-trips correctly through to_csv output.
    EXPECT_NE(csv_text.find("Success"), std::string::npos);
}
