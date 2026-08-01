// tests/bench/test_report.cpp
#include <gtest/gtest.h>
#include <algorithm>
#include <string>
#include <vector>
#include "goss/bench/benchmark_result.hpp"
#include "goss/bench/report.hpp"
#include "goss/solver/solver_result.hpp"

namespace {
// Helper: build a synthetic BenchmarkResult with controlled, non-random values.
// elapsed_seconds is set to a fixed positive value so we can verify it appears in output
// without asserting its exact magnitude (which would be non-deterministic in a real run).
goss::bench::BenchmarkResult make_result(
        const std::string& scheme_name,
        const std::string& solver_name,
        goss::solver::SolverStatus status,
        double objective_value,
        double elapsed_seconds,
        double validation_error,
        std::size_t num_variables) {
    goss::bench::BenchmarkResult r;
    r.scheme_name     = scheme_name;
    r.solver_name     = solver_name;
    r.solve_status    = status;
    r.objective_value = objective_value;
    r.elapsed_seconds = elapsed_seconds;
    r.validation_error = validation_error;
    r.num_variables   = num_variables;
    return r;
}
}  // namespace

// --- to_table tests ---

TEST(ToTable, EmptyInputReturnsNonEmptyHeader) {
    // Even for zero results the table must have a header row.
    std::string table = goss::bench::to_table({});
    EXPECT_FALSE(table.empty());
    EXPECT_NE(table.find("Scheme"), std::string::npos);
    EXPECT_NE(table.find("Solver"), std::string::npos);
    EXPECT_NE(table.find("Status"), std::string::npos);
}

TEST(ToTable, OneRowContainsSchemeAndSolverLabels) {
    auto result = make_result("Trapezoidal", "IpoptSolver",
                              goss::solver::SolverStatus::Success,
                              /*obj=*/0.5, /*elapsed=*/0.123, /*val_err=*/1e-5, /*nvars=*/42);
    std::string table = goss::bench::to_table({result});
    EXPECT_NE(table.find("Trapezoidal"), std::string::npos);
    EXPECT_NE(table.find("IpoptSolver"), std::string::npos);
    EXPECT_NE(table.find("Success"),     std::string::npos);
}

TEST(ToTable, TwoRowsProduceTwoDataLines) {
    std::vector<goss::bench::BenchmarkResult> results = {
        make_result("Trapezoidal",   "IpoptSolver",  goss::solver::SolverStatus::Success, 0.5,  0.1, 1e-5, 40),
        make_result("HermiteSimpson","NloptSolver",  goss::solver::SolverStatus::Success, 0.51, 0.2, 2e-5, 80),
    };
    std::string table = goss::bench::to_table(results);
    EXPECT_NE(table.find("HermiteSimpson"), std::string::npos);
    EXPECT_NE(table.find("NloptSolver"),    std::string::npos);
}

// --- to_csv tests ---

TEST(ToCsv, HeaderRowHasAllExpectedColumns) {
    std::string csv = goss::bench::to_csv({});
    // Must start with the header.
    EXPECT_EQ(csv.find("scheme"), 0u);
    EXPECT_NE(csv.find("solver"),           std::string::npos);
    EXPECT_NE(csv.find("status"),           std::string::npos);
    EXPECT_NE(csv.find("objective"),        std::string::npos);
    EXPECT_NE(csv.find("elapsed_s"),        std::string::npos);
    EXPECT_NE(csv.find("validation_error"), std::string::npos);
    EXPECT_NE(csv.find("num_variables"),    std::string::npos);
    // Empty results: only the header row.
    EXPECT_EQ(std::count(csv.begin(), csv.end(), '\n'), 1);
}

TEST(ToCsv, TwoResultsProduceTwoDataRows) {
    std::vector<goss::bench::BenchmarkResult> results = {
        make_result("Trapezoidal",   "IpoptSolver", goss::solver::SolverStatus::Success, 0.5, 0.1, 1e-5, 40),
        make_result("HermiteSimpson","NloptSolver", goss::solver::SolverStatus::Success, 0.5, 0.2, 2e-5, 80),
    };
    std::string csv = goss::bench::to_csv(results);
    // 1 header + 2 data rows = 3 newlines.
    EXPECT_EQ(std::count(csv.begin(), csv.end(), '\n'), 3);
    EXPECT_NE(csv.find("Trapezoidal"),    std::string::npos);
    EXPECT_NE(csv.find("HermiteSimpson"), std::string::npos);
}

TEST(ToCsv, StatusNameIsHumanReadable) {
    auto result = make_result("S", "I", goss::solver::SolverStatus::IterationLimit,
                              0.0, 0.0, 0.0, 0);
    std::string csv = goss::bench::to_csv({result});
    EXPECT_NE(csv.find("IterationLimit"), std::string::npos);
}
