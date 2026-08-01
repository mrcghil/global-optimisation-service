// tests/bench/test_benchmark_result.cpp
#include <gtest/gtest.h>
#include <string>
#include "goss/bench/errors.hpp"
#include "goss/bench/benchmark_result.hpp"
#include "goss/solver/solver_result.hpp"

TEST(BenchError, IsThrowableAsRuntimeError) {
    EXPECT_THROW(throw goss::bench::BenchError("boom"), goss::bench::BenchError);
    EXPECT_THROW(throw goss::bench::BenchError("boom"), std::runtime_error);
}

TEST(BenchmarkResult, DefaultConstructsToSensibleValues) {
    goss::bench::BenchmarkResult result;
    EXPECT_TRUE(result.scheme_name.empty());
    EXPECT_TRUE(result.solver_name.empty());
    EXPECT_EQ(result.solve_status, goss::solver::SolverStatus::Failure);
    EXPECT_DOUBLE_EQ(result.objective_value, 0.0);
    EXPECT_GE(result.elapsed_seconds, 0.0);
    EXPECT_DOUBLE_EQ(result.validation_error, 0.0);
    EXPECT_EQ(result.num_variables, std::size_t{0});
}

TEST(BenchmarkResult, CanBePopulated) {
    goss::bench::BenchmarkResult result;
    result.scheme_name    = "Trapezoidal";
    result.solver_name    = "IpoptSolver";
    result.solve_status   = goss::solver::SolverStatus::Success;
    result.objective_value = 0.5;
    result.elapsed_seconds = 0.123;
    result.validation_error = 1e-5;
    result.num_variables   = 42;
    EXPECT_EQ(result.scheme_name, "Trapezoidal");
    EXPECT_EQ(result.num_variables, std::size_t{42});
    EXPECT_NEAR(result.elapsed_seconds, 0.123, 1e-9);
}

TEST(SolverStatusName, ReturnsNonEmptyStringForEveryStatus) {
    using goss::solver::SolverStatus;
    for (auto s : {SolverStatus::Success, SolverStatus::InfeasibleProblem,
                   SolverStatus::IterationLimit, SolverStatus::NumericalError,
                   SolverStatus::Failure}) {
        EXPECT_FALSE(goss::bench::solver_status_name(s).empty());
    }
}

TEST(SolverStatusName, SuccessNameContainsSuccess) {
    EXPECT_NE(
        goss::bench::solver_status_name(goss::solver::SolverStatus::Success).find("Success"),
        std::string::npos);
}
