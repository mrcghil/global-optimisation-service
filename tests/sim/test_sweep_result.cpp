// tests/sim/test_sweep_result.cpp
#include <gtest/gtest.h>
#include "goss/sim/sweep_result.hpp"

TEST(SweepResult, CountsSucceededPoints) {
    goss::sim::SweepResult result;
    goss::sim::SweepPoint a; a.status = goss::solver::SolverStatus::Success;
    goss::sim::SweepPoint b; b.status = goss::solver::SolverStatus::IterationLimit;
    goss::sim::SweepPoint c; c.status = goss::solver::SolverStatus::Success;
    result.points = {a, b, c};
    EXPECT_EQ(result.num_succeeded(), 2u);
}
