// tests/sim/test_sweep_grid.cpp
#include <gtest/gtest.h>
#include "goss/sim/sweep.hpp"
#include "goss/sim/errors.hpp"

TEST(SweepGrid, CartesianProductRowMajor) {
    auto grid = goss::sim::make_grid({{1.0, 2.0}, {10.0, 20.0, 30.0}});
    ASSERT_EQ(grid.size(), 6u);                       // 2 * 3
    EXPECT_EQ(grid.front(), (std::vector<double>{1.0, 10.0}));
    EXPECT_EQ(grid[1],      (std::vector<double>{1.0, 20.0}));
    EXPECT_EQ(grid[3],      (std::vector<double>{2.0, 10.0}));  // param0 slowest
    EXPECT_EQ(grid.back(),  (std::vector<double>{2.0, 30.0}));
}

TEST(SweepGrid, RejectsEmptyAxis) {
    EXPECT_THROW(goss::sim::make_grid({{1.0}, {}}), goss::sim::SimError);
}
