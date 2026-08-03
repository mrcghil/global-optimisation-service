// include/goss/sim/sweep_result.hpp
#pragma once
#include <cstddef>
#include <string>
#include <vector>
#include "goss/solver/solver_result.hpp"

namespace goss::sim {

struct SweepPoint {
    std::vector<double> parameters;
    goss::solver::SolverStatus status = goss::solver::SolverStatus::Failure;
    double objective_value = 0.0;
    std::vector<double> x;
    std::string message;
};

struct SweepResult {
    std::vector<SweepPoint> points;
    std::size_t num_succeeded() const {
        std::size_t count = 0;
        for (const SweepPoint& point : points)
            if (point.status == goss::solver::SolverStatus::Success) ++count;
        return count;
    }
};

}  // namespace goss::sim
