// tests/sim/test_trajectory.cpp
#include <gtest/gtest.h>
#include "goss/sim/trajectory.hpp"
#include "goss/model/model.hpp"
#include "goss/transcription/variable_layout.hpp"
#include "goss/solver/solver_result.hpp"

namespace {
goss::model::Model two_state_model() {
    goss::model::Model model;
    model.add_state("pos");
    model.add_state("vel");
    model.add_control("thrust");
    model.set_mesh(0.0, 2.0, 2);   // 3 nodes, width 1.0
    return model;
}
}  // namespace

TEST(Trajectory, ExtractsNamedSeriesAndTimes) {
    auto model = two_state_model();
    goss::transcription::VariableLayout layout(2, 1, 3);
    goss::transcription::Mesh mesh{0.0, 2.0, 2};
    goss::solver::SolverResult result;
    result.status = goss::solver::SolverStatus::Success;
    result.x.assign(layout.total_variables(), 0.0);
    // node k: pos=k, vel=10+k, thrust=100+k
    for (std::size_t k = 0; k < 3; ++k) {
        result.x[layout.state_index(k, 0)] = static_cast<double>(k);
        result.x[layout.state_index(k, 1)] = 10.0 + k;
        result.x[layout.control_index(k, 0)] = 100.0 + k;
    }
    auto traj = goss::sim::extract_trajectory(result, layout, model, mesh);
    ASSERT_EQ(traj.times.size(), 3u);
    EXPECT_DOUBLE_EQ(traj.times[1], 1.0);
    EXPECT_DOUBLE_EQ(traj.state("pos")[2], 2.0);
    EXPECT_DOUBLE_EQ(traj.state("vel")[0], 10.0);
    EXPECT_DOUBLE_EQ(traj.control("thrust")[2], 102.0);
}

TEST(Trajectory, RejectsWrongResultSize) {
    auto model = two_state_model();
    goss::transcription::VariableLayout layout(2, 1, 3);
    goss::transcription::Mesh mesh{0.0, 2.0, 2};
    goss::solver::SolverResult result;
    result.x.assign(3, 0.0);   // wrong size
    EXPECT_THROW(goss::sim::extract_trajectory(result, layout, model, mesh), goss::sim::SimError);
}

TEST(Trajectory, StateLookupUnknownNameThrows) {
    auto model = two_state_model();
    goss::transcription::VariableLayout layout(2, 1, 3);
    goss::transcription::Mesh mesh{0.0, 2.0, 2};
    goss::solver::SolverResult result;
    result.x.assign(layout.total_variables(), 0.0);
    auto traj = goss::sim::extract_trajectory(result, layout, model, mesh);
    EXPECT_THROW(traj.state("nonexistent"), goss::sim::SimError);
}
