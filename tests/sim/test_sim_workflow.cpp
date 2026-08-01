// tests/sim/test_sim_workflow.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <string>
#include <vector>
#include "goss/model/model.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "goss/sim/initial_guess.hpp"
#include "goss/sim/trajectory.hpp"
#include "goss/sim/validation.hpp"
#include "goss/sim/diagnostics.hpp"

namespace {
struct DecayDyn {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x, const std::vector<T>&, T) const {
        return { -x[0] };
    }
};
struct ZeroCost {
    template <typename T>
    T operator()(const std::vector<T>&, const std::vector<T>&, T) const { return T(0); }
};
}  // namespace

TEST(SimWorkflow, FullChainFromGuessToDiagnosis) {
    goss::model::Model model;
    auto x = model.add_state("x");
    model.set_initial_state(x, 1.0);
    model.set_mesh(0.0, 1.0, 20);

    auto ocp = model.build(DecayDyn{}, ZeroCost{});
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "sim_workflow");

    // sim::linear_guess drives the solve.
    auto z0 = goss::sim::linear_guess(model, compiled.layout);
    goss::solver::IpoptSolver solver;
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);

    // Extract + serialize.
    auto traj = goss::sim::extract_trajectory(result, compiled.layout, model, ocp.mesh);
    ASSERT_EQ(traj.state("x").size(), compiled.layout.num_nodes());
    EXPECT_NEAR(traj.state("x").front(), 1.0, 1e-9);            // pinned initial
    EXPECT_NEAR(traj.state("x").back(), std::exp(-1.0), 1e-3);  // analytic final
    std::string csv = goss::sim::to_csv(traj);
    EXPECT_EQ(csv.find("time,x\n"), 0u);

    // Validate + diagnose.
    double err = goss::sim::validate_by_integration(ocp, result, compiled.layout);
    EXPECT_LT(err, 1e-4);
    auto diag = goss::sim::diagnose(result, err);
    EXPECT_TRUE(diag.ok);
}
