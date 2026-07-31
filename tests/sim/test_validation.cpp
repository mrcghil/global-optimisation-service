// tests/sim/test_validation.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include "goss/sim/validation.hpp"
#include "goss/sim/initial_guess.hpp"
#include "goss/model/model.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"

namespace {
// dx/dt = -x, x(0)=1, no controls. Analytic x(t)=exp(-t).
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

TEST(Validation, SmallErrorForCorrectlySolvedProblem) {
    goss::model::Model model;
    auto x = model.add_state("x");
    model.set_initial_state(x, 1.0);
    model.set_mesh(0.0, 1.0, 20);
    auto ocp = model.build(DecayDyn{}, ZeroCost{});
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "sim_valid_ok");
    goss::solver::IpoptSolver solver;
    auto z0 = goss::sim::linear_guess(model, compiled.layout);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);

    double err = goss::sim::validate_by_integration(ocp, result, compiled.layout);
    EXPECT_LT(err, 1e-4) << "RK4 re-integration should match the HS solution closely";
}

TEST(Validation, FlagsCorruptedSolution) {
    goss::model::Model model;
    auto x = model.add_state("x");
    model.set_initial_state(x, 1.0);
    model.set_mesh(0.0, 1.0, 20);
    auto ocp = model.build(DecayDyn{}, ZeroCost{});
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "sim_valid_bad");
    goss::solver::IpoptSolver solver;
    auto z0 = goss::sim::linear_guess(model, compiled.layout);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);

    // Corrupt the solution: overwrite a middle node's state with a bogus value.
    std::size_t mid = compiled.layout.num_nodes() / 2;
    result.x[compiled.layout.state_index(mid, 0)] += 5.0;

    double err = goss::sim::validate_by_integration(ocp, result, compiled.layout);
    EXPECT_GT(err, 1.0) << "a corrupted node must produce a large integration deviation";
}
