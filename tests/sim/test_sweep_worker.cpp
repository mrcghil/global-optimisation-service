// tests/sim/test_sweep_worker.cpp
#include <gtest/gtest.h>
#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>
#include "goss/sim/sweep_worker.hpp"
#include "goss/sim/errors.hpp"

// Additional headers needed for the fork-solve test.
#include "goss/model/model.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "goss/sim/initial_guess.hpp"

// ---------------------------------------------------------------------------
// Test 1: Serialization round-trip (in-process, fast — no fork)
// ---------------------------------------------------------------------------

TEST(SweepWorker, SerializeRoundTrip) {
    goss::sim::SweepPoint point;
    point.parameters = {1.5, 2.5};
    point.status = goss::solver::SolverStatus::Success;
    point.objective_value = 42.25;
    point.x = {0.1, 0.2, 0.3};
    point.message = "solved";

    const auto bytes = goss::sim::serialize_sweep_point(point);
    const auto back  = goss::sim::deserialize_sweep_point(bytes);

    EXPECT_EQ(back.parameters, point.parameters);
    EXPECT_EQ(back.status, point.status);
    EXPECT_DOUBLE_EQ(back.objective_value, point.objective_value);
    EXPECT_EQ(back.x, point.x);
    EXPECT_EQ(back.message, point.message);
}

// ---------------------------------------------------------------------------
// Test 2: Fork-solve — end-to-end child process inheriting compiled .so
// ---------------------------------------------------------------------------

TEST(SweepWorker, SolvesOnePointInChildProcess) {
    goss::model::Model model;
    auto q    = model.add_state("queue_length");
    auto rate = model.add_control("service_rate");
    model.add_parameter("arrival_rate", 2.0, 0.0, 10.0);
    model.set_state_bounds(q, 0.0, 1e19);
    model.set_control_bounds(rate, 0.0, 5.0);
    model.set_initial_state(q, 10.0);
    model.set_mesh(0.0, 5.0, 30);
    auto ocp = model.build(
        [](const auto& x, const auto& u, const auto& p, auto){
            using T = std::decay_t<decltype(x[0])>; return std::vector<T>{ p[0]-u[0] }; },
        [](const auto& x, const auto& u, const auto&, auto){
            using T = std::decay_t<decltype(x[0])>; return x[0] + T(0.1)*u[0]*u[0]; });
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "sweep_worker_queue");
    const auto z0 = goss::sim::linear_guess(model, compiled.layout);
    goss::solver::IpoptSolver solver;

    auto point = goss::sim::solve_point_in_child(
        *compiled.problem, compiled.validator, solver, {2.0}, z0);

    EXPECT_EQ(point.status, goss::solver::SolverStatus::Success);
    EXPECT_EQ(point.parameters, (std::vector<double>{2.0}));
    EXPECT_GT(point.objective_value, 0.0);
}

// ---------------------------------------------------------------------------
// Test 3: Truncated buffer — regression guard for parent-crash bug
//
// A child that crashes mid-write leaves a partial (non-empty) payload in the
// pipe.  deserialize_sweep_point must throw SimError rather than memcpy past
// the buffer end (UB / SIGSEGV in the parent).
// ---------------------------------------------------------------------------

TEST(SweepWorker, DeserializeRejectsTruncatedBuffer) {
    // Build a valid SweepPoint and serialize it so we have a known-good payload.
    goss::sim::SweepPoint point;
    point.parameters      = {1.0, 2.0};
    point.status          = goss::solver::SolverStatus::Success;
    point.objective_value = 3.14;
    point.x               = {0.1, 0.2, 0.3};
    point.message         = "ok";

    const auto full_bytes = goss::sim::serialize_sweep_point(point);
    ASSERT_GT(full_bytes.size(), 1u)
        << "Serialized payload is unexpectedly tiny; test pre-condition failed";

    // Half-truncated: first serialized.size()/2 bytes (simulates mid-write crash).
    const std::vector<char> half_truncated(
        full_bytes.begin(),
        full_bytes.begin() + static_cast<std::ptrdiff_t>(full_bytes.size() / 2));
    EXPECT_THROW(goss::sim::deserialize_sweep_point(half_truncated), goss::sim::SimError);

    // Single-byte buffer: extreme truncation.
    const std::vector<char> one_byte(full_bytes.begin(), full_bytes.begin() + 1);
    EXPECT_THROW(goss::sim::deserialize_sweep_point(one_byte), goss::sim::SimError);
}

// ---------------------------------------------------------------------------
// Test 4: Overflow-form count rejected — regression guard for multiply-overflow
//
// A corrupt/malicious nparams value near SIZE_MAX / sizeof(double) would wrap
// to a small number when multiplied by sizeof(double), pass a multiply-based
// bounds check, and drive an OOB memcpy.  The division-form check must reject
// this without performing any multiply.
//
// Buffer layout matches serialize_sweep_point:
//   [int status][double objective][size_t nparams][...no doubles follow...]
// ---------------------------------------------------------------------------

TEST(SweepWorker, DeserializeRejectsOverflowingVectorCount) {
    // Craft a byte buffer with a valid status + objective prefix followed by
    // a nparams field set to a value that would overflow sizeof(double) multiply.
    const std::size_t overflow_count = std::numeric_limits<std::size_t>::max() / 4;

    std::vector<char> buffer;
    buffer.resize(sizeof(int) + sizeof(double) + sizeof(std::size_t));

    std::size_t write_pos = 0;

    // [int status]
    const int status_val = 0;
    std::memcpy(buffer.data() + write_pos, &status_val, sizeof(int));
    write_pos += sizeof(int);

    // [double objective]
    const double obj_val = 0.0;
    std::memcpy(buffer.data() + write_pos, &obj_val, sizeof(double));
    write_pos += sizeof(double);

    // [size_t nparams] — set to a value that overflows when multiplied by
    // sizeof(double); no actual double bytes follow.
    std::memcpy(buffer.data() + write_pos, &overflow_count, sizeof(std::size_t));

    EXPECT_THROW(goss::sim::deserialize_sweep_point(buffer), goss::sim::SimError);
}
