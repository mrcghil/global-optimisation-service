# Parallel Parameter Sweep Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a large parameter sweep — many independent solves across a grid of parameter sets — in parallel, where the AD tape is JIT-compiled ONCE and each worker only injects a fresh parameter vector before solving (no recompilation per point).

**Architecture:** A `sim/` sweep harness that treats a `CompiledOcp` as shared-immutable structure and each parameter point as a pure task `(compiled, params, guess) → SweepPoint`. Execution starts with a **process pool** (fork-based worker processes), because the JIT-compiled `CppAD::cg::GenericModel` and IPOPT's linear-solver backend (MA57) both carry per-call/static mutable state that is unsafe to share across threads. The harness mirrors the shape of the existing `bench/harness.hpp::run_scheme`. A serial reference implementation is built and tested first so parallel results can be verified against it.

**Tech Stack:** C++17, POSIX `fork`/`waitpid`/`pipe` (macOS + Linux), CppAD/CppADCodeGen (compile-once), IPOPT/NLopt, GoogleTest, CMake.

## Global Constraints

- **C++ standard:** C++17 (`CMakeLists.txt:5`). POSIX process APIs only (no platform-specific beyond `<unistd.h>`/`<sys/wait.h>`); the dev target is the Linux devcontainer + macOS host.
- **Header hygiene:** no third-party solver/AD type in public headers outside impl `.cpp`.
- **Error handling (org standard):** specific exception types (`sim::SimError`), meaningful messages, no silent catch-alls. A worker failure must surface as a classified `SweepPoint` result, never a silent drop.
- **Naming (user preference):** verbose descriptive names; type annotations throughout.
- **Depends on:** the **Parameter Binding plan** (`2026-08-02-parameter-binding.md`) — specifically `sim::apply_parameters`, `CompiledOcp.validator`, `nlp::NLPProblem::set_parameters`, and compile-once behavior. That plan MUST be landed first.
- **Compile-once is mandatory (user requirement):** the sweep MUST compile the model exactly once; workers only call `apply_parameters` + `solve`. No task may re-invoke a scheme's `compile` per point.
- **Build/test:** configure/build under `build/`; every task ends green via `ctest`.

---

### Task 1: `SweepPoint` / `SweepResult` value types

Result types first — they define the contract every executor (serial, then parallel) fills. Each point records its input parameters, solver status, objective, and a message, so a failed solve is a first-class classified outcome, not an exception that aborts the sweep.

**Files:**
- Create: `include/goss/sim/sweep_result.hpp`
- Test: `tests/sim/test_sweep_result.cpp`
- Modify: `CMakeLists.txt:177-188` (add TU to `goss_sim_tests`)

**Interfaces:**
- Produces:
  ```cpp
  namespace goss::sim {
  struct SweepPoint {
      std::vector<double> parameters;                 // the input parameter set
      goss::solver::SolverStatus status = goss::solver::SolverStatus::Failure;
      double objective_value = 0.0;
      std::vector<double> x;                           // final primal solution (may be empty on failure)
      std::string message;                             // solver/worker message
  };
  struct SweepResult {
      std::vector<SweepPoint> points;                  // aligned to the input grid order
      std::size_t num_succeeded() const;               // count of status == Success
  };
  }  // namespace goss::sim
  ```

- [ ] **Step 1: Write the failing test**

```cpp
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
```

- [ ] **Step 2: Wire test TU; run to verify FAIL**

Add `tests/sim/test_sweep_result.cpp` to `goss_sim_tests` (`CMakeLists.txt:177-182`).

Run: `cmake --build build --target goss_sim_tests`
Expected: FAIL — header missing.

- [ ] **Step 3: Implement `sweep_result.hpp`**

```cpp
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
```

- [ ] **Step 4: Run to verify PASS**

Run: `cd build && ctest -R "SweepResult" --output-on-failure`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add include/goss/sim/sweep_result.hpp tests/sim/test_sweep_result.cpp CMakeLists.txt
git commit -m "feat(sim): SweepPoint/SweepResult value types for parameter sweeps"
```

---

### Task 2: Serial sweep (the correctness oracle)

A single-process, single-threaded sweep: compile once, then for each parameter set validate-bind-solve. This is the reference the parallel executor must match exactly, and it is independently useful for small sweeps and CI determinism.

**Files:**
- Create: `include/goss/sim/sweep.hpp` (`run_sweep_serial`)
- Test: `tests/sim/test_sweep_serial.cpp`
- Modify: `CMakeLists.txt:177-188`

**Interfaces:**
- Consumes: `CompiledOcp` (`problem`, `layout`, `validator`), `sim::apply_parameters`, `solver::Solver`, `SweepPoint`/`SweepResult`.
- Produces:
  ```cpp
  /// Solve `problem` once per parameter set in `parameter_grid`, in order, on a
  /// single thread. `problem` is compiled ONCE by the caller; each point only
  /// binds parameters + solves. A non-Success solve is recorded as a SweepPoint
  /// (not thrown). A validation failure for a point is recorded as
  /// status=Failure with the validator's explicit message — the sweep continues.
  goss::sim::SweepResult run_sweep_serial(
      goss::nlp::NLPProblem& problem,
      const goss::model::ParameterValidator& validator,
      goss::solver::Solver& solver,
      const std::vector<std::vector<double>>& parameter_grid,
      const std::vector<double>& initial_guess);
  ```

- [ ] **Step 1: Write the failing test (uses the parametric queue from Plan A)**

```cpp
// tests/sim/test_sweep_serial.cpp
#include <gtest/gtest.h>
#include <vector>
#include "goss/model/model.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "goss/sim/initial_guess.hpp"
#include "goss/sim/sweep.hpp"

namespace {
goss::transcription::CompiledOcp build_queue(goss::model::Model& model) {
    auto q    = model.add_state("queue_length");
    auto rate = model.add_control("service_rate");
    model.add_parameter("arrival_rate", 2.0, 0.0, 10.0);
    model.set_state_bounds(q, 0.0, 1e19);
    model.set_control_bounds(rate, 0.0, 5.0);
    model.set_initial_state(q, 10.0);
    model.set_mesh(0.0, 5.0, 30);
    auto ocp = model.build(
        [](const auto& x, const auto& u, const auto& p, auto){
            using T = std::decay_t<decltype(x[0])>; return std::vector<T>{ p[0] - u[0] }; },
        [](const auto& x, const auto& u, const auto&, auto){
            using T = std::decay_t<decltype(x[0])>; return x[0] + T(0.1)*u[0]*u[0]; });
    return goss::transcription::HermiteSimpson::compile(ocp, "sweep_serial_queue");
}
}  // namespace

TEST(SweepSerial, SolvesEveryPointAndRecordsResults) {
    goss::model::Model model;
    auto compiled = build_queue(model);
    const auto z0 = goss::sim::linear_guess(model, compiled.layout);
    goss::solver::IpoptSolver solver;

    std::vector<std::vector<double>> grid = {{1.0}, {2.0}, {3.0}, {4.0}};
    auto result = goss::sim::run_sweep_serial(
        *compiled.problem, compiled.validator, solver, grid, z0);

    ASSERT_EQ(result.points.size(), 4u);
    EXPECT_EQ(result.num_succeeded(), 4u);
    // Objective monotonic in arrival rate.
    EXPECT_LT(result.points[0].objective_value, result.points[3].objective_value);
    // Parameters echoed back in order.
    EXPECT_EQ(result.points[2].parameters, (std::vector<double>{3.0}));
}

TEST(SweepSerial, InvalidPointRecordedNotThrown) {
    goss::model::Model model;
    auto compiled = build_queue(model);
    const auto z0 = goss::sim::linear_guess(model, compiled.layout);
    goss::solver::IpoptSolver solver;

    std::vector<std::vector<double>> grid = {{1.0}, {999.0}};  // 999 out of [0,10]
    auto result = goss::sim::run_sweep_serial(
        *compiled.problem, compiled.validator, solver, grid, z0);

    ASSERT_EQ(result.points.size(), 2u);
    EXPECT_EQ(result.points[0].status, goss::solver::SolverStatus::Success);
    EXPECT_EQ(result.points[1].status, goss::solver::SolverStatus::Failure);
    EXPECT_NE(result.points[1].message.find("arrival_rate"), std::string::npos);
}
```

- [ ] **Step 2: Wire test TU; run to verify FAIL**

Add `tests/sim/test_sweep_serial.cpp` to `goss_sim_tests` (`CMakeLists.txt:177-182`). The sim test target already links solver + transcription (`CMakeLists.txt:184-187`).

Run: `cmake --build build --target goss_sim_tests`
Expected: FAIL — `sweep.hpp` / `run_sweep_serial` missing.

- [ ] **Step 3: Implement `run_sweep_serial`**

```cpp
// include/goss/sim/sweep.hpp
#pragma once
#include <vector>
#include "goss/model/errors.hpp"
#include "goss/model/parameter.hpp"
#include "goss/nlp/nlp_problem.hpp"
#include "goss/sim/parameters.hpp"
#include "goss/sim/sweep_result.hpp"
#include "goss/solver/solver.hpp"

namespace goss::sim {

inline SweepResult run_sweep_serial(
        nlp::NLPProblem& problem,
        const model::ParameterValidator& validator,
        solver::Solver& solver,
        const std::vector<std::vector<double>>& parameter_grid,
        const std::vector<double>& initial_guess) {
    SweepResult result;
    result.points.reserve(parameter_grid.size());

    for (const std::vector<double>& parameters : parameter_grid) {
        SweepPoint point;
        point.parameters = parameters;
        try {
            apply_parameters(problem, validator, parameters);   // validate + inject
        } catch (const model::ModelError& validation_error) {
            point.status = solver::SolverStatus::Failure;
            point.message = validation_error.what();            // explicit, names param
            result.points.push_back(std::move(point));
            continue;
        }
        const solver::SolverResult solve_result = solver.solve(problem, initial_guess);
        point.status = solve_result.status;
        point.objective_value = solve_result.objective_value;
        point.x = solve_result.x;
        point.message = solve_result.message;
        result.points.push_back(std::move(point));
    }
    return result;
}

}  // namespace goss::sim
```

- [ ] **Step 4: Run to verify PASS**

Run: `cd build && ctest -R "SweepSerial" --output-on-failure`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add include/goss/sim/sweep.hpp tests/sim/test_sweep_serial.cpp CMakeLists.txt
git commit -m "feat(sim): serial parameter sweep (compile-once) as parallel correctness oracle"
```

---

### Task 3: Fork-based worker — solve one point in a child process

The atom of the process pool: fork a child that binds one parameter set, solves, and writes the `SweepPoint` back to the parent over a pipe. This isolates IPOPT/MA57 static state and any JIT model mutation to a throwaway process. Serialization is a fixed binary layout (no library dependency).

**Files:**
- Create: `include/goss/sim/sweep_worker.hpp` (serialization + `solve_point_in_child`)
- Create: `src/sim/sweep_worker.cpp` (POSIX fork/pipe implementation)
- Modify: `CMakeLists.txt` — add a `goss_sim_impl` static lib (there is none today; `goss_sim` is INTERFACE-only, `CMakeLists.txt:173-175`) and link it into `goss_sim_tests`
- Test: `tests/sim/test_sweep_worker.cpp`

**Design note (why a `.cpp`, and why processes):**
- `goss_sim` is currently header-only (`CMakeLists.txt:173`). The fork/pipe plumbing is non-template and should be compiled once, so this task introduces `goss_sim_impl` (STATIC), mirroring `goss_ad_impl` (`CMakeLists.txt:47`).
- Processes over threads because a shared `CppAD::cg::GenericModel` mutates internal buffers on each `SparseJacobian`/`SparseHessian` call (not const-safe across threads) and IPOPT's MA57 backend has static state. Fork gives each solve its own address space; the compiled `.so` is inherited copy-on-write, so no recompilation. (Threads remain a future optimization once a thread-safe linear solver is confirmed — see Task 6 note.)

**Interfaces:**
- Consumes: `nlp::NLPProblem&`, `model::ParameterValidator`, `solver::Solver&`, one parameter set, `initial_guess`.
- Produces:
  ```cpp
  /// Runs ONE parameter point in a freshly forked child process. The child binds
  /// parameters, solves, serializes the SweepPoint to a pipe, and _exit()s. The
  /// parent reads the SweepPoint. If the child crashes/segfaults, returns a
  /// SweepPoint with status=Failure and a message naming the signal/exit code —
  /// never throws for a child-side failure.
  SweepPoint solve_point_in_child(
      nlp::NLPProblem& problem,
      const model::ParameterValidator& validator,
      solver::Solver& solver,
      const std::vector<double>& parameters,
      const std::vector<double>& initial_guess);

  // Serialization used across the pipe (also unit-tested directly, in-process):
  std::vector<char> serialize_sweep_point(const SweepPoint& point);
  SweepPoint deserialize_sweep_point(const std::vector<char>& bytes);
  ```

- [ ] **Step 1: Write the failing serialization round-trip test (in-process, fast)**

```cpp
// tests/sim/test_sweep_worker.cpp
#include <gtest/gtest.h>
#include <vector>
#include "goss/sim/sweep_worker.hpp"

TEST(SweepWorker, SerializeRoundTrip) {
    goss::sim::SweepPoint point;
    point.parameters = {1.5, 2.5};
    point.status = goss::solver::SolverStatus::Success;
    point.objective_value = 42.25;
    point.x = {0.1, 0.2, 0.3};
    point.message = "solved";

    auto bytes = goss::sim::serialize_sweep_point(point);
    auto back  = goss::sim::deserialize_sweep_point(bytes);

    EXPECT_EQ(back.parameters, point.parameters);
    EXPECT_EQ(back.status, point.status);
    EXPECT_DOUBLE_EQ(back.objective_value, point.objective_value);
    EXPECT_EQ(back.x, point.x);
    EXPECT_EQ(back.message, point.message);
}
```

- [ ] **Step 2: Add `goss_sim_impl` lib + wire test; run to verify FAIL**

In `CMakeLists.txt`, after the `goss_sim` INTERFACE lib (`CMakeLists.txt:173-175`):

```cmake
add_library(goss_sim_impl STATIC src/sim/sweep_worker.cpp)
target_include_directories(goss_sim_impl PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(goss_sim_impl PUBLIC goss_sim goss_nlp goss_ad
  PRIVATE goss_solver goss_ipopt_iface goss_nlopt_iface)
```

Add `tests/sim/test_sweep_worker.cpp` to `goss_sim_tests` and link `goss_sim_impl` into it (`CMakeLists.txt:184-187`).

Run: `cmake --build build --target goss_sim_tests`
Expected: FAIL — `sweep_worker.hpp` / serialization missing.

- [ ] **Step 3: Implement serialization in `sweep_worker.hpp` + `.cpp`**

Header declares the API; `.cpp` implements a fixed little-endian-agnostic layout using raw `double`/`int`/length-prefixed strings & vectors:

```
[int status][double objective]
[size_t nparams][double * nparams]
[size_t nx][double * nx]
[size_t msglen][char * msglen]
```

Implement `serialize_sweep_point` / `deserialize_sweep_point` in `src/sim/sweep_worker.cpp` by `memcpy`-ing scalars and vectors into/out of `std::vector<char>`. (Same host on both ends of the fork, so byte layout is compatible — no cross-arch concern.)

Run: `cd build && ctest -R "SweepWorker.SerializeRoundTrip" --output-on-failure`
Expected: PASS.

- [ ] **Step 4: Write the failing fork-solve test**

```cpp
#include "goss/model/model.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "goss/sim/initial_guess.hpp"

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
```

- [ ] **Step 5: Implement `solve_point_in_child`**

In `src/sim/sweep_worker.cpp` using `<unistd.h>`, `<sys/wait.h>`:

```cpp
SweepPoint solve_point_in_child(nlp::NLPProblem& problem,
                                const model::ParameterValidator& validator,
                                solver::Solver& solver,
                                const std::vector<double>& parameters,
                                const std::vector<double>& initial_guess) {
    int pipe_fds[2];
    if (::pipe(pipe_fds) != 0)
        throw SimError("solve_point_in_child: pipe() failed");

    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(pipe_fds[0]); ::close(pipe_fds[1]);
        throw SimError("solve_point_in_child: fork() failed");
    }

    if (pid == 0) {
        // ---- CHILD ----
        ::close(pipe_fds[0]);                 // close read end
        SweepPoint point;
        point.parameters = parameters;
        try {
            apply_parameters(problem, validator, parameters);
            const solver::SolverResult r = solver.solve(problem, initial_guess);
            point.status = r.status;
            point.objective_value = r.objective_value;
            point.x = r.x;
            point.message = r.message;
        } catch (const std::exception& error) {
            point.status = solver::SolverStatus::Failure;
            point.message = std::string("child exception: ") + error.what();
        }
        const std::vector<char> bytes = serialize_sweep_point(point);
        std::size_t total = 0;
        while (total < bytes.size()) {
            const ssize_t written = ::write(pipe_fds[1], bytes.data() + total, bytes.size() - total);
            if (written <= 0) break;
            total += static_cast<std::size_t>(written);
        }
        ::close(pipe_fds[1]);
        ::_exit(0);                           // do NOT run atexit/global dtors
    }

    // ---- PARENT ----
    ::close(pipe_fds[1]);                     // close write end
    std::vector<char> buffer;
    char chunk[4096];
    ssize_t got;
    while ((got = ::read(pipe_fds[0], chunk, sizeof(chunk))) > 0)
        buffer.insert(buffer.end(), chunk, chunk + got);
    ::close(pipe_fds[0]);

    int wait_status = 0;
    ::waitpid(pid, &wait_status, 0);

    if (buffer.empty()) {
        SweepPoint point;
        point.parameters = parameters;
        point.status = solver::SolverStatus::Failure;
        if (WIFSIGNALED(wait_status))
            point.message = "worker killed by signal " + std::to_string(WTERMSIG(wait_status));
        else
            point.message = "worker produced no output (exit " +
                            std::to_string(WEXITSTATUS(wait_status)) + ")";
        return point;
    }
    return deserialize_sweep_point(buffer);
}
```

Run: `cd build && ctest -R "SweepWorker.SolvesOnePointInChildProcess" --output-on-failure`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add include/goss/sim/sweep_worker.hpp src/sim/sweep_worker.cpp tests/sim/test_sweep_worker.cpp CMakeLists.txt
git commit -m "feat(sim): fork-based single-point worker with pipe serialization"
```

---

### Task 4: Process-pool sweep executor

Drive a bounded pool of concurrent worker processes across the whole grid, collecting `SweepPoint`s in grid order. Concurrency is capped by `max_parallel_workers` (default = hardware concurrency).

**Files:**
- Modify: `include/goss/sim/sweep.hpp` (add `SweepConfig` + `run_sweep_parallel` declaration)
- Modify: `src/sim/sweep_worker.cpp` (or new `src/sim/sweep.cpp`) — pool scheduler
- Test: `tests/sim/test_sweep_parallel.cpp`
- Modify: `CMakeLists.txt` (add scheduler source to `goss_sim_impl`; wire test)

**Interfaces:**
- Consumes: `solve_point_in_child` (Task 3), `SweepResult` (Task 1).
- Produces:
  ```cpp
  struct SweepConfig {
      std::size_t max_parallel_workers = 0;   // 0 => std::thread::hardware_concurrency()
  };

  /// Runs the grid across a bounded pool of forked worker processes. Results are
  /// returned in the SAME ORDER as parameter_grid (order-preserving despite
  /// out-of-order completion). Compile-once: `problem` is compiled by the caller;
  /// each worker only binds+solves. Deterministic given a deterministic solver.
  SweepResult run_sweep_parallel(
      nlp::NLPProblem& problem,
      const model::ParameterValidator& validator,
      solver::Solver& solver,
      const std::vector<std::vector<double>>& parameter_grid,
      const std::vector<double>& initial_guess,
      const SweepConfig& config = {});
  ```

- [ ] **Step 1: Write the failing test — parallel matches serial exactly**

```cpp
// tests/sim/test_sweep_parallel.cpp
#include <gtest/gtest.h>
#include <vector>
#include "goss/model/model.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "goss/sim/initial_guess.hpp"
#include "goss/sim/sweep.hpp"

namespace {
goss::transcription::CompiledOcp build_queue(goss::model::Model& model, const char* name) {
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
    return goss::transcription::HermiteSimpson::compile(ocp, name);
}
}  // namespace

TEST(SweepParallel, MatchesSerialResultsInOrder) {
    std::vector<std::vector<double>> grid = {{1.0},{2.0},{3.0},{4.0},{5.0},{6.0}};

    goss::model::Model model_s;
    auto compiled_s = build_queue(model_s, "sweep_par_serial");
    const auto z0_s = goss::sim::linear_guess(model_s, compiled_s.layout);
    goss::solver::IpoptSolver solver_s;
    auto serial = goss::sim::run_sweep_serial(
        *compiled_s.problem, compiled_s.validator, solver_s, grid, z0_s);

    goss::model::Model model_p;
    auto compiled_p = build_queue(model_p, "sweep_par_parallel");
    const auto z0_p = goss::sim::linear_guess(model_p, compiled_p.layout);
    goss::solver::IpoptSolver solver_p;
    goss::sim::SweepConfig config; config.max_parallel_workers = 4;
    auto parallel = goss::sim::run_sweep_parallel(
        *compiled_p.problem, compiled_p.validator, solver_p, grid, z0_p, config);

    ASSERT_EQ(parallel.points.size(), serial.points.size());
    for (std::size_t i = 0; i < grid.size(); ++i) {
        EXPECT_EQ(parallel.points[i].parameters, serial.points[i].parameters);
        EXPECT_EQ(parallel.points[i].status, serial.points[i].status);
        EXPECT_NEAR(parallel.points[i].objective_value,
                    serial.points[i].objective_value, 1e-6);
    }
}
```

- [ ] **Step 2: Wire test TU; run to verify FAIL**

Add `tests/sim/test_sweep_parallel.cpp` to `goss_sim_tests`.

Run: `cmake --build build --target goss_sim_tests`
Expected: FAIL — `run_sweep_parallel` / `SweepConfig` missing.

- [ ] **Step 3: Implement the bounded process pool**

Add `SweepConfig` + `run_sweep_parallel` declaration to `sweep.hpp`; implement the scheduler in the impl `.cpp`. Approach: maintain a map from live `pid` → grid index and its read-pipe fd; launch up to `max_parallel_workers` children (each via the same fork/pipe logic factored out of Task 3 so a worker can be launched non-blocking), then `waitpid` for any child, drain its pipe, place the deserialized `SweepPoint` at its grid index, and launch the next pending point until the grid is exhausted.

Key requirements:
- **Order-preserving:** pre-size `result.points` to `grid.size()`; write each completed point to its original index.
- **Bounded concurrency:** never exceed `max_parallel_workers` live children.
- **`max_parallel_workers == 0`** resolves to `std::thread::hardware_concurrency()` (fallback to 1 if that returns 0).
- Factor the fork/launch out of `solve_point_in_child` into a `launch_worker(...)` returning `{pid, read_fd}` and a `collect_worker(pid, read_fd)` returning `SweepPoint`, so both Task 3's blocking call and this pool reuse identical child logic (DRY).

- [ ] **Step 4: Run to verify PASS**

Run: `cd build && ctest -R "SweepParallel" --output-on-failure`
Expected: PASS.

- [ ] **Step 5: Add a bad-point-in-pool regression test**

```cpp
TEST(SweepParallel, InvalidPointsRecordedAtCorrectIndices) {
    std::vector<std::vector<double>> grid = {{1.0},{999.0},{3.0}};  // middle invalid
    goss::model::Model model;
    auto compiled = build_queue(model, "sweep_par_bad");
    const auto z0 = goss::sim::linear_guess(model, compiled.layout);
    goss::solver::IpoptSolver solver;
    goss::sim::SweepConfig config; config.max_parallel_workers = 3;
    auto result = goss::sim::run_sweep_parallel(
        *compiled.problem, compiled.validator, solver, grid, z0, config);

    ASSERT_EQ(result.points.size(), 3u);
    EXPECT_EQ(result.points[0].status, goss::solver::SolverStatus::Success);
    EXPECT_EQ(result.points[1].status, goss::solver::SolverStatus::Failure);
    EXPECT_NE(result.points[1].message.find("arrival_rate"), std::string::npos);
    EXPECT_EQ(result.points[2].status, goss::solver::SolverStatus::Success);
}
```

Run: `cd build && ctest -R "SweepParallel" --output-on-failure`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add include/goss/sim/sweep.hpp src/sim/sweep_worker.cpp tests/sim/test_sweep_parallel.cpp CMakeLists.txt
git commit -m "feat(sim): order-preserving bounded process-pool parameter sweep"
```

---

### Task 5: Grid construction helper

A convenience for building a Cartesian-product parameter grid from per-parameter axes, so callers don't hand-roll nested loops. Small, pure, and independently testable.

**Files:**
- Modify: `include/goss/sim/sweep.hpp` (add `make_grid`)
- Test: `tests/sim/test_sweep_grid.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces:
  ```cpp
  /// Cartesian product of per-parameter value axes. axes[i] is the list of
  /// values for parameter i; returns every combination, with parameter 0 varying
  /// slowest (row-major). Throws SimError if axes is empty or any axis is empty.
  std::vector<std::vector<double>> make_grid(const std::vector<std::vector<double>>& axes);
  ```

- [ ] **Step 1: Write the failing test**

```cpp
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
```

- [ ] **Step 2: Wire test TU; run to verify FAIL**

Add `tests/sim/test_sweep_grid.cpp` to `goss_sim_tests`.

Run: `cmake --build build --target goss_sim_tests`
Expected: FAIL — `make_grid` missing.

- [ ] **Step 3: Implement `make_grid`**

```cpp
inline std::vector<std::vector<double>> make_grid(
        const std::vector<std::vector<double>>& axes) {
    if (axes.empty())
        throw SimError("make_grid: at least one parameter axis is required");
    for (std::size_t i = 0; i < axes.size(); ++i)
        if (axes[i].empty())
            throw SimError("make_grid: axis " + std::to_string(i) + " is empty");

    std::vector<std::vector<double>> grid = {{}};
    for (const std::vector<double>& axis : axes) {
        std::vector<std::vector<double>> next;
        next.reserve(grid.size() * axis.size());
        for (const std::vector<double>& prefix : grid)
            for (double value : axis) {
                std::vector<double> combination = prefix;
                combination.push_back(value);
                next.push_back(std::move(combination));
            }
        grid = std::move(next);
    }
    return grid;
}
```

(Add `#include "goss/sim/errors.hpp"` to `sweep.hpp`.)

- [ ] **Step 4: Run to verify PASS**

Run: `cd build && ctest -R "SweepGrid" --output-on-failure`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add include/goss/sim/sweep.hpp tests/sim/test_sweep_grid.cpp CMakeLists.txt
git commit -m "feat(sim): Cartesian-product make_grid helper for sweeps"
```

---

### Task 6: End-to-end 2-D sweep + full-suite regression

A realistic 2-D sweep (arrival rate × cost weight) driven end-to-end through `make_grid` + `run_sweep_parallel`, asserting compile-once and correct point count, plus a full `goss_sim_tests` run to confirm nothing regressed.

**Files:**
- Test: `tests/sim/test_sweep_workflow.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `make_grid`, `run_sweep_parallel`, everything above.

- [ ] **Step 1: Write the workflow test**

```cpp
// tests/sim/test_sweep_workflow.cpp
#include <gtest/gtest.h>
#include <vector>
#include "goss/model/model.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "goss/sim/initial_guess.hpp"
#include "goss/sim/sweep.hpp"

TEST(SweepWorkflow, TwoDimensionalArrivalRateByCostWeight) {
    goss::model::Model model;
    auto q    = model.add_state("queue_length");
    auto rate = model.add_control("service_rate");
    model.add_parameter("arrival_rate", 2.0, 0.0, 10.0);   // param 0
    model.add_parameter("cost_weight",  0.1, 0.0, 10.0);   // param 1
    model.set_state_bounds(q, 0.0, 1e19);
    model.set_control_bounds(rate, 0.0, 5.0);
    model.set_initial_state(q, 10.0);
    model.set_mesh(0.0, 5.0, 25);
    auto ocp = model.build(
        [](const auto& x, const auto& u, const auto& p, auto){
            using T = std::decay_t<decltype(x[0])>; return std::vector<T>{ p[0] - u[0] }; },
        [](const auto& x, const auto& u, const auto& p, auto){
            using T = std::decay_t<decltype(x[0])>; return x[0] + p[1]*u[0]*u[0]; });
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "sweep_workflow_queue");
    const auto z0 = goss::sim::linear_guess(model, compiled.layout);
    goss::solver::IpoptSolver solver;

    auto grid = goss::sim::make_grid({{1.0, 2.0, 3.0}, {0.05, 0.1, 0.5}});  // 9 points
    goss::sim::SweepConfig config; config.max_parallel_workers = 4;
    auto result = goss::sim::run_sweep_parallel(
        *compiled.problem, compiled.validator, solver, grid, z0, config);

    ASSERT_EQ(result.points.size(), 9u);
    EXPECT_EQ(result.num_succeeded(), 9u);
}
```

- [ ] **Step 2: Wire test TU; run to verify PASS**

Add `tests/sim/test_sweep_workflow.cpp` to `goss_sim_tests`.

Run: `cmake --build build --target goss_sim_tests && cd build && ctest -R "SweepWorkflow" --output-on-failure`
Expected: PASS.

- [ ] **Step 3: Full sim-suite regression**

Run: `cd build && ctest -R "goss_sim_tests" --output-on-failure`
Expected: PASS — all existing sim tests plus every sweep test green.

- [ ] **Step 4: Document the threading follow-on (no code)**

Append a short note to `docs/superpowers/plans/notes/2026-08-02-param-mechanism-decision.md` (created in Plan A) recording that the sweep uses a process pool, why (GenericModel per-call mutation + IPOPT/MA57 static state), and the condition to revisit a thread-pool executor: a confirmed thread-safe linear solver (e.g. Pardiso) plus per-thread NLPProblem copies. This captures the earlier design conversation so it isn't lost.

- [ ] **Step 5: Commit**

```bash
git add tests/sim/test_sweep_workflow.cpp docs/superpowers/plans/notes/2026-08-02-param-mechanism-decision.md CMakeLists.txt
git commit -m "test(sim): end-to-end 2-D parallel sweep; document process-pool rationale"
```

---

## Self-Review

**Requirement coverage:**
- Many concurrent solves for sweeps (user's primary need) → Task 4 process pool + Task 6 workflow.
- Compile-once, workers only inject parameters (user requirement) → enforced by design (caller compiles `CompiledOcp` once; workers call `apply_parameters`+`solve`), depended-on from Plan A, asserted in Tasks 2/3/4/6.
- Explicit parameter-validation errors (user requirement) → surfaced through `apply_parameters` and recorded per-point in Tasks 2/4 (bad-point tests assert the message names the parameter).
- GPU note: intentionally out of scope here — for many small solves the earlier analysis concluded CPU process-pool throughput is the right first target; batched-GPU is a later optimization, recorded in the Task 6 note.

**Placeholder scan:** No TBD/TODO. Task 4's scheduler is described with concrete requirements (order-preserving via pre-sized indices, bounded concurrency, DRY `launch_worker`/`collect_worker` split) rather than left open.

**Type consistency:** `SweepPoint`/`SweepResult`, `SweepConfig`, `run_sweep_serial`, `run_sweep_parallel`, `solve_point_in_child`, `serialize_sweep_point`/`deserialize_sweep_point`, and `make_grid` signatures are consistent across Tasks 1–6. All consume the Plan A surface (`CompiledOcp.validator`, `sim::apply_parameters`, `NLPProblem::set_parameters`) with matching types.

**Dependency note:** This plan is unbuildable until the Parameter Binding plan lands — Task 2's oracle and every worker rely on `apply_parameters` + compile-once `set_parameters`. Land `2026-08-02-parameter-binding.md` first.
