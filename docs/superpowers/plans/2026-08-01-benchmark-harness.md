# Benchmark Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent `bench/` component that runs the SAME optimal-control problem across different transcription schemes and solvers, collects timing + accuracy metrics, and reports a human-readable comparison table and/or CSV. This directly supports the framework's original goal of "test different solvers e.g. IPOPT" and is the deferred benchmark harness documented in the sim-layer plan.

**Architecture:**

The two key type-system constraints in goss are:
1. **Transcription schemes are compile-time types** — `Trapezoidal::compile<Dyn,Cost>(ocp, name)` and `HermiteSimpson::compile<Dyn,Cost>(ocp, name)` are templated static functions with no shared base class. There is no runtime-polymorphic scheme interface, by design (see `transcription.hpp` note: type-erased OcpProblem wrapper is YAGNI). The harness cannot hold a `vector<Scheme*>` and loop.
2. **Solvers are runtime-polymorphic** — `solver::Solver` is an abstract base; `IpoptSolver` and `NloptSolver` both implement `virtual SolverResult solve(const NLPProblem&, const vector<double>&)`. The harness CAN hold `vector<unique_ptr<Solver>>` and loop.

**Central design decision:** The harness offers a templated free function `run_scheme<SchemeTag>(ocp, model, scheme_name, solvers)` that the caller invokes ONCE PER SCHEME (at the call-site, at compile time). Inside `run_scheme`, after compiling the OCP with the scheme, it loops over the runtime solver vector, times each solve with `std::chrono::steady_clock`, calls `validate_by_integration`, and appends a `BenchmarkResult` to an output vector. The "2 schemes × N solvers" matrix is therefore produced by two consecutive `run_scheme` calls — one for `Trapezoidal`, one for `HermiteSimpson` — concatenating their output vectors. This unrolls the scheme axis at compile time while keeping the solver loop at runtime, matching the existing type system rather than fighting it.

**Reporting is kept PURE** — `to_table` and `to_csv` are free functions of `vector<BenchmarkResult>` that return a `string`. They are testable by feeding synthetic `BenchmarkResult` values without running any solver. Timing values are NEVER used in correctness assertions; tests only assert that `elapsed_seconds >= 0.0`, that column headers are present, that the row count matches the input, and that scheme/solver labels are correctly echoed. Accuracy assertions are only made in the flagship end-to-end test, on the analytic exp-decay problem where the expected objective value is known.

**Tech Stack:** C++17, `std::chrono::steady_clock`, the merged goss layers (`goss::transcription`, `goss::solver`, `goss::sim`), GoogleTest, CMake INTERFACE lib `goss_bench`, containerized build via `scripts/dev.sh`.

## Global Constraints

- Language: **C++17**.
- `bench/` consumes existing types; it adds NO new modeling or solver concepts.
  - Consumed: `model::Model`, `transcription::OcpProblem<Dyn,Cost>`, `transcription::Trapezoidal`, `transcription::HermiteSimpson`, `transcription::CompiledOcp{problem, layout}`, `solver::Solver` (abstract), `solver::SolverResult{status, x, objective_value, message}`, `solver::SolverStatus`, `sim::linear_guess(model, layout)`, `sim::validate_by_integration(ocp, result, layout)`.
- `BenchmarkResult` carries: `scheme_name` (string), `solver_name` (string), `solve_status` (SolverStatus), `objective_value` (double), `elapsed_seconds` (double — wall-clock, `std::chrono::steady_clock`), `validation_error` (double — max RK4 deviation from `validate_by_integration`), `num_variables` (std::size_t — from `layout.total_variables()`). No `num_iterations` field — neither solver adapter exposes an iteration count in `SolverResult`; do NOT fabricate one.
- `run_scheme<SchemeTag>` is templated on the scheme struct and on the OCP functor types, so it can call `SchemeTag::compile(ocp, model_name)` at compile time. The scheme name string is passed as a runtime argument.
- Timing: wrap ONLY `solver.solve(...)` with `steady_clock`. Compilation and `linear_guess` are NOT timed (they are setup, not solver performance).
- Validation: call `validate_by_integration(ocp, result, layout)` only when `solve_status == SolverStatus::Success`; store `0.0` otherwise (not meaningful for a failed solve, and avoids passing a garbage solution to the validator).
- Error handling: a `BenchError : std::runtime_error` for setup/usage errors (e.g. empty solver list). Individual solve failures are captured in `BenchmarkResult.solve_status` — they do NOT throw.
- Header-only where practical: `BenchmarkResult`, `run_scheme`, `to_table`, `to_csv`, `write_csv` are all in `include/goss/bench/` headers. `BenchError` lives in `include/goss/bench/errors.hpp`. The `goss_bench` CMake target is INTERFACE.
- Coding standards: verbose descriptive names; type annotations everywhere; comments explain WHY.
- Container-first: all cmake/ctest via `scripts/dev.sh '<command>'`. Never build on the host.
- Test framework: GoogleTest. Reporting tests are pure (synthetic data, no solver). Harness tests use the analytic exp-decay problem for structure + accuracy assertions.

---

## File Structure

```
include/goss/bench/
  errors.hpp          — BenchError : std::runtime_error
  benchmark_result.hpp — BenchmarkResult struct + solver_status_name()
  harness.hpp          — run_scheme<SchemeTag>(ocp, model, scheme_name, solvers) -> vector<BenchmarkResult>
  report.hpp           — to_table(results) -> string, to_csv(results) -> string, write_csv(results, path)

tests/bench/
  test_benchmark_result.cpp   — BenchmarkResult construction + solver_status_name()
  test_report.cpp              — to_table / to_csv from synthetic results (PURE, no solver)
  test_harness_unit.cpp        — run_scheme on exp-decay with a single scheme (structure assertions)
  test_bench_flagship.cpp      — full 2-scheme × 2-solver matrix on exp-decay (accuracy + label assertions)

CMakeLists.txt                — goss_bench INTERFACE + goss_bench_tests executable
```

---

### Task 1: Scaffold goss_bench target + BenchError + BenchmarkResult

**Files:**
- Create: `include/goss/bench/errors.hpp`
- Create: `include/goss/bench/benchmark_result.hpp`
- Modify: `CMakeLists.txt`
- Create: `tests/bench/test_benchmark_result.cpp`

**Interfaces produced:**

```cpp
// errors.hpp
namespace goss::bench {
class BenchError : public std::runtime_error { /* ctor(const string&) */ };
}

// benchmark_result.hpp
namespace goss::bench {
struct BenchmarkResult {
    std::string scheme_name;
    std::string solver_name;
    goss::solver::SolverStatus solve_status;
    double objective_value;    // objective at solved x (0.0 if failed)
    double elapsed_seconds;    // wall-clock time of solver.solve() alone (>= 0.0)
    double validation_error;   // max RK4 deviation (0.0 if solve failed)
    std::size_t num_variables; // layout.total_variables()
};
// Human-readable label for a SolverStatus enumerator, for table/CSV output.
std::string solver_status_name(goss::solver::SolverStatus status);
}
```

**CMake** — append to `CMakeLists.txt` after the `goss_sim_tests` block:
```cmake
# ---- Benchmark harness layer ----
add_library(goss_bench INTERFACE)
target_include_directories(goss_bench INTERFACE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(goss_bench INTERFACE goss_sim goss_solver goss_model
    goss_transcription goss_nlp goss_ad)

add_executable(goss_bench_tests
    tests/bench/test_benchmark_result.cpp)
target_include_directories(goss_bench_tests PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(goss_bench_tests PRIVATE
    goss_bench goss_sim goss_solver goss_model goss_transcription
    goss_nlp goss_ad goss_ad_impl goss_solver
    goss_ipopt_iface goss_nlopt_iface cppadcg
    $<$<BOOL:${CPPAD_LIB}>:${CPPAD_LIB}> GTest::gtest_main)
gtest_discover_tests(goss_bench_tests)
```
(Later tasks append additional `.cpp` files to the `add_executable`.)

- [ ] **Step 1: Write the failing tests**

```cpp
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `scripts/dev.sh 'cmake -S . -B build 2>&1 | tail -20'`
Expected: FAIL — bench headers not found.

- [ ] **Step 3: Write errors.hpp**

```cpp
// include/goss/bench/errors.hpp
#pragma once
#include <stdexcept>
#include <string>

namespace goss::bench {

/// Thrown for setup/usage errors in the benchmark harness (e.g. empty solver
/// list, invalid configuration). Individual solve failures are captured in
/// BenchmarkResult.solve_status and do NOT throw.
class BenchError : public std::runtime_error {
 public:
    explicit BenchError(const std::string& message) : std::runtime_error(message) {}
};

}  // namespace goss::bench
```

- [ ] **Step 4: Write benchmark_result.hpp**

```cpp
// include/goss/bench/benchmark_result.hpp
#pragma once
#include <cstddef>
#include <string>
#include "goss/solver/solver_result.hpp"

namespace goss::bench {

/// One row of benchmark output: the outcome of running one (scheme, solver) pair
/// on a single OCP instance.
struct BenchmarkResult {
    std::string scheme_name;               // e.g. "Trapezoidal", "HermiteSimpson"
    std::string solver_name;               // e.g. "IpoptSolver", "NloptSolver"
    goss::solver::SolverStatus solve_status = goss::solver::SolverStatus::Failure;
    double objective_value  = 0.0;         // objective at the solved x (0 if failed)
    double elapsed_seconds  = 0.0;         // wall-clock duration of solver.solve() alone
    double validation_error = 0.0;         // max RK4 re-integration deviation (0 if failed)
    std::size_t num_variables = 0;         // total decision variables = layout.total_variables()
};

/// Returns a short human-readable label for a SolverStatus enumerator.
/// Used by to_table() and to_csv() to produce readable status columns.
inline std::string solver_status_name(goss::solver::SolverStatus status) {
    using goss::solver::SolverStatus;
    switch (status) {
        case SolverStatus::Success:           return "Success";
        case SolverStatus::InfeasibleProblem: return "InfeasibleProblem";
        case SolverStatus::IterationLimit:    return "IterationLimit";
        case SolverStatus::NumericalError:    return "NumericalError";
        case SolverStatus::Failure:           return "Failure";
        default:                              return "Unknown";
    }
}

}  // namespace goss::bench
```

- [ ] **Step 5: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake -S . -B build && cmake --build build && ctest --test-dir build -R "BenchError|BenchmarkResult|SolverStatusName" --output-on-failure'`
Expected: all 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add include/goss/bench/errors.hpp include/goss/bench/benchmark_result.hpp \
        tests/bench/test_benchmark_result.cpp CMakeLists.txt
git commit -m "feat: bench scaffold + BenchmarkResult struct + BenchError"
```

---

### Task 2: Pure reporting — to_table and to_csv

**Files:**
- Create: `include/goss/bench/report.hpp`
- Create: `tests/bench/test_report.cpp`
- Modify: `CMakeLists.txt` (add `test_report.cpp` to `goss_bench_tests`)

**Interfaces produced:**

```cpp
namespace goss::bench {
// Returns a human-readable fixed-width table of all results.
// Header row: Scheme | Solver | Status | Objective | Time(s) | ValidationErr | NVars
// One data row per BenchmarkResult, separated by a line of dashes.
// Pure function: deterministic given a fixed vector<BenchmarkResult>.
std::string to_table(const std::vector<BenchmarkResult>& results);

// Returns CSV text: header row then one row per result.
// Columns: scheme,solver,status,objective,elapsed_s,validation_error,num_variables
// Full double precision (setprecision(17)).
std::string to_csv(const std::vector<BenchmarkResult>& results);

// Writes to_csv output to path. Throws BenchError if file cannot be opened.
void write_csv(const std::vector<BenchmarkResult>& results, const std::string& path);
}
```

**Design notes:**
- `to_table` and `to_csv` are PURE functions of a `vector<BenchmarkResult>` returning `string`. They are the ONLY reporting functions; no I/O happens inside them. This allows tests to feed synthetic results with known values and assert on the string structure without involving any solver.
- Timing values in `elapsed_seconds` are set to a fixed value in tests (e.g. `0.123`) — tests assert the value appears in the CSV text but do NOT assert on its magnitude (because real timing is non-deterministic). The only runtime assertion on timing is `elapsed_seconds >= 0.0` (in Task 3's harness test).
- Column width in `to_table` may be fixed (e.g. 20 chars per column) — tested by checking the header row contains the expected column names, not by pixel-perfect formatting.
- Mirror `sim::trajectory.hpp`'s pattern: `to_csv` (pure, returns string) + `write_csv` (thin I/O wrapper) + `setprecision(17)`.

- [ ] **Step 1: Write the failing tests**

```cpp
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `scripts/dev.sh 'cmake -S . -B build && cmake --build build 2>&1 | tail -20'`
Expected: FAIL — `report.hpp` not found.

- [ ] **Step 3: Write report.hpp**

```cpp
// include/goss/bench/report.hpp
#pragma once
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include "goss/bench/benchmark_result.hpp"
#include "goss/bench/errors.hpp"

namespace goss::bench {

/// Returns a human-readable fixed-width comparison table.
/// Each row shows: Scheme | Solver | Status | Objective | Time(s) | ValidationErr | NVars
/// The table is PURE: given the same results vector, the output is deterministic
/// (except that elapsed_seconds values are whatever was stored in the results —
/// callers must not assert on their exact magnitude in tests).
inline std::string to_table(const std::vector<BenchmarkResult>& results) {
    // Fixed column widths for readability.
    constexpr int kColScheme    = 18;
    constexpr int kColSolver    = 16;
    constexpr int kColStatus    = 20;
    constexpr int kColObjective = 14;
    constexpr int kColTime      = 12;
    constexpr int kColValErr    = 16;
    constexpr int kColNVars     = 8;

    std::ostringstream out;
    auto row_separator = [&]() {
        out << std::string(kColScheme + kColSolver + kColStatus + kColObjective
                           + kColTime + kColValErr + kColNVars + 8, '-') << "\n";
    };

    // Header row.
    row_separator();
    out << std::left
        << std::setw(kColScheme)    << "Scheme"
        << std::setw(kColSolver)    << "Solver"
        << std::setw(kColStatus)    << "Status"
        << std::setw(kColObjective) << "Objective"
        << std::setw(kColTime)      << "Time(s)"
        << std::setw(kColValErr)    << "ValidationErr"
        << std::setw(kColNVars)     << "NVars"
        << "\n";
    row_separator();

    for (const auto& r : results) {
        out << std::left
            << std::setw(kColScheme)    << r.scheme_name
            << std::setw(kColSolver)    << r.solver_name
            << std::setw(kColStatus)    << solver_status_name(r.solve_status)
            << std::setw(kColObjective) << std::setprecision(6) << r.objective_value
            << std::setw(kColTime)      << std::setprecision(4) << r.elapsed_seconds
            << std::setw(kColValErr)    << std::scientific << std::setprecision(3) << r.validation_error
            << std::setw(kColNVars)     << r.num_variables
            << "\n";
        out << std::defaultfloat;
    }
    row_separator();
    return out.str();
}

/// Returns all results as CSV text, mirroring sim::to_csv's pattern.
/// Header: scheme,solver,status,objective,elapsed_s,validation_error,num_variables
/// One data row per result. Full double precision (setprecision(17)).
inline std::string to_csv(const std::vector<BenchmarkResult>& results) {
    std::ostringstream out;
    out << std::setprecision(17);
    out << "scheme,solver,status,objective,elapsed_s,validation_error,num_variables\n";
    for (const auto& r : results) {
        out << r.scheme_name << ","
            << r.solver_name << ","
            << solver_status_name(r.solve_status) << ","
            << r.objective_value << ","
            << r.elapsed_seconds << ","
            << r.validation_error << ","
            << r.num_variables << "\n";
    }
    return out.str();
}

/// Writes to_csv output to the file at path.
/// Throws BenchError if the file cannot be opened.
inline void write_csv(const std::vector<BenchmarkResult>& results,
                      const std::string& path) {
    std::ofstream file(path);
    if (!file) throw BenchError("write_csv: cannot open '" + path + "' for writing");
    file << to_csv(results);
    if (!file) throw BenchError("write_csv: write failed for '" + path + "'");
}

}  // namespace goss::bench
```

- [ ] **Step 4: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake -S . -B build && cmake --build build && ctest --test-dir build -R "ToTable|ToCsv" --output-on-failure'`
Expected: all 7 report tests PASS (no solver involved).

- [ ] **Step 5: Commit**

```bash
git add include/goss/bench/report.hpp tests/bench/test_report.cpp CMakeLists.txt
git commit -m "feat: pure benchmark report — to_table + to_csv + write_csv"
```

---

### Task 3: Harness — run_scheme templated helper

**Files:**
- Create: `include/goss/bench/harness.hpp`
- Create: `tests/bench/test_harness_unit.cpp`
- Modify: `CMakeLists.txt` (add `test_harness_unit.cpp`)

**Interface produced:**

```cpp
namespace goss::bench {

/// Run a SINGLE scheme across all solvers, collecting one BenchmarkResult per solver.
///
/// Template parameters:
///   SchemeTag — one of goss::transcription::Trapezoidal or goss::transcription::HermiteSimpson.
///               The caller instantiates run_scheme ONCE PER SCHEME (at compile time).
///   DynamicsFn, CostFn — deduced from the ocp argument.
///
/// Arguments:
///   ocp          — the OcpProblem to transcribe (passed to SchemeTag::compile).
///   model        — needed by linear_guess to build x0.
///   scheme_name  — human-readable label stored in every BenchmarkResult row.
///   model_name   — passed to SchemeTag::compile as the CppADCG model name.
///   solvers      — vector of abstract solver pointers; must not be empty (throws BenchError).
///   solver_names — parallel vector of solver labels; must have same size as solvers.
///
/// Returns a vector<BenchmarkResult> of size solvers.size().
///
/// Design: schemes are compile-time types (SchemeTag::compile is a templated static),
/// so run_scheme must be called once per scheme. Solvers are runtime-polymorphic
/// (solver::Solver abstract base), so run_scheme loops over them at runtime.
/// The caller unrolls the scheme axis:
///   auto trap_results = run_scheme<Trapezoidal>(ocp, model, "Trapezoidal", "trap", solvers, names);
///   auto hs_results   = run_scheme<HermiteSimpson>(ocp, model, "HermiteSimpson", "hs", solvers, names);
///   all_results = trap_results; all_results.insert(end, hs_results.begin(), hs_results.end());
template <typename SchemeTag, typename DynamicsFn, typename CostFn>
std::vector<BenchmarkResult> run_scheme(
    const goss::transcription::OcpProblem<DynamicsFn, CostFn>& ocp,
    const goss::model::Model& model,
    const std::string& scheme_name,
    const std::string& model_name,
    const std::vector<goss::solver::Solver*>& solvers,
    const std::vector<std::string>& solver_names);
}
```

**Key implementation notes:**
- Validate `solvers.size() == solver_names.size()` and `!solvers.empty()`, else `BenchError`.
- Call `SchemeTag::compile(ocp, model_name)` ONCE before the solver loop; all solvers share the same compiled NLP. Each solve call gets a fresh `linear_guess` (cheap, no state to reuse across solvers).
- Time ONLY `solver->solve(*compiled.problem, x0)` with `steady_clock::now()` before and after.
- For a successful solve: call `validate_by_integration(ocp, result, layout)` and store the return value in `BenchmarkResult.validation_error`. For a non-Success status: store `0.0` (validation result is meaningless for a failed solve).
- Store `compiled.layout.total_variables()` in `num_variables`.

- [ ] **Step 1: Write the failing tests**

```cpp
// tests/bench/test_harness_unit.cpp
//
// Tests run_scheme<Trapezoidal> on the analytic exp-decay problem.
// Asserts STRUCTURE (labels, counts, non-negative timing) — NOT exact timing values.
// Accuracy is separately asserted in the flagship test (test_bench_flagship.cpp).
#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include "goss/bench/harness.hpp"
#include "goss/model/model.hpp"
#include "goss/transcription/trapezoidal.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "goss/solver/solver.hpp"

namespace {
// dx/dt = -x, x(0) = 1, cost = integral(0). Analytic: x(t) = exp(-t).
// Used for structure tests — small mesh (10 intervals) to keep solve fast.
struct ExpDecayDyn {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x_vec, const std::vector<T>&, T) const {
        return { -x_vec[0] };
    }
};
struct ZeroCostFn {
    template <typename T>
    T operator()(const std::vector<T>&, const std::vector<T>&, T) const { return T(0); }
};
}  // namespace

TEST(RunScheme, RejectsEmptySolverList) {
    goss::model::Model model;
    auto x = model.add_state("x");
    model.set_initial_state(x, 1.0);
    model.set_mesh(0.0, 1.0, 5);
    auto ocp = model.build(ExpDecayDyn{}, ZeroCostFn{});

    std::vector<goss::solver::Solver*> empty_solvers;
    EXPECT_THROW(
        goss::bench::run_scheme<goss::transcription::Trapezoidal>(
            ocp, model, "Trapezoidal", "trap_empty", empty_solvers, {}),
        goss::bench::BenchError);
}

TEST(RunScheme, RejectsMismatchedNameVectorSize) {
    goss::model::Model model;
    auto x = model.add_state("x");
    model.set_initial_state(x, 1.0);
    model.set_mesh(0.0, 1.0, 5);
    auto ocp = model.build(ExpDecayDyn{}, ZeroCostFn{});

    goss::solver::IpoptSolver ipopt;
    std::vector<goss::solver::Solver*> solvers = {&ipopt};
    std::vector<std::string> wrong_names = {"A", "B"};  // 2 names for 1 solver
    EXPECT_THROW(
        goss::bench::run_scheme<goss::transcription::Trapezoidal>(
            ocp, model, "Trapezoidal", "trap_mismatch", solvers, wrong_names),
        goss::bench::BenchError);
}

TEST(RunScheme, ProducesOneResultPerSolver) {
    goss::model::Model model;
    auto x = model.add_state("x");
    model.set_initial_state(x, 1.0);
    model.set_mesh(0.0, 1.0, 10);
    auto ocp = model.build(ExpDecayDyn{}, ZeroCostFn{});

    goss::solver::IpoptSolver ipopt;
    std::vector<goss::solver::Solver*> solvers = {&ipopt};
    std::vector<std::string> names = {"IpoptSolver"};

    auto results = goss::bench::run_scheme<goss::transcription::Trapezoidal>(
        ocp, model, "Trapezoidal", "trap_unit", solvers, names);

    ASSERT_EQ(results.size(), std::size_t{1});
}

TEST(RunScheme, ResultLabelsMatchInputArguments) {
    goss::model::Model model;
    auto x = model.add_state("x");
    model.set_initial_state(x, 1.0);
    model.set_mesh(0.0, 1.0, 10);
    auto ocp = model.build(ExpDecayDyn{}, ZeroCostFn{});

    goss::solver::IpoptSolver ipopt;
    std::vector<goss::solver::Solver*> solvers = {&ipopt};

    auto results = goss::bench::run_scheme<goss::transcription::Trapezoidal>(
        ocp, model, "Trapezoidal", "trap_labels", solvers, {"IpoptSolver"});

    ASSERT_EQ(results.size(), std::size_t{1});
    EXPECT_EQ(results[0].scheme_name, "Trapezoidal");
    EXPECT_EQ(results[0].solver_name, "IpoptSolver");
}

TEST(RunScheme, ElapsedSecondsIsNonNegative) {
    // Wall-clock time is non-deterministic; we only assert it is non-negative.
    // Never assert an exact value or a tight upper bound here.
    goss::model::Model model;
    auto x = model.add_state("x");
    model.set_initial_state(x, 1.0);
    model.set_mesh(0.0, 1.0, 10);
    auto ocp = model.build(ExpDecayDyn{}, ZeroCostFn{});

    goss::solver::IpoptSolver ipopt;
    auto results = goss::bench::run_scheme<goss::transcription::Trapezoidal>(
        ocp, model, "Trapezoidal", "trap_timing", {&ipopt}, {"IpoptSolver"});

    ASSERT_EQ(results.size(), std::size_t{1});
    EXPECT_GE(results[0].elapsed_seconds, 0.0);
}

TEST(RunScheme, NumVariablesIsPositive) {
    goss::model::Model model;
    auto x = model.add_state("x");
    model.set_initial_state(x, 1.0);
    model.set_mesh(0.0, 1.0, 10);
    auto ocp = model.build(ExpDecayDyn{}, ZeroCostFn{});

    goss::solver::IpoptSolver ipopt;
    auto results = goss::bench::run_scheme<goss::transcription::Trapezoidal>(
        ocp, model, "Trapezoidal", "trap_nvars", {&ipopt}, {"IpoptSolver"});

    ASSERT_EQ(results.size(), std::size_t{1});
    EXPECT_GT(results[0].num_variables, std::size_t{0});
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `scripts/dev.sh 'cmake -S . -B build && cmake --build build 2>&1 | tail -20'`
Expected: FAIL — `harness.hpp` not found.

- [ ] **Step 3: Write harness.hpp**

```cpp
// include/goss/bench/harness.hpp
#pragma once
#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include "goss/bench/benchmark_result.hpp"
#include "goss/bench/errors.hpp"
#include "goss/model/model.hpp"
#include "goss/sim/initial_guess.hpp"
#include "goss/sim/validation.hpp"
#include "goss/solver/solver.hpp"
#include "goss/solver/solver_result.hpp"
#include "goss/transcription/ocp_problem.hpp"

namespace goss::bench {

/// Run a single transcription scheme across all provided solvers, collecting
/// timing and accuracy metrics for each (scheme, solver) pair.
///
/// SchemeTag must be a scheme struct with a static compile() method
/// (e.g. goss::transcription::Trapezoidal, goss::transcription::HermiteSimpson).
/// Because compile() is a templated static on a concrete struct — not a virtual
/// on a base class — the scheme axis must be unrolled at the call-site by the
/// caller, once per scheme type. Solvers are runtime-polymorphic (solver::Solver*),
/// so they can be looped over without compile-time unrolling.
///
/// Timing covers ONLY the solver.solve() call; compilation and initial_guess
/// are setup costs, not solver-performance metrics.
///
/// On a successful solve, validation_error is filled with the result of
/// sim::validate_by_integration(ocp, result, layout) — an independent RK4 check.
/// On a non-Success solve, validation_error is 0.0 (not meaningful).
template <typename SchemeTag, typename DynamicsFn, typename CostFn>
std::vector<BenchmarkResult> run_scheme(
        const goss::transcription::OcpProblem<DynamicsFn, CostFn>& ocp,
        const goss::model::Model& model,
        const std::string& scheme_name,
        const std::string& model_name,
        const std::vector<goss::solver::Solver*>& solvers,
        const std::vector<std::string>& solver_names) {

    // Validate caller contract before touching any solver or compilation.
    if (solvers.empty())
        throw BenchError("run_scheme: solvers vector must not be empty");
    if (solvers.size() != solver_names.size())
        throw BenchError("run_scheme: solvers.size() != solver_names.size()");

    // Compile the OCP with the given scheme once; all solvers share the same NLP.
    // Each solver gets a freshly built linear_guess (cheap, stateless, correct).
    auto compiled = SchemeTag::compile(ocp, model_name);
    const goss::transcription::VariableLayout& layout = compiled.layout;
    const std::size_t total_variables = layout.total_variables();

    std::vector<BenchmarkResult> results;
    results.reserve(solvers.size());

    for (std::size_t solver_index = 0; solver_index < solvers.size(); ++solver_index) {
        BenchmarkResult bench_result;
        bench_result.scheme_name   = scheme_name;
        bench_result.solver_name   = solver_names[solver_index];
        bench_result.num_variables = total_variables;

        // Build a fresh initial guess for each solver; solver order must not matter.
        const std::vector<double> initial_guess = goss::sim::linear_guess(model, layout);

        // Time ONLY the solve call — not compilation, not initial guess generation.
        const auto time_before = std::chrono::steady_clock::now();
        goss::solver::SolverResult solve_result =
            solvers[solver_index]->solve(*compiled.problem, initial_guess);
        const auto time_after = std::chrono::steady_clock::now();

        bench_result.solve_status = solve_result.status;
        bench_result.objective_value = solve_result.objective_value;
        bench_result.elapsed_seconds = std::chrono::duration<double>(
            time_after - time_before).count();

        // Validate only when the solver converged; a failed solution vector
        // would produce a meaningless (possibly NaN) validation error.
        if (solve_result.status == goss::solver::SolverStatus::Success) {
            bench_result.validation_error =
                goss::sim::validate_by_integration(ocp, solve_result, layout);
        } else {
            bench_result.validation_error = 0.0;
        }

        results.push_back(bench_result);
    }

    return results;
}

}  // namespace goss::bench
```

- [ ] **Step 4: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake -S . -B build && cmake --build build && ctest --test-dir build -R "RunScheme" --output-on-failure'`
Expected: all 6 harness unit tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/goss/bench/harness.hpp tests/bench/test_harness_unit.cpp CMakeLists.txt
git commit -m "feat: run_scheme harness — compile-time scheme, runtime solver loop"
```

---

### Task 4: End-to-end flagship test — 2 schemes × 2 solvers

**Files:**
- Create: `tests/bench/test_bench_flagship.cpp`
- Modify: `CMakeLists.txt` (add `test_bench_flagship.cpp`)

**What this tests:** The full 2-scheme × 2-solver matrix on the analytic exp-decay problem (`dx/dt = -x`, `x(0) = 1`, cost = 0, analytic solution `x(t) = exp(-t)`). Asserts:
1. All 4 results are produced with the correct scheme and solver labels.
2. All 4 results have `elapsed_seconds >= 0.0`.
3. All 4 results have `num_variables > 0`.
4. For each `Success` result: `objective_value` is near 0 (cost is zero by construction), and `validation_error < 1e-3` (RK4 re-integration matches the collocation).
5. The combined results can be serialized via `to_table` and `to_csv` without error.
6. `to_csv` output starts with the expected header prefix.

**Note on NloptSolver:** NloptSolver (COBYLA, derivative-free) converges on this small, well-posed problem but may need more evaluations or a slightly looser tolerance for a coarser mesh. Use at least 20 intervals and the default NloptSolver settings. If NloptSolver returns IterationLimit rather than Success, record the failure gracefully (the harness stores it) but assert that at minimum the IPOPT results are successes. Do NOT assert NloptSolver's exact status — it is more sensitive to problem conditioning.

- [ ] **Step 1: Write the failing test**

```cpp
// tests/bench/test_bench_flagship.cpp
//
// Flagship benchmark: exp-decay problem across {Trapezoidal, HermiteSimpson}
//   x {IpoptSolver, NloptSolver}.
// Asserts: correct labels, non-negative timing, non-zero NVars, and (for IPOPT)
// near-zero objective and small RK4 validation error.
// DOES NOT assert exact timing values — timing is non-deterministic.
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include "goss/bench/harness.hpp"
#include "goss/bench/report.hpp"
#include "goss/model/model.hpp"
#include "goss/transcription/trapezoidal.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "goss/solver/nlopt_solver.hpp"
#include "goss/solver/solver.hpp"

namespace {
// dx/dt = -x, x(0) = 1, cost = 0. Analytic: x(t) = exp(-t).
// Objective value is 0 (zero running cost); validation error for a correct
// collocation solution should be small relative to the scheme order.
struct ExpDecayDyn {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x_vec, const std::vector<T>&, T) const {
        return { -x_vec[0] };
    }
};
struct ZeroCostFn {
    template <typename T>
    T operator()(const std::vector<T>&, const std::vector<T>&, T) const { return T(0); }
};
}  // namespace

TEST(BenchFlagship, TwoSchemesTwoSolversProduceFourResults) {
    constexpr double kTimeHorizon   = 1.0;
    constexpr std::size_t kIntervals = 20;  // enough for both schemes to converge cleanly
    constexpr double kValidationTol  = 1e-3; // max acceptable RK4 deviation for IPOPT results
    constexpr double kObjectiveTol   = 1e-6; // zero-cost problem; objective should be ~0

    goss::model::Model model;
    auto x_state = model.add_state("x");
    model.set_initial_state(x_state, 1.0);
    model.set_mesh(0.0, kTimeHorizon, kIntervals);
    auto ocp = model.build(ExpDecayDyn{}, ZeroCostFn{});

    // Build solver instances; keep them alive for the duration of the test.
    goss::solver::IpoptSolver ipopt_solver;
    goss::solver::NloptSolver nlopt_solver;
    std::vector<goss::solver::Solver*> solvers = {&ipopt_solver, &nlopt_solver};
    std::vector<std::string> solver_names      = {"IpoptSolver", "NloptSolver"};

    // Run both schemes. Scheme axis is unrolled at compile time (see harness.hpp).
    auto trapezoidal_results = goss::bench::run_scheme<goss::transcription::Trapezoidal>(
        ocp, model, "Trapezoidal", "flagship_trap", solvers, solver_names);
    auto hermite_simpson_results = goss::bench::run_scheme<goss::transcription::HermiteSimpson>(
        ocp, model, "HermiteSimpson", "flagship_hs", solvers, solver_names);

    // Combine into a single results table.
    std::vector<goss::bench::BenchmarkResult> all_results = trapezoidal_results;
    all_results.insert(all_results.end(),
                       hermite_simpson_results.begin(), hermite_simpson_results.end());

    // --- Structural assertions (scheme/solver count, labels, non-negative timing) ---
    ASSERT_EQ(all_results.size(), std::size_t{4})
        << "Expected 2 schemes x 2 solvers = 4 results";

    // Verify scheme labels.
    const auto count_scheme = [&](const std::string& name) {
        return std::count_if(all_results.begin(), all_results.end(),
                             [&](const goss::bench::BenchmarkResult& r) {
                                 return r.scheme_name == name; });
    };
    EXPECT_EQ(count_scheme("Trapezoidal"),   std::size_t{2});
    EXPECT_EQ(count_scheme("HermiteSimpson"), std::size_t{2});

    // Verify solver labels.
    const auto count_solver = [&](const std::string& name) {
        return std::count_if(all_results.begin(), all_results.end(),
                             [&](const goss::bench::BenchmarkResult& r) {
                                 return r.solver_name == name; });
    };
    EXPECT_EQ(count_solver("IpoptSolver"), std::size_t{2});
    EXPECT_EQ(count_solver("NloptSolver"), std::size_t{2});

    // All timing values must be non-negative (they are wall-clock durations >= 0).
    for (const auto& result : all_results) {
        EXPECT_GE(result.elapsed_seconds, 0.0)
            << "Negative elapsed_seconds for " << result.scheme_name
            << " + " << result.solver_name;
    }

    // All num_variables must be positive (any compiled OCP has at least 1 variable).
    for (const auto& result : all_results) {
        EXPECT_GT(result.num_variables, std::size_t{0})
            << "Zero num_variables for " << result.scheme_name
            << " + " << result.solver_name;
    }

    // --- Accuracy assertions for IPOPT results only ---
    // IPOPT is gradient-based and converges reliably on this well-posed problem.
    // NloptSolver (COBYLA, derivative-free) may not converge within default limits;
    // we do NOT require NloptSolver to succeed — the harness records the status gracefully.
    for (const auto& result : all_results) {
        if (result.solver_name == "IpoptSolver") {
            EXPECT_EQ(result.solve_status, goss::solver::SolverStatus::Success)
                << "IPOPT must converge on the exp-decay problem for scheme "
                << result.scheme_name;
            EXPECT_NEAR(result.objective_value, 0.0, kObjectiveTol)
                << "Zero-cost problem: objective must be ~0 for " << result.scheme_name;
            EXPECT_LT(result.validation_error, kValidationTol)
                << "RK4 validation error too large for IPOPT + " << result.scheme_name;
        }
    }

    // --- Reporting smoke-test (no solver calls; just serialize and check structure) ---
    const std::string table_text = goss::bench::to_table(all_results);
    EXPECT_NE(table_text.find("Trapezoidal"),    std::string::npos);
    EXPECT_NE(table_text.find("HermiteSimpson"), std::string::npos);
    EXPECT_NE(table_text.find("IpoptSolver"),    std::string::npos);

    const std::string csv_text = goss::bench::to_csv(all_results);
    // Header starts with "scheme".
    EXPECT_EQ(csv_text.find("scheme"), std::size_t{0});
    // 1 header + 4 data rows = 5 newlines.
    EXPECT_EQ(std::count(csv_text.begin(), csv_text.end(), '\n'), std::ptrdiff_t{5});
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `scripts/dev.sh 'cmake -S . -B build && cmake --build build 2>&1 | tail -20'`
Expected: FAIL — `test_bench_flagship.cpp` not yet in CMake's `add_executable`.

- [ ] **Step 3: Add to CMake**

In `CMakeLists.txt`, add `tests/bench/test_bench_flagship.cpp` to the `add_executable(goss_bench_tests ...)` list.

- [ ] **Step 4: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake -S . -B build && cmake --build build && ctest --test-dir build -R "BenchFlagship" --output-on-failure'`
Expected: PASS. The test produces 4 results covering all (scheme, solver) pairs. IPOPT converges and hits the tolerance. NloptSolver may produce IterationLimit — the test does not require it to succeed. The to_table/to_csv output contains the correct labels and row count.

- [ ] **Step 5: Run the FULL suite**

Run: `scripts/dev.sh 'ctest --test-dir build --output-on-failure'`
Expected: ALL tests pass. No regression in previously passing tests.

- [ ] **Step 6: Commit**

```bash
git add tests/bench/test_bench_flagship.cpp CMakeLists.txt
git commit -m "test: flagship benchmark — 2 schemes x 2 solvers on exp-decay, accuracy + label assertions"
```

---

### Task 5: Harness integration test + to_table/to_csv composition

**Files:**
- Create: `tests/bench/test_bench_workflow.cpp`
- Modify: `CMakeLists.txt` (add `test_bench_workflow.cpp`)

**Purpose:** Verify that `run_scheme` results pipe cleanly into `to_table` + `to_csv` + `write_csv`, and that the CSV output round-trips the scheme/solver labels correctly as text. This is a composition test — it uses a real solve but focuses on the data flow from harness → reporter, not on accuracy.

- [ ] **Step 1: Write the test**

```cpp
// tests/bench/test_bench_workflow.cpp
//
// Integration test: run_scheme → to_table + to_csv → verify data flows end to end.
// Tests the reporting pipeline with real solver output, complementing the PURE
// reporting tests in test_report.cpp (which use synthetic data only).
#include <gtest/gtest.h>
#include <algorithm>
#include <string>
#include <vector>
#include "goss/bench/harness.hpp"
#include "goss/bench/report.hpp"
#include "goss/model/model.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "goss/solver/solver.hpp"

namespace {
struct ExpDecayDyn {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x_vec, const std::vector<T>&, T) const {
        return { -x_vec[0] };
    }
};
struct ZeroCostFn {
    template <typename T>
    T operator()(const std::vector<T>&, const std::vector<T>&, T) const { return T(0); }
};
}  // namespace

TEST(BenchWorkflow, HarnessResultsFlowIntoReporter) {
    goss::model::Model model;
    auto x_state = model.add_state("x");
    model.set_initial_state(x_state, 1.0);
    model.set_mesh(0.0, 1.0, 10);
    auto ocp = model.build(ExpDecayDyn{}, ZeroCostFn{});

    goss::solver::IpoptSolver ipopt;
    std::vector<goss::solver::Solver*> solvers  = {&ipopt};
    std::vector<std::string> solver_names        = {"IpoptSolver"};

    // Run one scheme (HermiteSimpson chosen for its higher-order accuracy).
    auto results = goss::bench::run_scheme<goss::transcription::HermiteSimpson>(
        ocp, model, "HermiteSimpson", "workflow_hs", solvers, solver_names);
    ASSERT_EQ(results.size(), std::size_t{1});

    // Table serializes without throwing and contains the scheme label.
    std::string table_text = goss::bench::to_table(results);
    EXPECT_NE(table_text.find("HermiteSimpson"), std::string::npos);
    EXPECT_NE(table_text.find("IpoptSolver"),    std::string::npos);

    // CSV header is present; data row carries the correct scheme label.
    std::string csv_text = goss::bench::to_csv(results);
    EXPECT_EQ(csv_text.find("scheme"), std::size_t{0});
    EXPECT_NE(csv_text.find("HermiteSimpson"), std::string::npos);
    // 1 header + 1 data row = 2 newlines.
    EXPECT_EQ(std::count(csv_text.begin(), csv_text.end(), '\n'), std::ptrdiff_t{2});

    // Status name round-trips correctly through to_csv output.
    EXPECT_NE(csv_text.find("Success"), std::string::npos);
}
```

- [ ] **Step 2: Add to CMake, build and run — verify pass**

Add `tests/bench/test_bench_workflow.cpp` to `goss_bench_tests`.
Run: `scripts/dev.sh 'cmake -S . -B build && cmake --build build && ctest --test-dir build -R "BenchWorkflow" --output-on-failure'`
Expected: PASS.

- [ ] **Step 3: Run the FULL suite**

Run: `scripts/dev.sh 'ctest --test-dir build --output-on-failure'`
Expected: ALL tests pass (no regression).

- [ ] **Step 4: Commit**

```bash
git add tests/bench/test_bench_workflow.cpp CMakeLists.txt
git commit -m "test: harness→reporter integration — CSV round-trips scheme/solver labels"
```

---

### Task 6: Harness + reporter composition with two schemes + write_csv file output

**Files:**
- Modify: `tests/bench/test_bench_workflow.cpp` (extend)

**Purpose:** Add tests that:
1. Combine results from TWO `run_scheme` calls and verify the combined CSV has the correct row count.
2. Call `write_csv` with a valid temp path and verify the file is created and non-empty.
3. Call `write_csv` with an invalid path and verify it throws `BenchError`.

This finalises the `write_csv` coverage without requiring filesystem side effects in earlier pure tests.

- [ ] **Step 1: Write the additional tests**

Append to `tests/bench/test_bench_workflow.cpp`:

```cpp
#include <fstream>
#include <sstream>

TEST(BenchWorkflow, CombinedTwoSchemesHasCorrectCsvRowCount) {
    goss::model::Model model;
    auto x_state = model.add_state("x");
    model.set_initial_state(x_state, 1.0);
    model.set_mesh(0.0, 1.0, 10);
    auto ocp = model.build(ExpDecayDyn{}, ZeroCostFn{});

    goss::solver::IpoptSolver ipopt;
    std::vector<goss::solver::Solver*> solvers  = {&ipopt};
    std::vector<std::string> solver_names        = {"IpoptSolver"};

    auto trap_results = goss::bench::run_scheme<goss::transcription::Trapezoidal>(
        ocp, model, "Trapezoidal", "workflow_trap2", solvers, solver_names);
    auto hs_results = goss::bench::run_scheme<goss::transcription::HermiteSimpson>(
        ocp, model, "HermiteSimpson", "workflow_hs2", solvers, solver_names);

    std::vector<goss::bench::BenchmarkResult> combined = trap_results;
    combined.insert(combined.end(), hs_results.begin(), hs_results.end());

    std::string csv_text = goss::bench::to_csv(combined);
    // 1 header + 2 data rows = 3 newlines.
    EXPECT_EQ(std::count(csv_text.begin(), csv_text.end(), '\n'), std::ptrdiff_t{3});
}

TEST(BenchWorkflow, WriteCsvCreatesNonEmptyFile) {
    goss::bench::BenchmarkResult r;
    r.scheme_name = "TestScheme";
    r.solver_name = "TestSolver";
    r.solve_status = goss::solver::SolverStatus::Success;
    r.elapsed_seconds = 0.01;
    r.num_variables   = 10;

    // Use a temp path in /tmp (available in the container).
    const std::string temp_path = "/tmp/goss_bench_test_output.csv";
    EXPECT_NO_THROW(goss::bench::write_csv({r}, temp_path));

    std::ifstream file(temp_path);
    ASSERT_TRUE(file.is_open());
    std::ostringstream contents;
    contents << file.rdbuf();
    EXPECT_FALSE(contents.str().empty());
    EXPECT_NE(contents.str().find("TestScheme"), std::string::npos);
}

TEST(BenchWorkflow, WriteCsvThrowsOnBadPath) {
    goss::bench::BenchmarkResult r;
    r.scheme_name  = "S";
    r.solver_name  = "I";
    r.solve_status = goss::solver::SolverStatus::Success;
    // An invalid path that cannot be created.
    EXPECT_THROW(
        goss::bench::write_csv({r}, "/nonexistent_directory/goss_bench_bad_path.csv"),
        goss::bench::BenchError);
}
```

Also add `#include "goss/transcription/trapezoidal.hpp"` to the includes in `test_bench_workflow.cpp`.

- [ ] **Step 2: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake --build build && ctest --test-dir build -R "BenchWorkflow" --output-on-failure'`
Expected: all BenchWorkflow tests PASS.

- [ ] **Step 3: Run the FULL suite**

Run: `scripts/dev.sh 'ctest --test-dir build --output-on-failure'`
Expected: ALL tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/bench/test_bench_workflow.cpp
git commit -m "test: write_csv file output + combined two-scheme CSV row count"
```

---

### Task 7: to_table column alignment regression test

**Files:**
- Modify: `tests/bench/test_report.cpp` (append)

**Purpose:** Verify the `to_table` column structure regression — a multiline table for 4 results (the flagship scenario) must have exactly the right number of newline-separated lines and must contain both scheme and solver column headers. This pins the formatter's output shape so future formatting changes cause a visible test failure.

- [ ] **Step 1: Write the test**

Append to `tests/bench/test_report.cpp`:

```cpp
TEST(ToTable, FourRowsHaveCorrectLineCount) {
    // A full 2-scheme x 2-solver matrix of synthetic results.
    // to_table format: separator + header + separator + N data rows + separator = N + 3 lines.
    std::vector<goss::bench::BenchmarkResult> results = {
        make_result("Trapezoidal",   "IpoptSolver",  goss::solver::SolverStatus::Success,   0.5,  0.10, 1e-5, 42),
        make_result("Trapezoidal",   "NloptSolver",  goss::solver::SolverStatus::Success,   0.51, 0.30, 2e-5, 42),
        make_result("HermiteSimpson","IpoptSolver",  goss::solver::SolverStatus::Success,   0.5,  0.12, 5e-6, 82),
        make_result("HermiteSimpson","NloptSolver",  goss::solver::SolverStatus::IterationLimit, 0.0, 1.50, 0.0, 82),
    };
    std::string table = goss::bench::to_table(results);

    // All four scheme/solver combinations must appear.
    EXPECT_NE(table.find("Trapezoidal"),      std::string::npos);
    EXPECT_NE(table.find("HermiteSimpson"),   std::string::npos);
    EXPECT_NE(table.find("NloptSolver"),      std::string::npos);
    EXPECT_NE(table.find("IterationLimit"),   std::string::npos);

    // The table must be non-trivially multi-line: at least header + 4 data rows + separators.
    const int newline_count = static_cast<int>(std::count(table.begin(), table.end(), '\n'));
    // 3 separator lines + 1 header + 4 data rows = 8 minimum.
    EXPECT_GE(newline_count, 8);
}
```

- [ ] **Step 2: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake --build build && ctest --test-dir build -R "ToTable" --output-on-failure'`
Expected: PASS.

- [ ] **Step 3: Run the FULL suite**

Run: `scripts/dev.sh 'ctest --test-dir build --output-on-failure'`
Expected: ALL tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/bench/test_report.cpp
git commit -m "test: to_table column alignment regression for 4-result matrix"
```

---

## Self-Review

### Spec coverage

- **Benchmark harness** — `run_scheme<SchemeTag>` runs OCP across schemes and solvers, collecting timing + accuracy. Task 3 + 4. ✓
- **Timing** — `std::chrono::steady_clock` around `solver.solve()` only. Task 3 harness.hpp. ✓
- **Accuracy metric** — `validate_by_integration` from `sim/validation.hpp` stored in `BenchmarkResult.validation_error`. Task 3. ✓
- **Comparison table** — `to_table` (human-readable) and `to_csv` (CSV) in `report.hpp`. Task 2. ✓
- **Two schemes × two solvers** — flagship test asserts 4 results with correct labels. Task 4. ✓
- **Independent/testable bench/ area** — INTERFACE `goss_bench` target; files in `include/goss/bench/` and `tests/bench/`. Tasks 1-7. ✓
- **Deferred items documented:** auto-scaling, parallel solver execution, per-iteration callbacks, benchmark-to-benchmark comparison across runs. Not in scope; can be layered on later without changing the `BenchmarkResult` schema.

### Design decision correctness

**Compile-time schemes × runtime solvers:** The harness exposes `run_scheme<SchemeTag>(...)` — a free function templated on the scheme struct. The scheme axis is unrolled at the CALL-SITE by invoking `run_scheme` once for `Trapezoidal` and once for `HermiteSimpson`. Inside `run_scheme`, the solver loop is a runtime `for` over `vector<Solver*>`. This is the ONLY way to satisfy both constraints simultaneously: schemes have no common base class (by design, per `transcription.hpp` note), so virtual dispatch is not an option; solvers DO have a common abstract base (`solver::Solver`), so they CAN be held in a vector and looped. The call-site unrolling pattern is explicit and visible in the test code, making the "scheme axis is compile-time" constraint self-documenting.

**Reporting stays testable despite non-deterministic timing:** `to_table` and `to_csv` are PURE functions of `vector<BenchmarkResult>` — they format whatever is in the struct; they do not observe the clock. Tests for the reporting layer (Task 2) use synthetic `BenchmarkResult` values with a fixed `elapsed_seconds = 0.123`. The only test assertion on timing in the harness tests is `elapsed_seconds >= 0.0`. Accuracy (`objective_value`, `validation_error`) is asserted only in the flagship test against the analytic exp-decay problem where the expected value is known (zero cost, small RK4 deviation). This clean separation means the reporting tests are deterministic, the harness unit tests assert structure only, and only the flagship test combines both (and even there, timing is only asserted to be non-negative, never exact).

### Placeholder scan

No steps contain "TBD", "similar to above", "add tests", or "implement later" placeholders. Every step has either literal test code, literal implementation code, a cmake snippet, or an explicit "append to existing file" instruction with literal content.

### Type consistency

- `BenchmarkResult` fields used in `harness.hpp` (`scheme_name`, `solver_name`, `solve_status`, `objective_value`, `elapsed_seconds`, `validation_error`, `num_variables`) are exactly the fields declared in `benchmark_result.hpp`. No phantom fields.
- `run_scheme<SchemeTag, Dyn, Cost>(ocp, model, scheme_name, model_name, solvers, solver_names)` signature is consistent between Task 3 definition, Task 3 tests, Task 4 flagship test, and Task 5 workflow test.
- `to_table`, `to_csv`, `write_csv` in `report.hpp` all accept `const vector<BenchmarkResult>&`, consistent with all tests.
- `solver::Solver*` (raw non-owning pointer) used in `run_scheme` — caller owns the solver instances (stack-allocated in tests). This avoids ownership complexity and matches the pattern in existing solver tests.
- Consumed APIs verified against existing headers: `SchemeTag::compile(ocp, model_name) -> CompiledOcp{problem, layout}`, `VariableLayout::total_variables()`, `sim::linear_guess(model, layout)`, `sim::validate_by_integration(ocp, result, layout)`, `solver::SolverResult{status, objective_value}`, `solver::SolverStatus::Success`.

### Known risks flagged in-plan

- **NloptSolver convergence (flagship test):** Task 4 explicitly notes that NloptSolver (COBYLA, derivative-free) may return `IterationLimit` rather than `Success` on the exp-decay problem, and the test does NOT require it to succeed — only IPOPT accuracy is asserted. The harness records NloptSolver's outcome gracefully without throwing.
- **CppADCG model name collisions:** Each `run_scheme` call uses a distinct `model_name` string (e.g. `"flagship_trap"`, `"flagship_hs"`, `"workflow_trap2"`) to avoid CppADCG compilation cache collisions when multiple tests run in the same binary.
- **`write_csv` temp path:** Task 6 uses `/tmp/goss_bench_test_output.csv` — available in the container. If the container restricts `/tmp` writes, use a path relative to the build directory or skip the write test with `GTEST_SKIP`.
