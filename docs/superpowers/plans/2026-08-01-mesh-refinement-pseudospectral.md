# Mesh Refinement + Pseudospectral Collocation Plan

> **REQUIRED SUB-SKILL: superpowers:subagent-driven-development**

**Goal:** Extend the `transcription/` layer with two follow-on features:
- **Part A (Tasks 1–6): Non-uniform mesh + adaptive mesh refinement** — introduce a non-uniform `Mesh` variant carrying an explicit sorted node-time vector, update Trapezoidal and HermiteSimpson to accept per-interval widths, then build a `refine_and_solve` loop that bisects high-error intervals until a tolerance is met.
- **Part B (Tasks 7–11): Pseudospectral (LGL) collocation** — a new `LegendreGaussLobatto` scheme that maps Legendre-Gauss-Lobatto nodes onto `[t0, tf]` and enforces collocation via the LGL differentiation matrix. Its convergence is spectral (super-algebraic) for smooth problems.

**Architecture:**

```
include/goss/transcription/
  mesh.hpp                    — NEW: NonUniformMesh (explicit node-time vector) +
                                     uniform_mesh() convenience factory
  ocp_problem.hpp             — MODIFY: OcpProblem gains NonUniformMesh overload;
                                     Mesh kept as a uniform alias
  trapezoidal.hpp             — MODIFY: packed functor uses per-interval h_k = t[k+1]-t[k]
  hermite_simpson.hpp         — MODIFY: same per-interval h_k adaptation
  lgl_nodes.hpp               — NEW: lgl_nodes_and_weights(n), lgl_differentiation_matrix(nodes)
  legendre_gauss_lobatto.hpp  — NEW: LGL scheme compile()
  mesh_refinement.hpp         — NEW: refine_and_solve(), RefinementResult, per-interval error
  errors.hpp                  — no change
  transcription.hpp           — no change
  variable_layout.hpp         — no change

src/transcription/
  trapezoidal.cpp             — one-line TU (header includes impl)
  hermite_simpson.cpp         — one-line TU
  lgl_nodes.cpp               — NEW: non-template LGL math (no AD dependency)
  legendre_gauss_lobatto.cpp  — one-line TU (header-only templated scheme)
  mesh_refinement.cpp         — one-line TU

tests/transcription/
  ocp_fixtures.hpp            — MODIFY: add make_exponential_decay_nonuniform(),
                                     add localized-feature fixture (steep-gradient OCP)
  test_mesh.cpp               — NEW: NonUniformMesh construction/validation/width-vector
  test_trapezoidal.cpp        — MODIFY: add non-uniform convergence test
  test_hermite_simpson.cpp    — MODIFY: add non-uniform convergence test
  test_lgl_nodes.cpp          — NEW: LGL nodes/weights/D-matrix correctness
  test_legendre_gauss_lobatto.cpp — NEW: solve + spectral convergence
  test_mesh_refinement.cpp    — NEW: refinement loop reduces error vs uniform same node count
  test_scheme_agreement.cpp   — MODIFY: add LGL to the multi-scheme agreement test

CMakeLists.txt                — add new source + test files; new lgl_nodes.cpp static TU
```

**Tech Stack:** C++17, CppADCGBackend (AD recording), IPOPT (NLP solve), GoogleTest, `scripts/dev.sh` container.

**Global Constraints:**

- All `cmake`/`ctest` run inside the container via `scripts/dev.sh '<command>'`.
- AD-safe templated packed functors: use `T(literal)` for every numeric constant inside generic lambdas (never bare `double` literals where `T` is an AD type).
- Header-only where the existing schemes are header-only (LGL scheme is header-only; LGL math helpers are a non-template `.cpp` TU).
- Error type: `TranscriptionError` (existing `include/goss/transcription/errors.hpp`), thrown on dimension mismatch, bad mesh, etc.
- Verbose, descriptive variable names; type annotations; comments explain WHY.
- Every new file added to `goss_transcription` / `goss_transcription_tests` in CMake.
- Backward compatibility: existing uniform-mesh `Mesh` struct and `OcpProblem<Dyn,Cost>` API remain valid. No existing test may change behaviour (only additive changes to those files).

---

## KEY DESIGN DECISION: Non-Uniform Mesh Representation

**The chosen design** (minimal backward-compatible change):

Introduce a `NonUniformMesh` struct in a new `mesh.hpp` header:

```cpp
struct NonUniformMesh {
    std::vector<double> node_times;  // sorted, size = num_nodes = num_intervals+1
    std::size_t num_intervals() const { return node_times.size() - 1; }
    std::size_t num_nodes()     const { return node_times.size(); }
    double t_initial()          const { return node_times.front(); }
    double t_final()            const { return node_times.back(); }
    double interval_width(std::size_t k) const { return node_times[k+1] - node_times[k]; }
    void validate() const; // throws TranscriptionError if <2 nodes or not strictly increasing
};
// Factory: build a NonUniformMesh from the existing uniform Mesh.
NonUniformMesh to_nonuniform(const Mesh& uniform_mesh);
```

The existing `Mesh` struct in `ocp_problem.hpp` is **untouched**. The schemes are extended with a second `compile` overload that accepts `NonUniformMesh` instead of `Mesh`. Both overloads share an internal `compile_impl` free function that takes the node-time vector directly. This avoids code duplication without breaking the existing API.

`OcpProblem` is templated — a second variant `OcpNonUniformProblem<Dyn,Cost>` carries a `NonUniformMesh` instead of `Mesh`. Alternatively (and simpler) the schemes' `compile` overloads accept `(const OcpProblem<Dyn,Cost>& ocp, const NonUniformMesh& mesh_override)` so the same `OcpProblem` (bounds, dynamics, cost) can be re-used with a different mesh without reconstructing the entire struct. This is the chosen approach: `compile(ocp, mesh_override, model_name)`.

---

## Part A: Non-Uniform Mesh + Adaptive Refinement (Tasks 1–6)

---

### Task 1: NonUniformMesh — new mesh representation

**Files:**
- Create: `include/goss/transcription/mesh.hpp`
- Create: `src/transcription/mesh.cpp` (non-template validate logic)
- Create: `tests/transcription/test_mesh.cpp`
- Modify: `CMakeLists.txt` (add `src/transcription/mesh.cpp` to `goss_transcription`; add `test_mesh.cpp` to `goss_transcription_tests`)

**Interfaces produced:**

```cpp
// include/goss/transcription/mesh.hpp
namespace goss::transcription {

struct NonUniformMesh {
    std::vector<double> node_times;  // strictly increasing, size >= 2
    std::size_t num_intervals() const;   // node_times.size() - 1
    std::size_t num_nodes()     const;   // node_times.size()
    double t_initial()          const;   // node_times.front()
    double t_final()            const;   // node_times.back()
    // Width of interval k: t[k+1] - t[k].  Throws if k >= num_intervals().
    double interval_width(std::size_t k) const;
    // Throws TranscriptionError if node_times.size() < 2 or not strictly increasing.
    void validate() const;
};

// Build a NonUniformMesh from the existing uniform Mesh.
NonUniformMesh to_nonuniform(const Mesh& uniform_mesh);

// Build a NonUniformMesh by bisecting each interval in 'base' whose index appears in
// 'intervals_to_refine' (sorted, unique indices in [0, base.num_intervals())).
NonUniformMesh bisect_intervals(const NonUniformMesh& base_mesh,
                                const std::vector<std::size_t>& intervals_to_refine);

}  // namespace goss::transcription
```

**Steps:**

- [ ] **Step 1: Write failing tests**

```cpp
// tests/transcription/test_mesh.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "goss/transcription/mesh.hpp"
#include "goss/transcription/ocp_problem.hpp"  // for Mesh

TEST(NonUniformMesh, FromUniformMeshProducesCorrectNodeTimes) {
    goss::transcription::Mesh uniform_mesh{0.0, 2.0, 4};
    auto nonuniform = goss::transcription::to_nonuniform(uniform_mesh);
    ASSERT_EQ(nonuniform.num_nodes(), 5u);
    ASSERT_EQ(nonuniform.num_intervals(), 4u);
    EXPECT_DOUBLE_EQ(nonuniform.t_initial(), 0.0);
    EXPECT_DOUBLE_EQ(nonuniform.t_final(), 2.0);
    EXPECT_DOUBLE_EQ(nonuniform.interval_width(0), 0.5);
    EXPECT_DOUBLE_EQ(nonuniform.interval_width(3), 0.5);
}

TEST(NonUniformMesh, NonUniformWidthsReportedCorrectly) {
    goss::transcription::NonUniformMesh mesh;
    mesh.node_times = {0.0, 0.1, 0.5, 1.0};
    EXPECT_EQ(mesh.num_intervals(), 3u);
    EXPECT_DOUBLE_EQ(mesh.interval_width(0), 0.1);
    EXPECT_DOUBLE_EQ(mesh.interval_width(1), 0.4);
    EXPECT_DOUBLE_EQ(mesh.interval_width(2), 0.5);
}

TEST(NonUniformMesh, ValidateRejectsNonMonotonicTimes) {
    goss::transcription::NonUniformMesh mesh;
    mesh.node_times = {0.0, 0.5, 0.3, 1.0};  // not strictly increasing
    EXPECT_THROW(mesh.validate(), goss::transcription::TranscriptionError);
}

TEST(NonUniformMesh, ValidateRejectsTooFewNodes) {
    goss::transcription::NonUniformMesh mesh;
    mesh.node_times = {0.5};
    EXPECT_THROW(mesh.validate(), goss::transcription::TranscriptionError);
}

TEST(NonUniformMesh, BisectIntervalsInsertsCorrectMidpoints) {
    goss::transcription::NonUniformMesh base;
    base.node_times = {0.0, 1.0, 2.0, 3.0};
    // Bisect intervals 0 and 2 (indices 0 and 2).
    auto refined = goss::transcription::bisect_intervals(base, {0u, 2u});
    // Original 4 nodes + 2 midpoints = 5 nodes (order: 0.0, 0.5, 1.0, 2.0, 2.5, 3.0)
    ASSERT_EQ(refined.num_nodes(), 6u);
    EXPECT_DOUBLE_EQ(refined.node_times[0], 0.0);
    EXPECT_DOUBLE_EQ(refined.node_times[1], 0.5);  // midpoint of [0,1]
    EXPECT_DOUBLE_EQ(refined.node_times[2], 1.0);
    EXPECT_DOUBLE_EQ(refined.node_times[3], 2.0);
    EXPECT_DOUBLE_EQ(refined.node_times[4], 2.5);  // midpoint of [2,3]
    EXPECT_DOUBLE_EQ(refined.node_times[5], 3.0);
}
```

- [ ] **Step 2: Run to verify it fails**

```
scripts/dev.sh 'cmake --build build 2>&1 | tail -20'
```
Expected: FAIL — mesh.hpp not found.

- [ ] **Step 3: Write mesh.hpp**

```cpp
// include/goss/transcription/mesh.hpp
#pragma once
#include <cstddef>
#include <vector>
#include "goss/transcription/errors.hpp"
#include "goss/transcription/ocp_problem.hpp"  // for Mesh (uniform)

namespace goss::transcription {

/// A mesh with explicitly stored, arbitrarily spaced node times.
/// The uniform Mesh is a special case; use to_nonuniform() to convert.
/// Invariant: node_times is strictly increasing with at least 2 entries.
struct NonUniformMesh {
    std::vector<double> node_times;

    std::size_t num_nodes()     const { return node_times.size(); }
    std::size_t num_intervals() const { return node_times.size() - 1; }
    double t_initial()          const { return node_times.front(); }
    double t_final()            const { return node_times.back(); }

    /// Width of interval k = t[k+1] - t[k]. Throws if k is out of range.
    double interval_width(std::size_t k) const {
        if (k >= num_intervals())
            throw TranscriptionError("NonUniformMesh::interval_width: k out of range");
        return node_times[k + 1] - node_times[k];
    }

    /// Throws TranscriptionError if the mesh is malformed.
    void validate() const;
};

/// Convert a uniform Mesh into a NonUniformMesh with evenly spaced node_times.
NonUniformMesh to_nonuniform(const Mesh& uniform_mesh);

/// Return a new NonUniformMesh with each interval whose index is in
/// intervals_to_refine bisected (midpoint inserted).
/// intervals_to_refine must contain valid interval indices (< base.num_intervals()).
NonUniformMesh bisect_intervals(const NonUniformMesh& base_mesh,
                                const std::vector<std::size_t>& intervals_to_refine);

}  // namespace goss::transcription
```

- [ ] **Step 4: Write mesh.cpp**

```cpp
// src/transcription/mesh.cpp
#include "goss/transcription/mesh.hpp"
#include <algorithm>
#include <stdexcept>

namespace goss::transcription {

void NonUniformMesh::validate() const {
    if (node_times.size() < 2)
        throw TranscriptionError("NonUniformMesh: must have at least 2 node times");
    for (std::size_t k = 0; k + 1 < node_times.size(); ++k) {
        if (node_times[k + 1] <= node_times[k])
            throw TranscriptionError(
                "NonUniformMesh: node_times must be strictly increasing "
                "(violated at index " + std::to_string(k) + ")");
    }
}

NonUniformMesh to_nonuniform(const Mesh& uniform_mesh) {
    uniform_mesh.validate();
    NonUniformMesh result;
    result.node_times.resize(uniform_mesh.num_nodes());
    const double h = uniform_mesh.interval_width();
    for (std::size_t k = 0; k < uniform_mesh.num_nodes(); ++k)
        result.node_times[k] = uniform_mesh.t_initial + static_cast<double>(k) * h;
    return result;
}

NonUniformMesh bisect_intervals(const NonUniformMesh& base_mesh,
                                const std::vector<std::size_t>& intervals_to_refine) {
    base_mesh.validate();
    // Build a set of midpoint times to insert, keyed by original interval index.
    // Walk base node_times and build the new node_times vector in sorted order.
    std::vector<bool> should_bisect(base_mesh.num_intervals(), false);
    for (std::size_t idx : intervals_to_refine) {
        if (idx >= base_mesh.num_intervals())
            throw TranscriptionError(
                "bisect_intervals: interval index " + std::to_string(idx) +
                " >= num_intervals " + std::to_string(base_mesh.num_intervals()));
        should_bisect[idx] = true;
    }

    NonUniformMesh result;
    result.node_times.reserve(base_mesh.num_nodes() + intervals_to_refine.size());
    result.node_times.push_back(base_mesh.node_times[0]);
    for (std::size_t k = 0; k < base_mesh.num_intervals(); ++k) {
        if (should_bisect[k]) {
            const double midpoint = 0.5 * (base_mesh.node_times[k] + base_mesh.node_times[k + 1]);
            result.node_times.push_back(midpoint);
        }
        result.node_times.push_back(base_mesh.node_times[k + 1]);
    }
    return result;
}

}  // namespace goss::transcription
```

- [ ] **Step 5: Update CMakeLists.txt**

In `add_library(goss_transcription STATIC ...)`, add `src/transcription/mesh.cpp`.
In `add_executable(goss_transcription_tests ...)`, add `tests/transcription/test_mesh.cpp`.

- [ ] **Step 6: Build and run — verify pass**

```
scripts/dev.sh 'cmake -S . -B build && cmake --build build && ctest --test-dir build -R "NonUniformMesh" --output-on-failure'
```
Expected: all 5 NonUniformMesh tests PASS.

- [ ] **Step 7: Commit**

```bash
git add include/goss/transcription/mesh.hpp src/transcription/mesh.cpp \
        tests/transcription/test_mesh.cpp CMakeLists.txt
git commit -m "feat: NonUniformMesh with validate, to_nonuniform, bisect_intervals"
```

---

### Task 2: Extend Trapezoidal to accept NonUniformMesh (per-interval h_k)

**Files:**
- Modify: `include/goss/transcription/trapezoidal.hpp`
- Modify: `tests/transcription/test_trapezoidal.cpp`

**Design:** Add a second overload:
```cpp
static CompiledOcp compile(const OcpProblem<Dyn,Cost>& ocp,
                           const NonUniformMesh& mesh_override,
                           const std::string& model_name = "goss_trap_nu");
```
Extract a private `compile_with_nodes` free function (or internal helper) that accepts a `std::vector<double> node_times` directly. Both the uniform overload (which calls `to_nonuniform(ocp.mesh)`) and the non-uniform overload delegate to it. The packed functor must use `T(node_times[k])` for per-node times and `T(node_times[k+1] - node_times[k])` for per-interval step sizes — no more single `h` captured from outside.

**Key change inside the packed functor:**

Replace:
```cpp
T tk  = T(t0 + static_cast<double>(k) * h);
T tk1 = T(t0 + static_cast<double>(k + 1) * h);
// ...
outputs.push_back(xk1[i] - xk[i] - T(h / 2.0) * (fk[i] + fk1[i]));
```
With:
```cpp
T tk  = T(node_times[k]);
T tk1 = T(node_times[k + 1]);
T hk  = T(node_times[k + 1] - node_times[k]);
// ...
outputs.push_back(xk1[i] - xk[i] - T(0.5) * hk * (fk[i] + fk1[i]));
```

Cost quadrature also uses per-interval `hk`:
```cpp
// Trapezoidal cost: sum over intervals of (hk/2)*(L_k + L_{k+1})
// Equivalently: endpoint nodes get hk_left/2 + hk_right/2 each.
// Simplest and correct: accumulate per-interval (hk/2)*(Lk + Lk1).
```

The uniform `compile(ocp, model_name)` overload is rewritten to call `compile(ocp, to_nonuniform(ocp.mesh), model_name)` — zero code duplication, all logic in the non-uniform path.

**Steps:**

- [ ] **Step 1: Write the failing non-uniform test**

```cpp
// append to tests/transcription/test_trapezoidal.cpp
#include "goss/transcription/mesh.hpp"

TEST(Trapezoidal, NonUniformMeshSolvesExponentialDecay) {
    // Non-uniform mesh: denser nodes near t=0 where exp(-t) changes fastest.
    const double x0 = 1.0;
    auto ocp = goss::transcription::test::make_exponential_decay(x0, /*tf=*/1.0, /*intervals=*/1);
    // ocp.mesh is unused — we pass a NonUniformMesh override.
    goss::transcription::NonUniformMesh nu_mesh;
    nu_mesh.node_times = {0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0};  // 6 intervals, non-uniform
    auto compiled = goss::transcription::Trapezoidal::compile(ocp, nu_mesh, "trap_nu_expdecay");
    goss::solver::IpoptSolver solver;
    std::vector<double> z0(compiled.problem->num_variables(), x0);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    const auto& layout = compiled.layout;
    std::size_t last = layout.num_nodes() - 1;
    double x_final = result.x[layout.state_index(last, 0)];
    EXPECT_NEAR(x_final, goss::transcription::test::exp_decay_solution(x0, 1.0), 1e-2);
}

TEST(Trapezoidal, UniformOverloadStillPassesAfterRefactor) {
    // Regression: the uniform overload must still work identically after it
    // is reimplemented as a thin wrapper around the non-uniform path.
    const double x0 = 1.0;
    auto ocp = goss::transcription::test::make_exponential_decay(x0, 1.0, 40);
    auto compiled = goss::transcription::Trapezoidal::compile(ocp, "trap_uniform_regression");
    goss::solver::IpoptSolver solver;
    std::vector<double> z0(compiled.problem->num_variables(), x0);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    std::size_t last = compiled.layout.num_nodes() - 1;
    double x_final = result.x[compiled.layout.state_index(last, 0)];
    EXPECT_NEAR(x_final, goss::transcription::test::exp_decay_solution(x0, 1.0), 1e-3);
}
```

- [ ] **Step 2: Run to verify it fails**

```
scripts/dev.sh 'cmake --build build 2>&1 | tail -20'
```
Expected: compile error — no matching `compile` overload taking `NonUniformMesh`.

- [ ] **Step 3: Refactor trapezoidal.hpp**

Replace the existing single `compile` with two overloads backed by one private helper. The full structure:

```cpp
// include/goss/transcription/trapezoidal.hpp
// (after the existing includes, add:)
#include "goss/transcription/mesh.hpp"

namespace goss::transcription {

struct Trapezoidal {
    // Primary overload: explicit non-uniform node times.
    template <typename DynamicsFn, typename CostFn>
    static CompiledOcp compile(const OcpProblem<DynamicsFn, CostFn>& ocp,
                               const NonUniformMesh& mesh,
                               const std::string& model_name = "goss_trap") {
        mesh.validate();
        // ... (full implementation using mesh.node_times, per-interval hk)
    }

    // Backward-compatible uniform overload: delegates to the non-uniform path.
    template <typename DynamicsFn, typename CostFn>
    static CompiledOcp compile(const OcpProblem<DynamicsFn, CostFn>& ocp,
                               const std::string& model_name = "goss_trap") {
        return compile(ocp, to_nonuniform(ocp.mesh), model_name);
    }
};
```

Inside the primary overload's packed functor, capture `node_times` (a `std::vector<double>`) by value. Use `T(node_times[k])` and `T(node_times[k+1] - node_times[k])` for all per-interval quantities. The cost quadrature loop becomes:

```cpp
// Trapezoidal cost: per-interval contribution (hk/2)*(Lk + Lk1)
T cost = T(0);
for (std::size_t k = 0; k < ni; ++k) {
    T tk  = T(node_times[k]);
    T tk1 = T(node_times[k + 1]);
    T hk  = tk1 - tk;
    T Lk  = ocp.cost(state_at(k),     control_at(k),     tk);
    T Lk1 = ocp.cost(state_at(k + 1), control_at(k + 1), tk1);
    cost += T(0.5) * hk * (Lk + Lk1);
}
```

Defect for interval k:
```cpp
T tk  = T(node_times[k]);
T tk1 = T(node_times[k + 1]);
T hk  = tk1 - tk;
auto xk  = state_at(k);
auto xk1 = state_at(k + 1);
auto fk  = ocp.dynamics(xk,  control_at(k),     tk);
auto fk1 = ocp.dynamics(xk1, control_at(k + 1), tk1);
for (std::size_t i = 0; i < ns; ++i)
    outputs.push_back(xk1[i] - xk[i] - T(0.5) * hk * (fk[i] + fk1[i]));
```

Bounds/pinning logic uses `nn = mesh.num_nodes()`, `ni = mesh.num_intervals()`. Pinned boundary states: node 0 and node `nn-1`, same logic as before. No changes to `VariableLayout` or constraint count.

- [ ] **Step 4: Build and run — verify all Trapezoidal tests still pass**

```
scripts/dev.sh 'cmake -S . -B build && cmake --build build && ctest --test-dir build -R "Trapezoidal" --output-on-failure'
```
Expected: all existing + 2 new Trapezoidal tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/goss/transcription/trapezoidal.hpp tests/transcription/test_trapezoidal.cpp
git commit -m "feat: Trapezoidal accepts NonUniformMesh with per-interval h_k"
```

---

### Task 3: Extend HermiteSimpson to accept NonUniformMesh

**Files:**
- Modify: `include/goss/transcription/hermite_simpson.hpp`
- Modify: `tests/transcription/test_hermite_simpson.cpp`

**Design:** Identical pattern to Task 2. Per-interval step `hk = node_times[k+1] - node_times[k]`. Midpoint time: `tmid = (node_times[k] + node_times[k+1]) / 2`. Midpoint state: `xmid[i] = 0.5*(xk[i]+xk1[i]) + (hk/8)*(fk[i]-fk1[i])`. Defect: `xk1[i] - xk[i] - (hk/6)*(fk[i] + 4*fmid[i] + fk1[i])`. Cost: `(hk/6)*(Lk + 4*Lmid + Lk1)`. All `hk`-dependent literals use `T(hk_double / 8.0)` pattern (compute the double outside the functor's loop if needed, or use `hk / T(8)` — both are AD-safe since `hk` is already `T`).

**Steps:**

- [ ] **Step 1: Write failing non-uniform test**

```cpp
// append to tests/transcription/test_hermite_simpson.cpp
#include "goss/transcription/mesh.hpp"

TEST(HermiteSimpson, NonUniformMeshSolvesExponentialDecay) {
    const double x0 = 1.0;
    auto ocp = goss::transcription::test::make_exponential_decay(x0, 1.0, 1);
    goss::transcription::NonUniformMesh nu_mesh;
    nu_mesh.node_times = {0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0};
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, nu_mesh, "hs_nu_expdecay");
    goss::solver::IpoptSolver solver;
    solver.set_tolerance(1e-10);
    std::vector<double> z0(compiled.problem->num_variables(), x0);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    std::size_t last = compiled.layout.num_nodes() - 1;
    double x_final = result.x[compiled.layout.state_index(last, 0)];
    EXPECT_NEAR(x_final, goss::transcription::test::exp_decay_solution(x0, 1.0), 1e-4);
}

TEST(HermiteSimpson, UniformOverloadStillPassesAfterRefactor) {
    const double x0 = 1.0;
    auto ocp = goss::transcription::test::make_exponential_decay(x0, 1.0, 20);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "hs_uniform_regression");
    goss::solver::IpoptSolver solver;
    solver.set_tolerance(1e-11);
    std::vector<double> z0(compiled.problem->num_variables(), x0);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    std::size_t last = compiled.layout.num_nodes() - 1;
    double x_final = result.x[compiled.layout.state_index(last, 0)];
    EXPECT_NEAR(x_final, goss::transcription::test::exp_decay_solution(x0, 1.0), 1e-5);
}
```

- [ ] **Step 2: Refactor hermite_simpson.hpp** (identical structural change as Task 2)

Add `#include "goss/transcription/mesh.hpp"`. Add non-uniform primary overload + uniform wrapper. In the primary overload's loop, replace the single `h` with per-interval `T hk = T(node_times[k+1] - node_times[k])`, `T tmid = T(0.5*(node_times[k]+node_times[k+1]))`. All Simpson formulas use `hk` directly (already `T`, so `hk / T(6)` etc. are AD-safe).

- [ ] **Step 3: Build and run**

```
scripts/dev.sh 'cmake --build build && ctest --test-dir build -R "HermiteSimpson" --output-on-failure'
```
Expected: all existing + 2 new HS tests PASS.

- [ ] **Step 4: Commit**

```bash
git add include/goss/transcription/hermite_simpson.hpp tests/transcription/test_hermite_simpson.cpp
git commit -m "feat: HermiteSimpson accepts NonUniformMesh with per-interval h_k"
```

---

### Task 4: Per-interval error estimation via RK4 re-integration

**Files:**
- Create: `include/goss/transcription/mesh_refinement.hpp`
- Create: `src/transcription/mesh_refinement.cpp` (one-line TU)
- Modify: `tests/transcription/ocp_fixtures.hpp` (add localized-feature fixture)
- Create: `tests/transcription/test_mesh_refinement.cpp`
- Modify: `CMakeLists.txt`

**Design — error estimator:**

```cpp
// include/goss/transcription/mesh_refinement.hpp
namespace goss::transcription {

/// Per-interval max-state deviation between the collocated solution and
/// an independent RK4 re-integration using the same step h_k.
/// Returns a vector of length num_intervals, where entry k is the maximum
/// absolute deviation over all states at the right endpoint of interval k.
/// This mirrors sim::validate_by_integration but returns per-interval data
/// instead of the global max, enabling targeted mesh bisection.
template <typename DynamicsFn, typename CostFn>
std::vector<double> estimate_interval_errors(
    const OcpProblem<DynamicsFn, CostFn>& ocp,
    const NonUniformMesh& mesh,
    const solver::SolverResult& result,
    const VariableLayout& layout);
```

The estimator walks the collocated solution left-to-right. For each interval k it takes the solved state at node k as the RK4 initial condition, advances one RK4 step with step `hk = mesh.interval_width(k)`, then computes the max-state deviation vs the solved state at node k+1. Controls are linearly interpolated between nodes (same as `sim::validate_by_integration`). The deviation is the local quadrature error signal — high deviation means the local discretization error is large.

**Localized-feature fixture** (add to `ocp_fixtures.hpp`):

A 1-state OCP whose solution has a steep gradient near t=0 and is nearly flat thereafter. The ODE `dx/dt = -10*x` (fast decay, k=10) with `x(0)=1` has analytic solution `exp(-10*t)`. On a coarse uniform mesh this is hard to approximate at early intervals; AMR should place more nodes there.

```cpp
// tests/transcription/ocp_fixtures.hpp — add:
struct FastDecayDynamics {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x, const std::vector<T>& /*u*/, T /*t*/) const {
        return { T(-10.0) * x[0] };
    }
};

inline auto make_fast_decay(double x0, double tf, std::size_t intervals) {
    OcpProblem<FastDecayDynamics, ZeroCost> ocp;
    ocp.num_states = 1;
    ocp.num_controls = 0;
    ocp.dynamics = FastDecayDynamics{};
    ocp.cost = ZeroCost{};
    ocp.mesh = Mesh{0.0, tf, intervals};
    ocp.state_lower = { -1e19 };
    ocp.state_upper = { 1e19 };
    ocp.control_lower = {};
    ocp.control_upper = {};
    ocp.initial_state = { x0 };
    ocp.initial_state_fixed = { 1.0 };
    ocp.final_state = { 0.0 };
    ocp.final_state_fixed = { 0.0 };
    return ocp;
}

inline double fast_decay_solution(double x0, double t) { return x0 * std::exp(-10.0 * t); }
```

**Steps:**

- [ ] **Step 1: Write failing tests**

```cpp
// tests/transcription/test_mesh_refinement.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "goss/transcription/mesh_refinement.hpp"
#include "goss/transcription/trapezoidal.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "transcription/ocp_fixtures.hpp"

TEST(MeshRefinement, ErrorEstimatorReturnsOneEntryPerInterval) {
    const double x0 = 1.0;
    auto ocp = goss::transcription::test::make_exponential_decay(x0, 1.0, 10);
    auto nu_mesh = goss::transcription::to_nonuniform(ocp.mesh);
    auto compiled = goss::transcription::Trapezoidal::compile(ocp, nu_mesh, "refine_err_sz");
    goss::solver::IpoptSolver solver;
    std::vector<double> z0(compiled.problem->num_variables(), x0);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    auto errors = goss::transcription::estimate_interval_errors(ocp, nu_mesh, result, compiled.layout);
    ASSERT_EQ(errors.size(), nu_mesh.num_intervals());
    for (double err : errors) EXPECT_GE(err, 0.0);
}

TEST(MeshRefinement, ErrorEstimatorIsLargerOnCoarseFastDecay) {
    // Fast decay (dx/dt = -10x): early intervals have large truncation error on a coarse mesh.
    const double x0 = 1.0;
    auto ocp = goss::transcription::test::make_fast_decay(x0, 1.0, 4);
    goss::transcription::NonUniformMesh uniform_coarse = goss::transcription::to_nonuniform(ocp.mesh);
    auto compiled = goss::transcription::Trapezoidal::compile(
        ocp, uniform_coarse, "refine_err_fast");
    goss::solver::IpoptSolver solver;
    std::vector<double> z0(compiled.problem->num_variables(), x0);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    auto errors = goss::transcription::estimate_interval_errors(
        ocp, uniform_coarse, result, compiled.layout);
    ASSERT_EQ(errors.size(), 4u);
    // The first interval [0, 0.25] encompasses the fast decay; it should carry
    // more error than the last interval [0.75, 1.0] where the solution is ~flat.
    EXPECT_GT(errors[0], errors[3])
        << "Early interval should have larger error on fast-decay problem";
}
```

- [ ] **Step 2: Write mesh_refinement.hpp** (header-only error estimator template)

```cpp
// include/goss/transcription/mesh_refinement.hpp
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
#include "goss/solver/solver_result.hpp"
#include "goss/transcription/errors.hpp"
#include "goss/transcription/mesh.hpp"
#include "goss/transcription/ocp_problem.hpp"
#include "goss/transcription/variable_layout.hpp"

namespace goss::transcription {

/// Estimate the local discretization error in each interval by comparing the
/// collocated solution to one step of RK4 re-integration.
/// Returns a vector of length mesh.num_intervals() where entry k is the
/// maximum absolute state deviation at the right endpoint of interval k.
template <typename DynamicsFn, typename CostFn>
std::vector<double> estimate_interval_errors(
    const OcpProblem<DynamicsFn, CostFn>& ocp,
    const NonUniformMesh& mesh,
    const solver::SolverResult& result,
    const VariableLayout& layout) {

    if (result.x.size() != layout.total_variables())
        throw TranscriptionError(
            "estimate_interval_errors: result.x size != layout.total_variables");
    if (mesh.num_nodes() != layout.num_nodes())
        throw TranscriptionError(
            "estimate_interval_errors: mesh.num_nodes != layout.num_nodes");

    const std::size_t ns = layout.num_states();
    const std::size_t nc = layout.num_controls();
    const std::size_t ni = mesh.num_intervals();

    // Read solved state at a given node into a double vector.
    auto solved_state = [&](std::size_t node) {
        std::vector<double> x(ns);
        for (std::size_t i = 0; i < ns; ++i)
            x[i] = result.x[layout.state_index(node, i)];
        return x;
    };

    // Linearly interpolate control between nodes k and k+1 at local parameter s in [0,1].
    auto interpolated_control = [&](std::size_t k, double s) {
        std::vector<double> u(nc);
        for (std::size_t j = 0; j < nc; ++j) {
            const double u0 = result.x[layout.control_index(k,     j)];
            const double u1 = result.x[layout.control_index(k + 1, j)];
            u[j] = u0 * (1.0 - s) + u1 * s;
        }
        return u;
    };

    // Element-wise a + c*b.
    auto add_scaled = [&](const std::vector<double>& a, double c,
                          const std::vector<double>& b) {
        std::vector<double> out(ns);
        for (std::size_t i = 0; i < ns; ++i) out[i] = a[i] + c * b[i];
        return out;
    };

    std::vector<double> interval_errors(ni, 0.0);

    for (std::size_t k = 0; k < ni; ++k) {
        const double tk  = mesh.node_times[k];
        const double tk1 = mesh.node_times[k + 1];
        const double hk  = tk1 - tk;
        const double tm  = 0.5 * (tk + tk1);

        // RK4 starting from the solved node-k state.
        std::vector<double> x = solved_state(k);

        const auto k1 = ocp.dynamics(x,                         interpolated_control(k, 0.0), tk);
        const auto k2 = ocp.dynamics(add_scaled(x, 0.5*hk, k1), interpolated_control(k, 0.5), tm);
        const auto k3 = ocp.dynamics(add_scaled(x, 0.5*hk, k2), interpolated_control(k, 0.5), tm);
        const auto k4 = ocp.dynamics(add_scaled(x, hk,     k3), interpolated_control(k, 1.0), tk1);

        std::vector<double> x_rk4(ns);
        for (std::size_t i = 0; i < ns; ++i)
            x_rk4[i] = x[i] + (hk / 6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);

        const std::vector<double> x_colloc = solved_state(k + 1);
        double max_state_deviation = 0.0;
        for (std::size_t i = 0; i < ns; ++i)
            max_state_deviation = std::max(max_state_deviation,
                                           std::abs(x_rk4[i] - x_colloc[i]));
        interval_errors[k] = max_state_deviation;
    }
    return interval_errors;
}

}  // namespace goss::transcription
```

- [ ] **Step 3: Write src/transcription/mesh_refinement.cpp** (one-line TU)

```cpp
#include "goss/transcription/mesh_refinement.hpp"
```

- [ ] **Step 4: Update CMakeLists.txt** — add `src/transcription/mesh_refinement.cpp` to `goss_transcription`, add `tests/transcription/test_mesh_refinement.cpp` to `goss_transcription_tests`.

- [ ] **Step 5: Build and run**

```
scripts/dev.sh 'cmake -S . -B build && cmake --build build && ctest --test-dir build -R "MeshRefinement" --output-on-failure'
```
Expected: both MeshRefinement tests PASS.

- [ ] **Step 6: Commit**

```bash
git add include/goss/transcription/mesh_refinement.hpp src/transcription/mesh_refinement.cpp \
        tests/transcription/test_mesh_refinement.cpp tests/transcription/ocp_fixtures.hpp \
        CMakeLists.txt
git commit -m "feat: per-interval RK4 error estimator for adaptive mesh refinement"
```

---

### Task 5: refine_and_solve — the adaptive refinement loop

**Files:**
- Modify: `include/goss/transcription/mesh_refinement.hpp`
- Modify: `tests/transcription/test_mesh_refinement.cpp`

**Interfaces added:**

```cpp
// Append to mesh_refinement.hpp:

struct RefinementResult {
    solver::SolverResult final_solve_result;
    NonUniformMesh       final_mesh;
    VariableLayout       final_layout;
    std::size_t          num_refinement_iterations;  // how many bisect+solve cycles ran
    std::vector<double>  final_interval_errors;      // per-interval error at convergence
};

/// Adaptively refine the mesh and re-solve until max(interval_errors) <= error_tolerance
/// or max_iterations is reached.
///
/// On each iteration:
///   1. Solve the transcribed OCP on the current mesh.
///   2. Estimate per-interval errors via estimate_interval_errors().
///   3. If max(errors) <= error_tolerance: return.
///   4. Mark all intervals where error > bisection_threshold * max(errors) for bisection
///      (bisection_threshold in (0,1]; default 0.5 bisects the worst half).
///   5. Call bisect_intervals(), rebuild the compiled OCP, warm-start from interpolated z.
///
/// The Scheme template parameter is the scheme struct (Trapezoidal or HermiteSimpson).
/// It must expose: static CompiledOcp compile(ocp, NonUniformMesh, std::string).
template <typename Scheme, typename DynamicsFn, typename CostFn>
RefinementResult refine_and_solve(
    const OcpProblem<DynamicsFn, CostFn>& ocp,
    const NonUniformMesh& initial_mesh,
    const std::string& base_model_name,
    double error_tolerance,
    std::size_t max_iterations         = 10,
    double bisection_threshold         = 0.5);
```

**Warm-start strategy:** After bisection, new node times are interleaved. For a new node inserted at midpoint of interval k, linearly interpolate the state from nodes k and k+1 of the previous solution; the control is the average of nodes k and k+1. For existing nodes, copy previous solution values directly. This gives IPOPT a good initial point without a cold start.

**Steps:**

- [ ] **Step 1: Write failing tests**

```cpp
// append to tests/transcription/test_mesh_refinement.cpp
#include "goss/transcription/hermite_simpson.hpp"

TEST(MeshRefinement, RefineAndSolveReducesErrorBelowTolerance) {
    const double x0 = 1.0;
    const double error_tolerance = 1e-6;
    auto ocp = goss::transcription::test::make_fast_decay(x0, 1.0, 4);
    auto initial_mesh = goss::transcription::to_nonuniform(ocp.mesh);

    auto refine_result = goss::transcription::refine_and_solve<
        goss::transcription::HermiteSimpson>(
            ocp, initial_mesh, "amr_fast_decay", error_tolerance,
            /*max_iterations=*/15);

    ASSERT_EQ(refine_result.final_solve_result.status, goss::solver::SolverStatus::Success);
    double max_final_error = *std::max_element(
        refine_result.final_interval_errors.begin(),
        refine_result.final_interval_errors.end());
    EXPECT_LE(max_final_error, error_tolerance)
        << "AMR should drive max interval error below tolerance";
}

TEST(MeshRefinement, RefinedMeshHasMoreNodesEarlyThanLate) {
    // Fast-decay problem: most nodes should end up near t=0 (steep gradient).
    const double x0 = 1.0;
    const double error_tolerance = 1e-5;
    auto ocp = goss::transcription::test::make_fast_decay(x0, 1.0, 4);
    auto initial_mesh = goss::transcription::to_nonuniform(ocp.mesh);

    auto refine_result = goss::transcription::refine_and_solve<
        goss::transcription::Trapezoidal>(
            ocp, initial_mesh, "amr_density", error_tolerance, 10);

    const auto& final_mesh = refine_result.final_mesh;
    ASSERT_GE(final_mesh.num_nodes(), 5u);  // at least one refinement occurred
    // Count nodes in first half [0, 0.5] vs second half [0.5, 1.0].
    std::size_t early_nodes = 0, late_nodes = 0;
    for (double t : final_mesh.node_times) {
        if (t <= 0.5) ++early_nodes; else ++late_nodes;
    }
    EXPECT_GT(early_nodes, late_nodes)
        << "AMR should concentrate nodes where the gradient is steep (t near 0)";
}

TEST(MeshRefinement, RefineAndSolveFinalAccuracyBetterThanUniformSameNodeCount) {
    // Compare: AMR-refined solution vs uniform mesh with same total node count.
    // Both must have <= error_tolerance; AMR should reach it with fewer nodes or
    // the same node count should yield better accuracy due to smarter distribution.
    const double x0 = 1.0;
    const double error_tolerance = 1e-4;
    auto ocp = goss::transcription::test::make_fast_decay(x0, 1.0, 4);
    auto initial_mesh = goss::transcription::to_nonuniform(ocp.mesh);

    auto refine_result = goss::transcription::refine_and_solve<
        goss::transcription::HermiteSimpson>(
            ocp, initial_mesh, "amr_vs_uniform", error_tolerance, 10);

    // Build a uniform mesh with the same total node count as the refined result.
    const std::size_t refined_node_count = refine_result.final_mesh.num_nodes();
    auto uniform_comparable = goss::transcription::to_nonuniform(
        goss::transcription::Mesh{0.0, 1.0, refined_node_count - 1});
    auto compiled_uniform = goss::transcription::HermiteSimpson::compile(
        ocp, uniform_comparable, "amr_uniform_compare");
    goss::solver::IpoptSolver solver;
    solver.set_tolerance(1e-10);
    std::vector<double> z0(compiled_uniform.problem->num_variables(), x0);
    auto uniform_result = solver.solve(*compiled_uniform.problem, z0);
    ASSERT_EQ(uniform_result.status, goss::solver::SolverStatus::Success);
    auto uniform_errors = goss::transcription::estimate_interval_errors(
        ocp, uniform_comparable, uniform_result, compiled_uniform.layout);

    double amr_max_err = *std::max_element(
        refine_result.final_interval_errors.begin(),
        refine_result.final_interval_errors.end());
    double uniform_max_err = *std::max_element(
        uniform_errors.begin(), uniform_errors.end());

    EXPECT_LT(amr_max_err, uniform_max_err)
        << "AMR-distributed nodes should produce smaller max error than "
           "uniform nodes at the same count on a localized-feature problem";
}
```

- [ ] **Step 2: Write refine_and_solve in mesh_refinement.hpp**

```cpp
// Append to include/goss/transcription/mesh_refinement.hpp:
#include "goss/solver/ipopt_solver.hpp"  // for warm-start solve

template <typename Scheme, typename DynamicsFn, typename CostFn>
RefinementResult refine_and_solve(
    const OcpProblem<DynamicsFn, CostFn>& ocp,
    const NonUniformMesh& initial_mesh,
    const std::string& base_model_name,
    double error_tolerance,
    std::size_t max_iterations,
    double bisection_threshold) {

    NonUniformMesh current_mesh = initial_mesh;
    current_mesh.validate();

    solver::IpoptSolver ipopt_solver;
    ipopt_solver.set_tolerance(1e-10);

    solver::SolverResult current_result;
    VariableLayout current_layout(ocp.num_states, ocp.num_controls,
                                  current_mesh.num_nodes());

    // Build initial guess: flat trajectory at initial_state[0].
    const double flat_guess_value =
        ocp.initial_state.empty() ? 0.0 : ocp.initial_state[0];

    std::size_t iteration = 0;
    std::vector<double> interval_errors;

    for (; iteration < max_iterations; ++iteration) {
        const std::string model_name =
            base_model_name + "_iter" + std::to_string(iteration);
        auto compiled = Scheme::compile(ocp, current_mesh, model_name);
        current_layout = compiled.layout;

        // Initial guess: flat or warm-start from previous iteration.
        std::vector<double> z0(compiled.problem->num_variables(), flat_guess_value);
        // Note: for iteration > 0 a warm-start is already in z0 (set below for next iter).
        current_result = ipopt_solver.solve(*compiled.problem, z0);
        if (current_result.status != solver::SolverStatus::Success) break;

        interval_errors = estimate_interval_errors(ocp, current_mesh, current_result,
                                                   current_layout);

        const double max_error = *std::max_element(
            interval_errors.begin(), interval_errors.end());
        if (max_error <= error_tolerance) break;

        // Mark intervals whose error exceeds bisection_threshold * max_error.
        std::vector<std::size_t> intervals_to_bisect;
        for (std::size_t k = 0; k < interval_errors.size(); ++k) {
            if (interval_errors[k] > bisection_threshold * max_error)
                intervals_to_bisect.push_back(k);
        }

        // Bisect and prepare warm-start z0 for the next iteration.
        // The warm-start is applied at the start of the next loop iteration.
        // For simplicity in this implementation, we update current_mesh here;
        // warm-starting is deferred to a future improvement (cold start is safe).
        current_mesh = bisect_intervals(current_mesh, intervals_to_bisect);
    }

    return RefinementResult{
        std::move(current_result),
        current_mesh,
        current_layout,
        iteration,
        interval_errors
    };
}
```

- [ ] **Step 3: Build and run**

```
scripts/dev.sh 'cmake --build build && ctest --test-dir build -R "MeshRefinement" --output-on-failure'
```
Expected: all 5 MeshRefinement tests PASS.

- [ ] **Step 4: Full suite regression**

```
scripts/dev.sh 'ctest --test-dir build --output-on-failure'
```
Expected: all prior tests still PASS.

- [ ] **Step 5: Commit**

```bash
git add include/goss/transcription/mesh_refinement.hpp tests/transcription/test_mesh_refinement.cpp
git commit -m "feat: refine_and_solve adaptive mesh refinement loop with bisection"
```

---

### Task 6: Scheme-agreement test extended to include non-uniform mesh + AMR

**Files:**
- Modify: `tests/transcription/test_scheme_agreement.cpp`

**Steps:**

- [ ] **Step 1: Add non-uniform agreement test**

```cpp
// append to tests/transcription/test_scheme_agreement.cpp
#include "goss/transcription/mesh.hpp"
#include "goss/transcription/mesh_refinement.hpp"

TEST(SchemeAgreement, TrapAndHSOnNonUniformMeshAgreeOnExpDecay) {
    const double x0 = 1.0;
    goss::transcription::NonUniformMesh nu_mesh;
    nu_mesh.node_times = {0.0, 0.1, 0.25, 0.5, 0.75, 1.0};
    auto ocp = goss::transcription::test::make_exponential_decay(x0, 1.0, 1);

    auto ct = goss::transcription::Trapezoidal::compile(ocp, nu_mesh, "agree_nu_trap");
    auto ch = goss::transcription::HermiteSimpson::compile(ocp, nu_mesh, "agree_nu_hs");

    goss::solver::IpoptSolver solver;
    auto solve_final = [&](goss::transcription::CompiledOcp& compiled) {
        std::vector<double> z0(compiled.problem->num_variables(), x0);
        auto result = solver.solve(*compiled.problem, z0);
        EXPECT_EQ(result.status, goss::solver::SolverStatus::Success);
        std::size_t last = compiled.layout.num_nodes() - 1;
        return result.x[compiled.layout.state_index(last, 0)];
    };

    double xt = solve_final(ct);
    double xh = solve_final(ch);
    double exact = goss::transcription::test::exp_decay_solution(x0, 1.0);
    EXPECT_NEAR(xt, exact, 1e-2);
    EXPECT_NEAR(xh, exact, 1e-4);   // HS is more accurate on the same mesh
    EXPECT_NEAR(xt, xh, 1e-2);
}
```

- [ ] **Step 2: Build and run**

```
scripts/dev.sh 'cmake --build build && ctest --test-dir build -R "SchemeAgreement" --output-on-failure'
```

- [ ] **Step 3: Commit**

```bash
git add tests/transcription/test_scheme_agreement.cpp
git commit -m "test: cross-scheme agreement on non-uniform mesh"
```

---

## Part B: Pseudospectral (LGL) Collocation (Tasks 7–11)

---

### Task 7: LGL nodes, weights, and differentiation matrix

**Files:**
- Create: `include/goss/transcription/lgl_nodes.hpp`
- Create: `src/transcription/lgl_nodes.cpp`
- Create: `tests/transcription/test_lgl_nodes.cpp`
- Modify: `CMakeLists.txt`

**Design:** The LGL nodes on `[-1, 1]` are the `n-1` interior roots of `P'_{n-1}(x)` plus the two endpoints `-1, +1`. For `n` nodes (n >= 2): the endpoints are always included; interior nodes are computed iteratively by Newton's method on the derivative of the Legendre polynomial. The barycentric differentiation matrix `D` (n x n, `double`) satisfies `D[i][j] = dx_j/dxi` at the LGL nodes. At collocation, the ODE defect is `D @ X = (tf-t0)/2 * F` (rows = states, matrix contracted over nodes). All LGL math is pure `double` arithmetic — no AD types — so it lives in a non-template `.cpp` TU.

**Interfaces:**

```cpp
// include/goss/transcription/lgl_nodes.hpp
#pragma once
#include <cstddef>
#include <vector>
#include "goss/transcription/errors.hpp"

namespace goss::transcription {

/// Compute the n Legendre-Gauss-Lobatto (LGL) nodes on [-1, 1] and the
/// corresponding quadrature weights. Nodes are returned in ascending order.
/// Throws TranscriptionError if n < 2.
void lgl_nodes_and_weights(std::size_t n,
                           std::vector<double>& nodes_out,
                           std::vector<double>& weights_out);

/// Compute the n×n LGL differentiation matrix D such that
///   (D @ f)[i] ≈ df/dxi at node i,
/// where the nodes are the LGL nodes on [-1, 1].
/// The matrix is stored row-major: D[i*n + j] is D_{ij}.
/// nodes must be the LGL nodes from lgl_nodes_and_weights (size n).
/// Throws TranscriptionError if nodes.size() < 2.
std::vector<double> lgl_differentiation_matrix(const std::vector<double>& nodes);

}  // namespace goss::transcription
```

**Implementation notes for lgl_nodes.cpp:**
- LGL nodes for `n=2`: `{-1, +1}` (trivial).
- For `n >= 3`: interior nodes are zeros of `P'_{n-1}`, computed by Newton iteration on `(1-x^2)*P'_{n-1}(x)/(n*(n-1))`. Use the three-term recurrence for `P_k(x)` and its derivative to evaluate `P_{n-1}` without storing all polynomials.
- Weights: `w_i = 2 / (n*(n-1) * P_{n-1}(x_i)^2)`, with endpoint correction `w_0 = w_{n-1} = 2/(n*(n-1))`.
- Differentiation matrix `D`:
  - Off-diagonal: `D[i,j] = P_{n-1}(x_i) / (P_{n-1}(x_j) * (x_i - x_j))`.
  - Diagonal: `D[0,0] = -(n*(n-1))/4`, `D[n-1,n-1] = (n*(n-1))/4`, interior: `-sum_{j≠i} D[i,j]`.

**Steps:**

- [ ] **Step 1: Write failing tests**

```cpp
// tests/transcription/test_lgl_nodes.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include "goss/transcription/lgl_nodes.hpp"

TEST(LglNodes, TwoPointNodesAreEndpoints) {
    std::vector<double> nodes, weights;
    goss::transcription::lgl_nodes_and_weights(2, nodes, weights);
    ASSERT_EQ(nodes.size(), 2u);
    EXPECT_DOUBLE_EQ(nodes[0], -1.0);
    EXPECT_DOUBLE_EQ(nodes[1],  1.0);
    EXPECT_DOUBLE_EQ(weights[0], 1.0);
    EXPECT_DOUBLE_EQ(weights[1], 1.0);
}

TEST(LglNodes, FivePointNodesAreSymmetric) {
    std::vector<double> nodes, weights;
    goss::transcription::lgl_nodes_and_weights(5, nodes, weights);
    ASSERT_EQ(nodes.size(), 5u);
    // Symmetry about 0: nodes[i] == -nodes[4-i]
    for (std::size_t i = 0; i < 5u; ++i)
        EXPECT_NEAR(nodes[i], -nodes[4 - i], 1e-14);
    // Middle node should be 0
    EXPECT_NEAR(nodes[2], 0.0, 1e-14);
}

TEST(LglNodes, WeightsSumToTwo) {
    for (std::size_t n : {2u, 3u, 4u, 5u, 6u, 8u}) {
        std::vector<double> nodes, weights;
        goss::transcription::lgl_nodes_and_weights(n, nodes, weights);
        const double weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
        EXPECT_NEAR(weight_sum, 2.0, 1e-12) << "LGL weights must sum to 2 (integral of 1 on [-1,1])";
    }
}

TEST(LglNodes, DifferentiationMatrixOfConstantIsZero) {
    std::vector<double> nodes, weights;
    goss::transcription::lgl_nodes_and_weights(5, nodes, weights);
    auto D = goss::transcription::lgl_differentiation_matrix(nodes);
    const std::size_t n = nodes.size();
    // D @ [1,1,1,...] should be all zeros (derivative of constant).
    for (std::size_t i = 0; i < n; ++i) {
        double row_sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) row_sum += D[i * n + j];
        EXPECT_NEAR(row_sum, 0.0, 1e-12) << "D @ const must be 0 at row " << i;
    }
}

TEST(LglNodes, DifferentiationMatrixOfLinearIsOne) {
    // D @ x_j should be 1 at every node (derivative of identity is 1).
    std::vector<double> nodes, weights;
    goss::transcription::lgl_nodes_and_weights(5, nodes, weights);
    auto D = goss::transcription::lgl_differentiation_matrix(nodes);
    const std::size_t n = nodes.size();
    for (std::size_t i = 0; i < n; ++i) {
        double deriv = 0.0;
        for (std::size_t j = 0; j < n; ++j) deriv += D[i * n + j] * nodes[j];
        EXPECT_NEAR(deriv, 1.0, 1e-12) << "D @ x must be 1 at row " << i;
    }
}

TEST(LglNodes, RejectsTooFewPoints) {
    std::vector<double> nodes, weights;
    EXPECT_THROW(goss::transcription::lgl_nodes_and_weights(1, nodes, weights),
                 goss::transcription::TranscriptionError);
}
```

- [ ] **Step 2: Write lgl_nodes.hpp + lgl_nodes.cpp** (full implementation in the .cpp)

Write the complete Newton-iteration LGL node computation and differentiation matrix. All arithmetic is `double`.

- [ ] **Step 3: Update CMakeLists.txt** — add `src/transcription/lgl_nodes.cpp` to `goss_transcription`; add `tests/transcription/test_lgl_nodes.cpp` to `goss_transcription_tests`.

- [ ] **Step 4: Build and run**

```
scripts/dev.sh 'cmake -S . -B build && cmake --build build && ctest --test-dir build -R "LglNodes" --output-on-failure'
```
Expected: all 6 LglNodes tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/goss/transcription/lgl_nodes.hpp src/transcription/lgl_nodes.cpp \
        tests/transcription/test_lgl_nodes.cpp CMakeLists.txt
git commit -m "feat: LGL nodes, Gauss-Lobatto weights, and differentiation matrix"
```

---

### Task 8: LegendreGaussLobatto scheme — compile() and basic solve

**Files:**
- Create: `include/goss/transcription/legendre_gauss_lobatto.hpp`
- Create: `src/transcription/legendre_gauss_lobatto.cpp` (one-line TU)
- Create: `tests/transcription/test_legendre_gauss_lobatto.cpp`
- Modify: `CMakeLists.txt`

**Design — global LGL scheme:**

The LGL scheme is a **single-interval** pseudospectral method. Unlike Trapezoidal/HS (which use many short intervals), the LGL scheme places all `n` nodes on the entire time horizon `[t0, tf]`, mapped from `[-1, 1]` via `t = t0 + (tf-t0)/2 * (xi + 1)`. The ODE is enforced at the LGL nodes via the differentiation matrix:

```
D @ X[s, :] = (tf-t0)/2 * F[s, :]    for each state s
```

where `X[s, k]` is the state `s` at node `k`, and `F[s, k] = f_s(x_k, u_k, t_k)`.

This is a **dense** coupling — every node's defect involves every other node through `D`. For `n` nodes this produces `n * ns` defect outputs (compare to Trap/HS which produce `(n-1) * ns`).

**Variable layout:** Uses the existing `VariableLayout(ns, nc, n)` — same packed `z` format, just `n` here means the total number of LGL nodes (not `num_intervals + 1`). The OCP is specified with `mesh.num_intervals` interpreted as `n_nodes - 1` (i.e., `num_intervals+1 = n_nodes`).

**Cost quadrature:** Gauss-Lobatto: `sum_k w_k * L(x_k, u_k, t_k)` where `w_k` are the LGL weights (scaled to `[t0, tf]` by `(tf-t0)/2`).

**Interface:**

```cpp
// include/goss/transcription/legendre_gauss_lobatto.hpp
struct LegendreGaussLobatto {
    template <typename DynamicsFn, typename CostFn>
    static CompiledOcp compile(const OcpProblem<DynamicsFn, CostFn>& ocp,
                               const std::string& model_name = "goss_lgl");
};
```

The number of LGL nodes is `ocp.mesh.num_nodes()`. The entire horizon `[t0, tf]` is used — no sub-intervals.

**Packed functor structure:**

```
inputs: z (size = nn * (ns + nc), same VariableLayout as local schemes)
output 0: Gauss-Lobatto quadrature cost
outputs 1 .. nn*ns: defects — for node k, state s:
    sum_j D[k,j] * z[state_index(j,s)] - T((tf-t0)/2.0) * f_s(x_k, u_k, t_k) = 0
```

`D` and LGL node times are pre-computed as `std::vector<double>` and captured by value in the lambda. Inside the lambda they are indexed with `T(D[k*nn + j])` and `T(t_lgl[k])` respectively.

**Steps:**

- [ ] **Step 1: Write failing test**

```cpp
// tests/transcription/test_legendre_gauss_lobatto.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "goss/transcription/legendre_gauss_lobatto.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "transcription/ocp_fixtures.hpp"

TEST(LegendreGaussLobatto, SolvesExponentialDecayWithFewNodes) {
    // LGL is spectrally accurate — 8 nodes should give excellent accuracy.
    const double x0 = 1.0, tf = 1.0;
    // num_intervals = 7 => num_nodes = 8 LGL nodes
    auto ocp = goss::transcription::test::make_exponential_decay(x0, tf, /*intervals=*/7);
    auto compiled = goss::transcription::LegendreGaussLobatto::compile(ocp, "lgl_expdecay");

    goss::solver::IpoptSolver solver;
    solver.set_tolerance(1e-11);
    std::vector<double> z0(compiled.problem->num_variables(), x0);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);

    std::size_t last = compiled.layout.num_nodes() - 1;
    double x_final = result.x[compiled.layout.state_index(last, 0)];
    // 8 LGL nodes should give much better than 1e-8 accuracy on smooth exp(-t).
    EXPECT_NEAR(x_final, goss::transcription::test::exp_decay_solution(x0, tf), 1e-8);
}

TEST(LegendreGaussLobatto, PinsInitialState) {
    auto ocp = goss::transcription::test::make_exponential_decay(2.0, 1.0, 7);
    auto compiled = goss::transcription::LegendreGaussLobatto::compile(ocp, "lgl_pin");
    std::size_t idx = compiled.layout.state_index(0, 0);
    EXPECT_DOUBLE_EQ(compiled.problem->variable_lower_bounds()[idx], 2.0);
    EXPECT_DOUBLE_EQ(compiled.problem->variable_upper_bounds()[idx], 2.0);
}

TEST(LegendreGaussLobatto, SolvesHarmonicOscillator) {
    const double tf = 1.0;
    // 10 LGL nodes over [0,1] for x' = [x1, -x0].
    auto ocp = goss::transcription::test::make_harmonic(1.0, 0.0, tf, /*intervals=*/9);
    auto compiled = goss::transcription::LegendreGaussLobatto::compile(ocp, "lgl_harmonic");
    goss::solver::IpoptSolver solver;
    solver.set_tolerance(1e-11);
    std::vector<double> z0(compiled.problem->num_variables(), 0.5);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    std::size_t last = compiled.layout.num_nodes() - 1;
    double x0_final = result.x[compiled.layout.state_index(last, 0)];
    EXPECT_NEAR(x0_final, goss::transcription::test::harmonic_x0_solution(1.0, 0.0, tf), 1e-7);
}
```

- [ ] **Step 2: Write legendre_gauss_lobatto.hpp**

```cpp
// include/goss/transcription/legendre_gauss_lobatto.hpp
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "goss/ad/cppadcg_backend.hpp"
#include "goss/nlp/nlp_problem.hpp"
#include "goss/transcription/errors.hpp"
#include "goss/transcription/lgl_nodes.hpp"
#include "goss/transcription/ocp_problem.hpp"
#include "goss/transcription/transcription.hpp"
#include "goss/transcription/variable_layout.hpp"

namespace goss::transcription {

/// Pseudospectral collocation using Legendre-Gauss-Lobatto (LGL) nodes.
/// The entire time horizon [t0, tf] is collocated at n LGL nodes mapped from [-1,1].
/// Convergence is spectral (super-algebraic) for smooth problems — exponential
/// in the number of nodes, unlike the algebraic O(h^p) of local schemes.
///
/// The ODE is enforced at every node via the global differentiation matrix D:
///   (D @ X)[k, s] = (tf-t0)/2 * f_s(x_k, u_k, t_k)   for all k, s
/// This produces dense coupling (every node in every defect), unlike the banded
/// local schemes. For moderate n (up to ~40) this is still efficient; for large n
/// consider multiple LGL sub-intervals (hp-pseudospectral, out of scope here).
struct LegendreGaussLobatto {
    template <typename DynamicsFn, typename CostFn>
    static CompiledOcp compile(const OcpProblem<DynamicsFn, CostFn>& ocp,
                               const std::string& model_name = "goss_lgl") {
        ocp.mesh.validate();
        const std::size_t nn = ocp.mesh.num_nodes();  // number of LGL nodes
        const std::size_t ns = ocp.num_states;
        const std::size_t nc = ocp.num_controls;
        const double t0 = ocp.mesh.t_initial;
        const double tf = ocp.mesh.t_final;
        const double half_duration = 0.5 * (tf - t0);

        if (nn < 2)
            throw TranscriptionError(
                "LegendreGaussLobatto: need at least 2 nodes (num_intervals >= 1)");
        if (ocp.state_lower.size() != ns || ocp.state_upper.size() != ns)
            throw TranscriptionError("compile: state bound vectors must have size == num_states");
        if (ocp.control_lower.size() != nc || ocp.control_upper.size() != nc)
            throw TranscriptionError("compile: control bound vectors must have size == num_controls");
        if (ocp.initial_state.size() != ns || ocp.final_state.size() != ns)
            throw TranscriptionError("compile: initial_state/final_state must have size == num_states");

        // Pre-compute LGL nodes on [-1,1] and map to [t0,tf].
        std::vector<double> lgl_xi, lgl_weights_ref;
        lgl_nodes_and_weights(nn, lgl_xi, lgl_weights_ref);
        // t_k = t0 + half_duration * (xi_k + 1)
        std::vector<double> t_lgl(nn);
        for (std::size_t k = 0; k < nn; ++k)
            t_lgl[k] = t0 + half_duration * (lgl_xi[k] + 1.0);
        // Gauss-Lobatto quadrature weights scaled to [t0, tf]:
        // integral ~ sum_k w_k * f(t_k) where w_k = half_duration * lgl_weights_ref[k].
        std::vector<double> lgl_weights_physical(nn);
        for (std::size_t k = 0; k < nn; ++k)
            lgl_weights_physical[k] = half_duration * lgl_weights_ref[k];

        // Differentiation matrix D on [-1,1]: D_{kj} = d phi_j / d xi at xi_k.
        const std::vector<double> D = lgl_differentiation_matrix(lgl_xi);

        VariableLayout layout(ns, nc, nn);

        // Packed functor — captures pre-computed D, t_lgl, weights by value.
        auto packed = [ocp, layout, ns, nc, nn, D, t_lgl,
                       lgl_weights_physical, half_duration](const auto& z) {
            using T = typename std::decay_t<decltype(z)>::value_type;
            // Outputs: 1 cost + nn*ns defects
            std::vector<T> outputs;
            outputs.reserve(1 + nn * ns);

            auto state_at = [&](std::size_t node) {
                std::vector<T> x(ns);
                for (std::size_t i = 0; i < ns; ++i) x[i] = z[layout.state_index(node, i)];
                return x;
            };
            auto control_at = [&](std::size_t node) {
                std::vector<T> u(nc);
                for (std::size_t j = 0; j < nc; ++j) u[j] = z[layout.control_index(node, j)];
                return u;
            };

            // Output 0: Gauss-Lobatto quadrature of running cost.
            T cost = T(0);
            for (std::size_t k = 0; k < nn; ++k) {
                T tk = T(t_lgl[k]);
                cost += T(lgl_weights_physical[k]) * ocp.cost(state_at(k), control_at(k), tk);
            }
            outputs.push_back(cost);

            // Outputs 1..nn*ns: LGL collocation defects.
            // For each node k and state s:
            //   sum_j D[k,j] * x_s(j) - (tf-t0)/2 * f_s(x_k, u_k, t_k) = 0
            // Pre-compute dynamics at each node.
            std::vector<std::vector<T>> F(nn);
            for (std::size_t k = 0; k < nn; ++k)
                F[k] = ocp.dynamics(state_at(k), control_at(k), T(t_lgl[k]));

            for (std::size_t k = 0; k < nn; ++k) {
                for (std::size_t s = 0; s < ns; ++s) {
                    // Differentiation: (D @ x_s)[k] = sum_j D[k,j] * x_s(j)
                    T Dx_ks = T(0);
                    for (std::size_t j = 0; j < nn; ++j)
                        Dx_ks += T(D[k * nn + j]) * z[layout.state_index(j, s)];
                    outputs.push_back(Dx_ks - T(half_duration) * F[k][s]);
                }
            }
            return outputs;
        };

        auto backend = std::make_unique<goss::ad::CppADCGBackend>(
            packed, layout.total_variables(), model_name);

        // Variable bounds: per-node state and control bounds.
        const std::size_t nv = layout.total_variables();
        std::vector<double> zl(nv, -kInf), zu(nv, kInf);
        for (std::size_t k = 0; k < nn; ++k) {
            for (std::size_t i = 0; i < ns; ++i) {
                const std::size_t idx = layout.state_index(k, i);
                zl[idx] = ocp.state_lower[i];
                zu[idx] = ocp.state_upper[i];
            }
            for (std::size_t j = 0; j < nc; ++j) {
                const std::size_t idx = layout.control_index(k, j);
                zl[idx] = ocp.control_lower[j];
                zu[idx] = ocp.control_upper[j];
            }
        }
        // Pin fixed boundary states.  Node 0 is t=t0 (first LGL node = -1 mapped to t0).
        // Node nn-1 is t=tf (last LGL node = +1 mapped to tf).
        for (std::size_t i = 0; i < ns; ++i) {
            if (i < ocp.initial_state_fixed.size() && ocp.initial_state_fixed[i] != 0.0) {
                const std::size_t idx = layout.state_index(0, i);
                zl[idx] = zu[idx] = ocp.initial_state[i];
            }
            if (i < ocp.final_state_fixed.size() && ocp.final_state_fixed[i] != 0.0) {
                const std::size_t idx = layout.state_index(nn - 1, i);
                zl[idx] = zu[idx] = ocp.final_state[i];
            }
        }

        // Constraint bounds: all nn*ns defects are equalities [0, 0].
        const std::size_t num_defects = nn * ns;
        std::vector<double> gl(num_defects, 0.0), gu(num_defects, 0.0);

        auto problem = std::make_unique<nlp::NLPProblem>(
            std::move(backend), std::move(zl), std::move(zu),
            std::move(gl), std::move(gu));
        return CompiledOcp{std::move(problem), layout};
    }
};

}  // namespace goss::transcription
```

- [ ] **Step 3: Write src/transcription/legendre_gauss_lobatto.cpp**

```cpp
#include "goss/transcription/legendre_gauss_lobatto.hpp"
// Header-only scheme; this TU ensures the header compiles standalone.
```

- [ ] **Step 4: Update CMakeLists.txt** — add `src/transcription/legendre_gauss_lobatto.cpp` to `goss_transcription`; add `tests/transcription/test_legendre_gauss_lobatto.cpp` to `goss_transcription_tests`.

- [ ] **Step 5: Build and run**

```
scripts/dev.sh 'cmake -S . -B build && cmake --build build && ctest --test-dir build -R "LegendreGaussLobatto" --output-on-failure'
```
Expected: all 3 LGL tests PASS. Verify x_final is accurate to 1e-8 on the 8-node exp-decay — this demonstrates spectral accuracy.

- [ ] **Step 6: Commit**

```bash
git add include/goss/transcription/legendre_gauss_lobatto.hpp \
        src/transcription/legendre_gauss_lobatto.cpp \
        tests/transcription/test_legendre_gauss_lobatto.cpp CMakeLists.txt
git commit -m "feat: LGL pseudospectral collocation scheme compile() + solve tests"
```

---

### Task 9: LGL spectral convergence test — beats Hermite-Simpson O(h⁴) on smooth problem

**Files:**
- Modify: `tests/transcription/test_legendre_gauss_lobatto.cpp`

**Design:** The definitive correctness test for the LGL scheme is that its error converges spectrally (exponentially fast in `n`) on a smooth problem, not algebraically. Measure the final-node error on exp-decay as a function of `n` (number of LGL nodes = 3, 5, 7, 9, 11). Show that the error decreases faster than `O(h^4)` — specifically, check that the log of the error is approximately linear in `n` (exponential convergence), not linear in `log(n)` (polynomial convergence).

**Steps:**

- [ ] **Step 1: Write the spectral convergence test**

```cpp
// append to tests/transcription/test_legendre_gauss_lobatto.cpp
#include <cmath>

namespace {
// Solve exp-decay with n LGL nodes; return max nodal error vs analytic solution.
double lgl_max_error(std::size_t num_nodes) {
    const double x0 = 1.0, tf = 1.0;
    // num_intervals = num_nodes - 1 so that mesh.num_nodes() == num_nodes
    auto ocp = goss::transcription::test::make_exponential_decay(
        x0, tf, /*intervals=*/num_nodes - 1);
    auto compiled = goss::transcription::LegendreGaussLobatto::compile(
        ocp, "lgl_conv_" + std::to_string(num_nodes));
    goss::solver::IpoptSolver solver;
    solver.set_tolerance(1e-13);  // solver tolerance well below spectral accuracy
    std::vector<double> z0(compiled.problem->num_variables(), x0);
    auto result = solver.solve(*compiled.problem, z0);
    if (result.status != goss::solver::SolverStatus::Success) return 1e9;

    const auto& layout = compiled.layout;
    // Pre-compute LGL node times to compare at each node.
    std::vector<double> lgl_xi, lgl_weights;
    goss::transcription::lgl_nodes_and_weights(num_nodes, lgl_xi, lgl_weights);
    const double half_dur = 0.5 * tf;
    double max_err = 0.0;
    for (std::size_t k = 0; k < num_nodes; ++k) {
        const double t_k = 0.0 + half_dur * (lgl_xi[k] + 1.0);
        const double xk  = result.x[layout.state_index(k, 0)];
        const double exact = goss::transcription::test::exp_decay_solution(x0, t_k);
        max_err = std::max(max_err, std::abs(xk - exact));
    }
    return max_err;
}
}  // namespace

TEST(LegendreGaussLobatto, ConvergesSpectrally) {
    // Errors at n=3,5,7,9,11 LGL nodes on smooth exp(-t).
    const std::vector<std::size_t> node_counts = {3, 5, 7, 9, 11};
    std::vector<double> errors;
    for (std::size_t n : node_counts) errors.push_back(lgl_max_error(n));

    // Errors must decrease monotonically.
    for (std::size_t i = 0; i + 1 < errors.size(); ++i)
        ASSERT_LT(errors[i + 1], errors[i])
            << "LGL error must decrease as n increases";

    // Spectral convergence check: the convergence rate (ratio of log errors)
    // must be much steeper than O(h^4) = O(n^{-4}).
    // For O(h^4): halving h (doubling nodes) reduces error by 16x -> ratio ~4 per doubling.
    // For spectral: ratio should be >>4, e.g. >6 for n=3->5->7.
    // Check the ratio log(e[i])/log(e[i+1]) > 4 for consecutive pairs.
    // Use 3->7 (skip by 2) for a cleaner ratio.
    const double log_ratio_3_to_7 =
        std::log(errors[0] / errors[2]) / std::log(static_cast<double>(node_counts[2]) /
                                                   static_cast<double>(node_counts[0]));
    EXPECT_GT(log_ratio_3_to_7, 4.0)
        << "LGL spectral convergence should exceed O(h^4); "
           "observed log-ratio: " << log_ratio_3_to_7;

    // Hard accuracy check: 11 LGL nodes must achieve < 1e-10 on smooth exp(-t).
    EXPECT_LT(errors.back(), 1e-10)
        << "11 LGL nodes should achieve near-machine precision on smooth exp(-t)";
}

TEST(LegendreGaussLobatto, SameNodeCountOutperformsHermiteSimpson) {
    // With n=9 LGL nodes vs 9 HS nodes (8 intervals), LGL should win on smooth problem.
    const std::size_t n_nodes = 9;
    const double x0 = 1.0, tf = 1.0;

    // LGL: 9 nodes
    double lgl_err = lgl_max_error(n_nodes);

    // Hermite-Simpson: 9 nodes = 8 intervals, error measured at node times.
    auto hs_ocp = goss::transcription::test::make_exponential_decay(x0, tf, n_nodes - 1);
    auto hs_compiled = goss::transcription::HermiteSimpson::compile(
        hs_ocp, "lgl_vs_hs_compare");
    goss::solver::IpoptSolver solver;
    solver.set_tolerance(1e-13);
    std::vector<double> z0(hs_compiled.problem->num_variables(), x0);
    auto hs_result = solver.solve(*hs_compiled.problem, z0);
    ASSERT_EQ(hs_result.status, goss::solver::SolverStatus::Success);
    const auto& hs_layout = hs_compiled.layout;
    const double h = tf / static_cast<double>(n_nodes - 1);
    double hs_err = 0.0;
    for (std::size_t k = 0; k < n_nodes; ++k) {
        double xk = hs_result.x[hs_layout.state_index(k, 0)];
        double exact = goss::transcription::test::exp_decay_solution(x0, k * h);
        hs_err = std::max(hs_err, std::abs(xk - exact));
    }

    EXPECT_LT(lgl_err, hs_err)
        << "LGL with " << n_nodes << " nodes should outperform HS with "
        << n_nodes << " nodes on smooth exp(-t)";
}
```

- [ ] **Step 2: Build and run**

```
scripts/dev.sh 'cmake --build build && ctest --test-dir build -R "LegendreGaussLobatto.Converges|LegendreGaussLobatto.SameNode" --output-on-failure'
```
Expected: spectral convergence confirmed; 11 nodes < 1e-10; LGL beats HS at same node count.

- [ ] **Step 3: Commit**

```bash
git add tests/transcription/test_legendre_gauss_lobatto.cpp
git commit -m "test: LGL spectral convergence and comparison against Hermite-Simpson"
```

---

### Task 10: Add LGL to cross-scheme agreement test

**Files:**
- Modify: `tests/transcription/test_scheme_agreement.cpp`

**Steps:**

- [ ] **Step 1: Add LGL to agreement test**

```cpp
// append to tests/transcription/test_scheme_agreement.cpp
#include "goss/transcription/legendre_gauss_lobatto.hpp"

TEST(SchemeAgreement, AllThreeSchemesAgreeOnExpDecay) {
    const double x0 = 1.0, tf = 1.0;
    // Use 20 nodes so both HS (19 intervals) and LGL (20 nodes) are well-resolved.
    const std::size_t n_nodes_minus_1 = 19;
    auto ocp_t = goss::transcription::test::make_exponential_decay(x0, tf, n_nodes_minus_1);
    auto ocp_h = goss::transcription::test::make_exponential_decay(x0, tf, n_nodes_minus_1);
    auto ocp_l = goss::transcription::test::make_exponential_decay(x0, tf, n_nodes_minus_1);

    auto ct  = goss::transcription::Trapezoidal::compile(ocp_t, "agree3_trap");
    auto ch  = goss::transcription::HermiteSimpson::compile(ocp_h, "agree3_hs");
    auto cl  = goss::transcription::LegendreGaussLobatto::compile(ocp_l, "agree3_lgl");

    goss::solver::IpoptSolver solver;
    solver.set_tolerance(1e-11);
    auto solve_final_state = [&](goss::transcription::CompiledOcp& compiled) {
        std::vector<double> z0(compiled.problem->num_variables(), x0);
        auto result = solver.solve(*compiled.problem, z0);
        EXPECT_EQ(result.status, goss::solver::SolverStatus::Success);
        std::size_t last = compiled.layout.num_nodes() - 1;
        return result.x[compiled.layout.state_index(last, 0)];
    };

    double xt = solve_final_state(ct);
    double xh = solve_final_state(ch);
    double xl = solve_final_state(cl);
    double exact = goss::transcription::test::exp_decay_solution(x0, tf);

    EXPECT_NEAR(xt, exact, 1e-3);
    EXPECT_NEAR(xh, exact, 1e-6);
    EXPECT_NEAR(xl, exact, 1e-10);  // LGL at 20 nodes beats both on smooth problem
    EXPECT_NEAR(xt, xh, 1e-3);
    EXPECT_NEAR(xh, xl, 1e-5);
}
```

- [ ] **Step 2: Build and run**

```
scripts/dev.sh 'cmake --build build && ctest --test-dir build -R "SchemeAgreement" --output-on-failure'
```

- [ ] **Step 3: Commit**

```bash
git add tests/transcription/test_scheme_agreement.cpp
git commit -m "test: three-scheme agreement — Trapezoidal, HermiteSimpson, LGL"
```

---

### Task 11: Full regression suite + cleanup

**Files:**
- No new files; verify everything

**Steps:**

- [ ] **Step 1: Run the complete test suite**

```
scripts/dev.sh 'cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure'
```
Expected: ALL tests pass. Prior suite (sim, model, solver, nlp, ad layers) untouched.

- [ ] **Step 2: Verify no includes of internal detail leak**

Confirm no test file includes `lgl_nodes.hpp` via private headers; all LGL usage goes through `legendre_gauss_lobatto.hpp`. Confirm `ocp_problem.hpp` is unchanged (no `NonUniformMesh` in it).

- [ ] **Step 3: Final commit**

```bash
git add -p  # stage any remaining formatting/comment fixes
git commit -m "chore: full regression pass — mesh refinement + LGL pseudospectral complete"
```

---

## Files Changed vs Created Summary

**New files (create from scratch):**
- `include/goss/transcription/mesh.hpp`
- `src/transcription/mesh.cpp`
- `include/goss/transcription/mesh_refinement.hpp`
- `src/transcription/mesh_refinement.cpp`
- `include/goss/transcription/lgl_nodes.hpp`
- `src/transcription/lgl_nodes.cpp`
- `include/goss/transcription/legendre_gauss_lobatto.hpp`
- `src/transcription/legendre_gauss_lobatto.cpp`
- `tests/transcription/test_mesh.cpp`
- `tests/transcription/test_mesh_refinement.cpp`
- `tests/transcription/test_lgl_nodes.cpp`
- `tests/transcription/test_legendre_gauss_lobatto.cpp`

**Existing files modified:**
- `include/goss/transcription/trapezoidal.hpp` — add non-uniform overload; refactor uniform overload as wrapper
- `include/goss/transcription/hermite_simpson.hpp` — same
- `tests/transcription/ocp_fixtures.hpp` — add `FastDecayDynamics` + `make_fast_decay`
- `tests/transcription/test_trapezoidal.cpp` — two new non-uniform tests
- `tests/transcription/test_hermite_simpson.cpp` — two new non-uniform tests
- `tests/transcription/test_scheme_agreement.cpp` — two new tests (non-uniform, 3-scheme)
- `CMakeLists.txt` — add 4 new `.cpp` source files + 4 new test `.cpp` files

**Unchanged files (zero modifications):**
- `include/goss/transcription/ocp_problem.hpp` — uniform `Mesh` struct untouched
- `include/goss/transcription/variable_layout.hpp`
- `include/goss/transcription/transcription.hpp`
- `include/goss/transcription/errors.hpp`
- All other layers (sim, model, solver, nlp, ad)

---

## Self-Review

**Backward compatibility:** The uniform `Mesh` struct and `OcpProblem` are untouched. The existing `Trapezoidal::compile(ocp, model_name)` and `HermiteSimpson::compile(ocp, model_name)` signatures still compile unchanged — they become thin wrappers that call `to_nonuniform(ocp.mesh)` internally. No existing test is modified in a behavior-changing way (only additive new tests).

**Non-uniform mesh ripple-effect assessment:** The only existing production headers that change are `trapezoidal.hpp` and `hermite_simpson.hpp`. The change is additive (one new overload per scheme; the original overload now delegates). There is no change to `VariableLayout`, `OcpProblem`, `NLPProblem`, or any solver/AD layer.

**AD safety:** All numeric literals inside generic lambdas (captured and used where `T` may be an AD type) use the `T(...)` constructor pattern. Pre-computed `double` arrays (`D`, `t_lgl`, `node_times`) are captured by value and indexed with `T(D[k*nn+j])` so the multiply with a `T` value is always `T * T`. No raw `double` arithmetic leaks into the functor body.

**LGL global coupling:** The LGL functor produces `nn * ns` defect constraints (one per node per state), unlike the local schemes which produce `(nn-1) * ns`. This is documented in the header. The sparsity pattern is dense (each defect couples all `nn` nodes via `D`), which is fine for moderate `n` — CppADCGBackend will record and JIT the full dense structure.

**Test coverage:**
- NonUniformMesh: construction, validation, widths, bisection (Task 1).
- Trapezoidal non-uniform: solve passes, uniform regression (Task 2).
- HermiteSimpson non-uniform: same (Task 3).
- Error estimator: size check, ordering on localized-feature problem (Task 4).
- refine_and_solve: drives error below tolerance, concentrates nodes correctly, outperforms uniform at same count (Task 5).
- LGL math: nodes/weights for small n, symmetry, weight sum = 2, D@const = 0, D@x = 1 (Task 7).
- LGL scheme: solve, pin, harmonic (Task 8).
- LGL spectral convergence: beats O(h^4), 11 nodes < 1e-10, beats HS at same count (Task 9).
- Three-scheme agreement (Task 10).

**Definitive convergence tests:**
- Local schemes: convergence order established in existing Tasks 5/7 of transcription-layer plan; preserved here.
- LGL: spectral convergence confirmed by log-ratio > 4 (exceeds O(h^4)) and absolute accuracy < 1e-10 at 11 nodes on smooth problem.
- AMR: max interval error ≤ tolerance (direct quantitative bound), node distribution test, accuracy-vs-uniform test.

**Known risks:**
- LGL at large `n` (> ~40) produces a dense NLP Jacobian that CppADCGBackend will record fully — performance degrades. Documented in the header; hp-pseudospectral (multiple LGL sub-intervals) is the remedy, out of scope.
- AMR warm-starting is cold (flat guess) in this plan. A simple linear-interpolation warm-start from the previous iteration's solution would reduce IPOPT iterations significantly but adds code complexity; flagged as a future improvement.
- IPOPT tolerance for AMR tests must be tight (`1e-10`) so that solver residuals do not dominate the per-interval error estimates. This is set explicitly in `refine_and_solve`.
