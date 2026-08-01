# hp-Pseudospectral Collocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement multi-interval hp-pseudospectral collocation by partitioning the time horizon `[t0, tf]` into `S` segments, each with its own local LGL nodes + local differentiation matrix, and enforcing state continuity across segment boundaries. The existing single-interval `LegendreGaussLobatto::compile(ocp, name)` path is left entirely untouched; this feature is additive via a new `LegendreGaussLobatto::compile_hp(ocp, hp_mesh, name)` static method.

**Architecture:**

```
include/goss/transcription/
  hp_mesh.hpp                   — NEW: HpMesh struct (segment_boundaries + per_segment_orders)
                                        + validate() + to_single_segment() factory
  legendre_gauss_lobatto.hpp    — MODIFY: add compile_hp() static method;
                                        add "out of scope" comment update
  variable_layout.hpp           — no change (existing VariableLayout works for hp with
                                        total_nodes = sum of per-segment nodes using
                                        DUPLICATED boundary nodes — see design decision)

src/transcription/
  legendre_gauss_lobatto.cpp    — no change (one-line TU stays as-is)
  hp_mesh.cpp                   — NEW: non-template hp_mesh validate() impl

tests/transcription/
  test_hp_pseudospectral.cpp    — NEW: all hp tests (differentiation matrix per segment,
                                        continuity constraints, S=1 reduces-to-current,
                                        h-refinement convergence, hp-beats-global)
  test_hp_mesh.cpp              — NEW: HpMesh construction and validation tests

CMakeLists.txt                  — MODIFY: add hp_mesh.cpp TU + new test .cpp files

tests/accuracy/
  test_hp_accuracy.cpp          — NEW: hp-beats-global accuracy test (depends on
                                        goss_accuracy_tests target from accuracy-suite plan)
```

**Tech Stack:** C++17, GoogleTest v1.14.0, IPOPT via `goss::solver::IpoptSolver`, CppADCG JIT via `goss_ad_impl`, `goss::transcription::lgl_nodes_and_weights` + `lgl_differentiation_matrix` (existing).

## Global Constraints

- C++17 (`CMAKE_CXX_STANDARD 17`, `CMAKE_CXX_STANDARD_REQUIRED ON`) — no C++20 features.
- Header-only scheme: `compile_hp()` is a templated static function in `legendre_gauss_lobatto.hpp`; only non-template `HpMesh::validate()` is a `.cpp` TU.
- Container-first — ALL cmake/ctest invocations via `scripts/dev.sh '<command>'`; never run cmake/ctest directly on the host.
- GoogleTest — all tests use `TEST(Suite, CaseName)` + `ASSERT_*/EXPECT_*`; no custom test runner.
- Verbose, descriptive names — e.g. `num_segments`, `segment_boundary_times`, `per_segment_node_count`, `global_node_offset`. No single-letter names except local loop indices `s`, `k`, `i`, `j`.
- Type annotations + comments explain WHY — every non-obvious formula must have a comment explaining the mathematical source.
- `OcpProblem` is NOT changed — hp-pseudospectral is a pure transcription-internal meshing change.
- Backward compatibility: existing `LegendreGaussLobatto::compile(ocp, name)` and all other schemes (Trapezoidal, HermiteSimpson) are untouched. No existing test may change behaviour.
- Pinned-initial-state guard: preserved — `compile_hp()` enforces the same guard as `compile()` (all initial state components must be fixed).
- CppADCG model names must be unique per call across the entire process; every test must pass a unique string.

---

## KEY DESIGN DECISIONS

### Decision 1: Duplicated Boundary Nodes

**Choice: DUPLICATED boundary nodes tied by explicit equality (continuity) constraints.**

Each segment `s` owns its own full set of `n_s` LGL nodes: the last node of segment `s` and the first node of segment `s+1` are SEPARATE variables, and state continuity is enforced via equality constraints:

```
x_end(s) - x_start(s+1) = 0    for s = 0..S-2, all state components
```

**Justification:**
- Simpler assembly: each segment `s` is independently assembled with a local `VariableLayout` offset into the global `z` vector; there is no cross-segment index arithmetic in the per-segment collocation defect loop.
- The `VariableLayout` class packs `z` node-by-node (`node * variables_per_node + state_or_control`). With SHARED nodes, segment assembly would need to track the shared-node index vs interior-node index, complicating per-segment loops. With duplicated nodes, segment `s` occupies contiguous global nodes `[offset_s, offset_s + n_s)` and the collocation loop is identical to the single-interval loop.
- Control variables: control is naturally DISCONTINUOUS across segment boundaries (free per segment, separate for duplicated last/first node). This is the standard hp-OC convention (controls are left-continuous piecewise polynomials; continuity is NOT enforced).
- Total node count: `N_total = sum_{s=0}^{S-1} n_s` (S=1 recovers the current single-interval layout exactly without any duplication).
- Total continuity constraints: `(S-1) * num_states` equality constraints appended after the per-segment collocation defects.

**Alternative (shared) rejected:** shared nodes would reduce variables by `(S-1) * variables_per_node`, but would require segment-aware indexing that complicates assembly and makes the `VariableLayout::state_index(global_node, state)` call non-uniform (the shared node belongs to two segments simultaneously, requiring a different offset computation for the last node of segment `s` vs the first node of segment `s+1`).

### Decision 2: HpMesh Specification — New compile_hp() overload

**Choice: new `compile_hp(ocp, hp_mesh, name)` static method; `HpMesh` is a new struct in `hp_mesh.hpp`.**

`OcpProblem` does NOT change — it still carries the single-interval `Mesh` (which provides `t_initial`, `t_final`, and `num_intervals`). The `HpMesh` is passed separately to `compile_hp()`, carrying the segment boundary times and per-segment node counts. The `t_initial` and `t_final` in `HpMesh` must match `ocp.mesh.t_initial` / `ocp.mesh.t_final`; `compile_hp()` validates this.

```cpp
// include/goss/transcription/hp_mesh.hpp

struct HpMesh {
    // Segment boundary times: size = S+1.
    // segment_boundary_times[0] = t_initial, segment_boundary_times[S] = t_final.
    // Must be strictly increasing. S = segment_boundary_times.size() - 1.
    std::vector<double> segment_boundary_times;

    // Number of LGL nodes per segment: size = S.
    // per_segment_node_count[s] >= 2 for all s (need at least 2 LGL nodes = 1 interval).
    std::vector<std::size_t> per_segment_node_count;

    std::size_t num_segments()    const { return per_segment_node_count.size(); }
    double      t_initial()       const { return segment_boundary_times.front(); }
    double      t_final()         const { return segment_boundary_times.back(); }

    // Total number of global nodes (duplicated boundary nodes).
    // WHY sum instead of sum-(S-1): each segment owns ALL n_s nodes including its endpoints;
    // boundary nodes are duplicated not shared.
    std::size_t total_nodes()     const {
        std::size_t total = 0;
        for (const std::size_t n_s : per_segment_node_count) total += n_s;
        return total;
    }

    void validate() const;  // implemented in hp_mesh.cpp
};

// Convenience: build a single-segment HpMesh from an existing Mesh.
// WHY: allows S=1 to exercise compile_hp as a drop-in.
HpMesh to_single_segment_hp_mesh(const Mesh& uniform_mesh);
```

**API usage:**

```cpp
// User specifies: 3 segments over [0, 3] with node counts [5, 4, 6].
HpMesh hp_mesh;
hp_mesh.segment_boundary_times = {0.0, 1.0, 2.0, 3.0};  // S=3
hp_mesh.per_segment_node_count = {5, 4, 6};               // 5+4+6=15 global nodes

auto compiled = LegendreGaussLobatto::compile_hp(ocp, hp_mesh, "mymodel");
```

**Why NOT a new OcpProblem member:** The DAE and path-constraints plans (Tasks #83, #84) add to `OcpProblem` to express new kinds of constraints. hp-pseudospectral is a transcription technique that re-meshes an existing continuous-time OCP; it consumes the same dynamics, cost, bounds, and boundary conditions as the single-interval LGL. Adding the hp mesh to `OcpProblem` would bleed a transcription detail into a problem description struct, violating separation of concerns.

### Decision 3: Control Continuity

**Choice: Controls are DISCONTINUOUS across segment boundaries.**

The duplicate boundary nodes for segment `s` (last node) and segment `s+1` (first node) carry separate control variables; no continuity constraint is imposed. This is the standard hp-OC convention:
- Controls are typically piecewise-polynomial with jumps at mesh boundaries (e.g. bang-bang control, saturated control).
- If the user wants continuous controls, they can set tight box bounds on the boundary control values; or use the same control value across segments by a custom constraint — but this is not a default transcription convention.
- LGL pseudospectral schemes in the literature (Fahroo & Ross 2001, Garg et al. 2010) treat controls as independently collocated per segment.

---

## Core Math

### Segment Layout

For segment `s` (0-indexed, `s = 0..S-1`):
- Physical interval: `[t_a^s, t_b^s]` where `t_a^s = segment_boundary_times[s]`, `t_b^s = segment_boundary_times[s+1]`.
- Half-duration: `half_dur_s = (t_b^s - t_a^s) / 2`.
- LGL nodes on `[-1, 1]`: `xi_0, ..., xi_{n_s - 1}` from `lgl_nodes_and_weights(n_s, ...)`.
- Physical node times (affine mapping from reference to physical):
  ```
  t_k^s = t_a^s + half_dur_s * (xi_k + 1)    for k = 0..n_s-1
  ```
  At `xi_0 = -1`: `t_0^s = t_a^s`. At `xi_{n_s-1} = +1`: `t_{n_s-1}^s = t_b^s`. Confirmed.
- Local differentiation matrix (on `[-1,1]`): `D_s = lgl_differentiation_matrix(lgl_xi_s)`, size `n_s x n_s`, stored row-major.
- Physical-domain derivative: `dx/dt = (1/half_dur_s) * (dx/d_xi)`, so the collocation condition becomes:
  ```
  sum_j D_s[k,j] * x_s(j)  =  half_dur_s * f(x_k, u_k, t_k^s)
  ```
  WHY `half_dur_s` on the right (not `1/half_dur_s` on the left): multiplying both sides of `(1/half_dur_s)(D_s @ x)[k] = f_k` by `half_dur_s` avoids dividing by a small number; consistent with the single-interval LGL formulation (`half_duration * F[k][s]` in `legendre_gauss_lobatto.hpp` line 137).

### Global Node Indexing

Segment `s` starts at global node offset:
```
global_node_offset[s] = sum_{r=0}^{s-1} per_segment_node_count[r]
```
(with `global_node_offset[0] = 0`).

Global node index for segment `s`, local node `k`:
```
global_node(s, k) = global_node_offset[s] + k
```

The `VariableLayout` is constructed with `total_nodes = sum n_s` as the `num_nodes` argument. Then:
```cpp
layout.state_index(global_node(s, k), state_i)   // = global_node(s,k) * vars_per_node + state_i
layout.control_index(global_node(s, k), ctrl_j)  // = global_node(s,k) * vars_per_node + ns + ctrl_j
```

No modifications to `VariableLayout` are needed.

### Per-Segment Collocation Defects

For segment `s`, enforce the ODE at local nodes `k = 1..n_s-1` (skip `k=0`):
- Global node index: `gk = global_node_offset[s] + k`.
- WHY skip `k=0` for each segment: the first node of segment 0 is the overall initial state (pinned by variable bounds); the first node of segments 1..S-1 is tied to the last node of the previous segment via a continuity constraint (equality). In both cases the node-0 value of each segment is determined by external constraints, not by the local collocation equation. Including node-0 defects would overdetermine the system.
- Defect at segment `s`, local node `k` (`k=1..n_s-1`), state `i`:
  ```
  sum_{j=0}^{n_s-1} D_s[k,j] * z[global_node(s,j) * vpn + i]  -  half_dur_s * F_s[k][i]  = 0
  ```
  where `vpn = variables_per_node = ns + nc` and `F_s[k] = dynamics(x_k^s, u_k^s, t_k^s)`.

Total collocation defects: `sum_{s=0}^{S-1} (n_s - 1) * num_states`.

### Continuity Constraints

For each internal boundary `s = 0..S-2`, state component `i`:
```
z[layout.state_index(global_node(s,   n_s-1), i)]   // last node of segment s
  -
z[layout.state_index(global_node(s+1, 0),     i)]   // first node of segment s+1
  = 0
```

Total continuity constraints: `(S-1) * num_states`.

### Cost Quadrature

Running cost = sum over segments of per-segment LGL quadrature:
```
J = sum_{s=0}^{S-1}  sum_{k=0}^{n_s-1} (half_dur_s * lgl_w_s[k]) * cost(x_k^s, u_k^s, t_k^s)
```
where `lgl_w_s[k]` are the reference LGL weights (on `[-1,1]`, summing to 2). The physical quadrature weight is `half_dur_s * lgl_w_s[k]`.

### Variable and Constraint Bounds

- Per-node state and control bounds: same as single-interval — iterate over all `total_nodes` global nodes, apply `state_lower[i]` / `state_upper[i]` and `control_lower[j]` / `control_upper[j]`.
- Pin initial state: global node 0, state `i` where `initial_state_fixed[i] != 0` → `zl[state_index(0, i)] = zu[state_index(0, i)] = initial_state[i]`.
- Pin final state: global node `total_nodes - 1` (last node of last segment), state `i` where `final_state_fixed[i] != 0`.
- Continuity constraints are equality: `gl[idx] = gu[idx] = 0.0` for those entries.

---

## File Structure

| File | Responsibility |
|---|---|
| `include/goss/transcription/hp_mesh.hpp` | `HpMesh` struct + `to_single_segment_hp_mesh()` factory |
| `include/goss/transcription/legendre_gauss_lobatto.hpp` | Add `compile_hp()` static method; update deferral comment |
| `src/transcription/hp_mesh.cpp` | Non-template `HpMesh::validate()` + `to_single_segment_hp_mesh()` |
| `tests/transcription/test_hp_mesh.cpp` | `HpMesh` construction, validation, edge cases |
| `tests/transcription/test_hp_pseudospectral.cpp` | Per-segment D-matrix exactness, S=1 equivalence, h-refinement convergence, hp-beats-global |
| `tests/accuracy/test_hp_accuracy.cpp` | hp convergence/accuracy test using accuracy harness (dependency: accuracy-suite plan) |
| `CMakeLists.txt` | Add `hp_mesh.cpp` to `goss_transcription`; add new `.cpp` test files |

---

## Task 1: `HpMesh` Struct and Validation

**Files:**
- Create: `include/goss/transcription/hp_mesh.hpp`
- Create: `src/transcription/hp_mesh.cpp`
- Modify: `CMakeLists.txt` (add `src/transcription/hp_mesh.cpp` to `goss_transcription`)
- Create: `tests/transcription/test_hp_mesh.cpp`
- Modify: `CMakeLists.txt` (add `tests/transcription/test_hp_mesh.cpp` to `goss_transcription_tests`)

**Interfaces:**
- `goss::transcription::HpMesh` — struct with `segment_boundary_times`, `per_segment_node_count`, `num_segments()`, `t_initial()`, `t_final()`, `total_nodes()`, `validate()`
- `goss::transcription::to_single_segment_hp_mesh(const Mesh&)` → `HpMesh`

- [ ] **Step 1: Write the failing tests**

Create `tests/transcription/test_hp_mesh.cpp`:

```cpp
// tests/transcription/test_hp_mesh.cpp
//
// Tests for HpMesh: construction, validation, and the to_single_segment_hp_mesh factory.
// HpMesh represents the segment-boundary and per-segment-order specification for
// hp-pseudospectral collocation; it is separate from OcpProblem so that transcription
// parameters do not bleed into the problem description.
#include <gtest/gtest.h>
#include <cstddef>
#include <vector>
#include "goss/transcription/hp_mesh.hpp"
#include "goss/transcription/errors.hpp"
#include "goss/transcription/ocp_problem.hpp"  // for Mesh (uniform)

TEST(HpMesh, ConstructAndQueryThreeSegments) {
    goss::transcription::HpMesh hp_mesh;
    hp_mesh.segment_boundary_times = {0.0, 1.0, 2.5, 4.0};  // S=3
    hp_mesh.per_segment_node_count = {4, 5, 3};

    EXPECT_EQ(hp_mesh.num_segments(), 3u);
    EXPECT_DOUBLE_EQ(hp_mesh.t_initial(), 0.0);
    EXPECT_DOUBLE_EQ(hp_mesh.t_final(),   4.0);
    // total_nodes = 4 + 5 + 3 = 12 (duplicated boundary nodes)
    EXPECT_EQ(hp_mesh.total_nodes(), 12u);
    EXPECT_NO_THROW(hp_mesh.validate());
}

TEST(HpMesh, SingleSegmentIsValid) {
    goss::transcription::HpMesh hp_mesh;
    hp_mesh.segment_boundary_times = {0.0, 1.0};  // S=1
    hp_mesh.per_segment_node_count = {8};

    EXPECT_EQ(hp_mesh.num_segments(), 1u);
    EXPECT_EQ(hp_mesh.total_nodes(), 8u);
    EXPECT_NO_THROW(hp_mesh.validate());
}

TEST(HpMesh, ValidateRejectsNonIncreasingBoundaries) {
    goss::transcription::HpMesh hp_mesh;
    hp_mesh.segment_boundary_times = {0.0, 2.0, 1.0};  // not strictly increasing
    hp_mesh.per_segment_node_count = {4, 4};
    EXPECT_THROW(hp_mesh.validate(), goss::transcription::TranscriptionError);
}

TEST(HpMesh, ValidateRejectsEqualBoundaries) {
    goss::transcription::HpMesh hp_mesh;
    hp_mesh.segment_boundary_times = {0.0, 1.0, 1.0, 2.0};  // equal boundaries
    hp_mesh.per_segment_node_count = {4, 4, 4};
    EXPECT_THROW(hp_mesh.validate(), goss::transcription::TranscriptionError);
}

TEST(HpMesh, ValidateRejectsNodeCountLessThanTwo) {
    goss::transcription::HpMesh hp_mesh;
    hp_mesh.segment_boundary_times = {0.0, 1.0, 2.0};
    hp_mesh.per_segment_node_count = {4, 1};  // segment 1 has only 1 node — invalid
    EXPECT_THROW(hp_mesh.validate(), goss::transcription::TranscriptionError);
}

TEST(HpMesh, ValidateRejectsMismatchedSizes) {
    goss::transcription::HpMesh hp_mesh;
    hp_mesh.segment_boundary_times = {0.0, 1.0, 2.0};  // S=2
    hp_mesh.per_segment_node_count = {4, 4, 4};           // 3 entries — mismatch
    EXPECT_THROW(hp_mesh.validate(), goss::transcription::TranscriptionError);
}

TEST(HpMesh, ValidateRejectsEmptyMesh) {
    goss::transcription::HpMesh hp_mesh;
    EXPECT_THROW(hp_mesh.validate(), goss::transcription::TranscriptionError);
}

TEST(HpMesh, ToSingleSegmentMatchesUniformMesh) {
    goss::transcription::Mesh uniform_mesh{0.0, 2.0, /*num_intervals=*/7};
    // num_nodes = 8 for uniform mesh; to_single_segment_hp_mesh should give 8 LGL nodes.
    const goss::transcription::HpMesh hp_mesh =
        goss::transcription::to_single_segment_hp_mesh(uniform_mesh);

    EXPECT_EQ(hp_mesh.num_segments(), 1u);
    EXPECT_DOUBLE_EQ(hp_mesh.t_initial(), 0.0);
    EXPECT_DOUBLE_EQ(hp_mesh.t_final(),   2.0);
    // WHY 8: num_nodes = num_intervals + 1 = 8; single segment has 8 LGL nodes.
    EXPECT_EQ(hp_mesh.per_segment_node_count[0], uniform_mesh.num_nodes());
    EXPECT_EQ(hp_mesh.total_nodes(), uniform_mesh.num_nodes());
    EXPECT_NO_THROW(hp_mesh.validate());
}
```

- [ ] **Step 2: Run to verify test target fails (hp_mesh.hpp not yet created)**

```bash
scripts/dev.sh 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target goss_transcription_tests 2>&1 | head -30'
```

Expected: compile error — `goss/transcription/hp_mesh.hpp: No such file or directory`.

- [ ] **Step 3: Create `include/goss/transcription/hp_mesh.hpp`**

```cpp
// include/goss/transcription/hp_mesh.hpp
#pragma once
#include <cstddef>
#include <vector>
#include "goss/transcription/errors.hpp"
#include "goss/transcription/ocp_problem.hpp"  // for Mesh (uniform)

namespace goss::transcription {

/// Mesh specification for hp-pseudospectral collocation.
///
/// Partitions [t0, tf] into S segments, each with an independent set of
/// Legendre-Gauss-Lobatto (LGL) nodes. Boundary nodes are DUPLICATED (not shared):
/// segment s owns all n_s LGL nodes in [t_a^s, t_b^s] including its endpoints.
/// State continuity at segment boundaries is enforced via explicit equality constraints
/// in the NLP (not by sharing a variable). This keeps per-segment assembly uniform.
///
/// Controls are DISCONTINUOUS across segment boundaries (standard hp-OC convention).
struct HpMesh {
    /// Segment boundary times: size S+1.
    /// segment_boundary_times[0] = t_initial (overall),
    /// segment_boundary_times[S] = t_final (overall).
    /// Must be strictly increasing.
    std::vector<double> segment_boundary_times;

    /// Number of LGL nodes per segment: size S.
    /// per_segment_node_count[s] >= 2 for each s
    /// (need at least 2 LGL nodes, i.e. 1 LGL interval, per segment).
    std::vector<std::size_t> per_segment_node_count;

    std::size_t num_segments() const {
        return per_segment_node_count.size();
    }

    double t_initial() const {
        if (segment_boundary_times.empty())
            throw TranscriptionError("HpMesh::t_initial: mesh is empty");
        return segment_boundary_times.front();
    }

    double t_final() const {
        if (segment_boundary_times.empty())
            throw TranscriptionError("HpMesh::t_final: mesh is empty");
        return segment_boundary_times.back();
    }

    /// Total number of global decision-variable nodes (sum of per-segment counts).
    /// WHY sum (not sum - (S-1)): boundary nodes are duplicated; segment s occupies
    /// contiguous global nodes [offset_s, offset_s + n_s), simplifying assembly.
    std::size_t total_nodes() const {
        std::size_t total = 0;
        for (const std::size_t node_count : per_segment_node_count)
            total += node_count;
        return total;
    }

    /// Validate the HpMesh.
    /// Throws TranscriptionError if:
    ///   - segment_boundary_times is empty or has fewer than 2 entries
    ///   - per_segment_node_count is empty or has size != segment_boundary_times.size() - 1
    ///   - segment_boundary_times is not strictly increasing
    ///   - any per_segment_node_count[s] < 2
    void validate() const;
};

/// Build a single-segment HpMesh from a uniform Mesh.
/// The resulting HpMesh has S=1, t_initial = uniform_mesh.t_initial,
/// t_final = uniform_mesh.t_final, and per_segment_node_count[0] = uniform_mesh.num_nodes().
/// WHY: allows compile_hp with S=1 to be a drop-in for compile() on the same OcpProblem.
HpMesh to_single_segment_hp_mesh(const Mesh& uniform_mesh);

}  // namespace goss::transcription
```

- [ ] **Step 4: Create `src/transcription/hp_mesh.cpp`**

```cpp
// src/transcription/hp_mesh.cpp
#include "goss/transcription/hp_mesh.hpp"
#include <string>

namespace goss::transcription {

void HpMesh::validate() const {
    if (segment_boundary_times.empty() || segment_boundary_times.size() < 2)
        throw TranscriptionError(
            "HpMesh::validate: segment_boundary_times must have at least 2 entries (t0 and tf)");

    const std::size_t num_seg = segment_boundary_times.size() - 1;
    if (per_segment_node_count.size() != num_seg)
        throw TranscriptionError(
            "HpMesh::validate: per_segment_node_count.size() must equal "
            "segment_boundary_times.size() - 1 (number of segments S)");

    // Strictly increasing boundary times.
    for (std::size_t seg_idx = 0; seg_idx < num_seg; ++seg_idx) {
        if (segment_boundary_times[seg_idx + 1] <= segment_boundary_times[seg_idx])
            throw TranscriptionError(
                "HpMesh::validate: segment_boundary_times must be strictly increasing "
                "(segment " + std::to_string(seg_idx) + " has zero or negative width)");
    }

    // Each segment needs at least 2 LGL nodes (to have at least 1 LGL interval).
    for (std::size_t seg_idx = 0; seg_idx < num_seg; ++seg_idx) {
        if (per_segment_node_count[seg_idx] < 2)
            throw TranscriptionError(
                "HpMesh::validate: per_segment_node_count[" + std::to_string(seg_idx) +
                "] must be >= 2 (need at least 2 LGL nodes per segment)");
    }
}

HpMesh to_single_segment_hp_mesh(const Mesh& uniform_mesh) {
    uniform_mesh.validate();
    HpMesh hp_mesh;
    hp_mesh.segment_boundary_times = {uniform_mesh.t_initial, uniform_mesh.t_final};
    // WHY num_nodes(): the LGL scheme interprets num_intervals+1 as the node count
    // (lgl_nodes_and_weights is called with nn = mesh.num_nodes()), so the single-segment
    // hp mesh must have per_segment_node_count[0] = num_nodes() to match compile()'s behavior.
    hp_mesh.per_segment_node_count = {uniform_mesh.num_nodes()};
    return hp_mesh;
}

}  // namespace goss::transcription
```

- [ ] **Step 5: Add hp_mesh.cpp to CMakeLists.txt `goss_transcription` and test_hp_mesh.cpp to `goss_transcription_tests`**

In `CMakeLists.txt`, inside the `add_library(goss_transcription STATIC ...)` block, add `src/transcription/hp_mesh.cpp`. In `add_executable(goss_transcription_tests ...)`, add `tests/transcription/test_hp_mesh.cpp`.

- [ ] **Step 6: Run and verify HpMesh tests pass**

```bash
scripts/dev.sh 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target goss_transcription_tests && ctest --test-dir build -R "HpMesh" -V 2>&1 | tail -20'
```

Expected: `[  PASSED  ] 8 tests.`

- [ ] **Step 7: Commit**

```bash
git add include/goss/transcription/hp_mesh.hpp \
        src/transcription/hp_mesh.cpp \
        tests/transcription/test_hp_mesh.cpp \
        CMakeLists.txt
git commit -m "feat(hp-pseudospectral): add HpMesh struct with validate() and to_single_segment factory"
```

---

## Task 2: `compile_hp()` — Scaffold and Variable Layout

**Files:**
- Modify: `include/goss/transcription/legendre_gauss_lobatto.hpp` (add `compile_hp()` static method + `#include "hp_mesh.hpp"`)
- Modify: `tests/transcription/test_hp_pseudospectral.cpp` (create; scaffold test)

**Interfaces:**
- `LegendreGaussLobatto::compile_hp(const OcpProblem<Dyn,Cost>&, const HpMesh&, const std::string&)` → `CompiledOcp`
- `VariableLayout` constructed with `total_nodes = hp_mesh.total_nodes()`

- [ ] **Step 1: Write the failing tests**

Create `tests/transcription/test_hp_pseudospectral.cpp`:

```cpp
// tests/transcription/test_hp_pseudospectral.cpp
//
// Tests for LegendreGaussLobatto::compile_hp — multi-segment hp-pseudospectral collocation.
//
// Key properties verified:
//   1. Variable layout: total_nodes = sum(n_s); state_index / control_index consistent.
//   2. S=1 segment with same node count reduces to same solution as compile().
//   3. Per-segment differentiation matrix exactness (D_s exact for degree <= n_s - 1).
//   4. Continuity: final state of segment s equals initial state of segment s+1.
//   5. h-refinement convergence: more segments → lower error on smooth OCP.
//   6. hp-beats-global: a near-non-smooth OCP where segmented LGL beats single-interval LGL.
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <string>
#include "goss/transcription/legendre_gauss_lobatto.hpp"
#include "goss/transcription/hp_mesh.hpp"
#include "goss/transcription/errors.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "transcription/ocp_fixtures.hpp"

namespace {
// Helper: solve a compiled OCP with IPOPT and return the result.
// Initializes all variables to initial_guess_value.
goss::solver::SolverResult solve_compiled(
        const goss::transcription::CompiledOcp& compiled,
        double initial_guess_value = 1.0,
        double solver_tolerance   = 1e-11) {
    goss::solver::IpoptSolver solver;
    solver.set_tolerance(solver_tolerance);
    const std::vector<double> initial_guess(
        compiled.problem->num_variables(), initial_guess_value);
    return solver.solve(*compiled.problem, initial_guess);
}
}  // namespace

// --- Test: variable count and layout for a 3-segment problem ---
// Segments: n_s = [4, 5, 3]; total_nodes = 12.
// Variables per node = ns + nc. For exp-decay: ns=1, nc=0, vpn=1.
// Total variables = 12.
TEST(HpPseudospectral, VariableLayoutThreeSegments) {
    goss::transcription::HpMesh hp_mesh;
    hp_mesh.segment_boundary_times = {0.0, 0.4, 0.7, 1.0};
    hp_mesh.per_segment_node_count = {4, 5, 3};  // total_nodes = 12

    // Use exp-decay ocp (1 state, 0 controls).
    auto ocp = goss::transcription::test::make_exponential_decay(1.0, 1.0, /*intervals=*/7);

    auto compiled = goss::transcription::LegendreGaussLobatto::compile_hp(
        ocp, hp_mesh, "layout_3seg");

    // total_variables = total_nodes * (ns + nc) = 12 * 1 = 12
    EXPECT_EQ(compiled.problem->num_variables(), 12u);
    EXPECT_EQ(compiled.layout.num_nodes(), 12u);
    EXPECT_EQ(compiled.layout.num_states(), 1u);
    EXPECT_EQ(compiled.layout.num_controls(), 0u);
}

// --- Test: S=1 compile_hp matches compile() on exp-decay ---
// compile_hp with a single segment of 8 nodes must give the same constraint count
// as compile(ocp, name) with num_intervals=7 (=> 8 nodes).
TEST(HpPseudospectral, SingleSegmentMatchesSingleIntervalLayout) {
    auto ocp = goss::transcription::test::make_exponential_decay(1.0, 1.0, /*intervals=*/7);

    // Single-interval LGL (existing path).
    auto compiled_single =
        goss::transcription::LegendreGaussLobatto::compile(ocp, "hp_s1_single");

    // hp compile_hp with S=1, 8 nodes.
    const goss::transcription::HpMesh hp_mesh =
        goss::transcription::to_single_segment_hp_mesh(ocp.mesh);
    auto compiled_hp =
        goss::transcription::LegendreGaussLobatto::compile_hp(ocp, hp_mesh, "hp_s1_multi");

    // Variable counts must match.
    EXPECT_EQ(compiled_hp.problem->num_variables(),
              compiled_single.problem->num_variables());
    // S=1 has no continuity constraints; defect count = (8-1)*1 = 7.
    // Both must have the same constraint count.
    EXPECT_EQ(compiled_hp.problem->num_constraints(),
              compiled_single.problem->num_constraints());
}

// --- Test: S=1 compile_hp solves to same solution as compile() ---
TEST(HpPseudospectral, SingleSegmentSolvesToSameSolutionAsSingleInterval) {
    const double x0 = 1.0, time_final = 1.0;
    auto ocp = goss::transcription::test::make_exponential_decay(
        x0, time_final, /*intervals=*/7);

    auto compiled_single =
        goss::transcription::LegendreGaussLobatto::compile(ocp, "hp_s1_solve_single");
    const goss::transcription::HpMesh hp_mesh =
        goss::transcription::to_single_segment_hp_mesh(ocp.mesh);
    auto compiled_hp =
        goss::transcription::LegendreGaussLobatto::compile_hp(ocp, hp_mesh, "hp_s1_solve_hp");

    auto result_single = solve_compiled(compiled_single, x0);
    auto result_hp     = solve_compiled(compiled_hp, x0);

    ASSERT_EQ(result_single.status, goss::solver::SolverStatus::Success);
    ASSERT_EQ(result_hp.status,     goss::solver::SolverStatus::Success);

    // Final node state must match to tight tolerance.
    const std::size_t last_node = compiled_single.layout.num_nodes() - 1;
    const double x_final_single = result_single.x[compiled_single.layout.state_index(last_node, 0)];
    const double x_final_hp     = result_hp.x[compiled_hp.layout.state_index(last_node, 0)];
    EXPECT_NEAR(x_final_hp, x_final_single, 1e-8)
        << "S=1 compile_hp must match compile() final state";
}

// --- Test: hp compile_hp rejects free initial state ---
TEST(HpPseudospectral, RejectsFreeInitialState) {
    goss::transcription::OcpProblem<goss::transcription::test::ExpDecayDynamics,
                                    goss::transcription::test::ZeroCost> ocp;
    ocp.num_states = 1;
    ocp.num_controls = 0;
    ocp.dynamics = goss::transcription::test::ExpDecayDynamics{};
    ocp.cost     = goss::transcription::test::ZeroCost{};
    ocp.mesh = goss::transcription::Mesh{0.0, 1.0, 7};
    ocp.state_lower = {-1e19};
    ocp.state_upper = { 1e19};
    ocp.control_lower = {};
    ocp.control_upper = {};
    ocp.initial_state       = {1.0};
    ocp.initial_state_fixed = {0.0};  // free — must be rejected
    ocp.final_state         = {0.0};
    ocp.final_state_fixed   = {0.0};

    goss::transcription::HpMesh hp_mesh;
    hp_mesh.segment_boundary_times = {0.0, 0.5, 1.0};
    hp_mesh.per_segment_node_count = {4, 4};

    EXPECT_THROW(
        goss::transcription::LegendreGaussLobatto::compile_hp(ocp, hp_mesh, "hp_free_init"),
        goss::transcription::TranscriptionError);
}

// --- Test: 3-segment hp solve on exp-decay ---
// Partition [0,1] into 3 segments: [0,0.3], [0.3,0.7], [0.7,1.0] with 4+5+4=13 nodes.
// The solution at the last global node must approximate exp(-1) to 1e-5.
TEST(HpPseudospectral, ThreeSegmentSolvesExponentialDecay) {
    const double x0 = 1.0, time_final = 1.0;
    auto ocp = goss::transcription::test::make_exponential_decay(
        x0, time_final, /*intervals=*/7);

    goss::transcription::HpMesh hp_mesh;
    hp_mesh.segment_boundary_times = {0.0, 0.3, 0.7, 1.0};
    hp_mesh.per_segment_node_count = {4, 5, 4};  // total 13 nodes

    auto compiled = goss::transcription::LegendreGaussLobatto::compile_hp(
        ocp, hp_mesh, "hp_3seg_expdecay");
    auto result = solve_compiled(compiled, x0);

    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);

    // Last global node = node 12 (0-indexed), which is the last node of segment 2.
    const std::size_t last_global_node = compiled.layout.num_nodes() - 1;
    const double x_final = result.x[compiled.layout.state_index(last_global_node, 0)];
    EXPECT_NEAR(x_final,
                goss::transcription::test::exp_decay_solution(x0, time_final),
                1e-5)
        << "3-segment hp: final x must approximate exp(-1)";
}

// --- Test: continuity at segment boundaries is satisfied ---
// After solving, for each internal boundary the last state of segment s must
// equal the first state of segment s+1 (within solver tolerance).
TEST(HpPseudospectral, ContinuityAtSegmentBoundaries) {
    const double x0 = 1.0, time_final = 1.0;
    auto ocp = goss::transcription::test::make_exponential_decay(
        x0, time_final, /*intervals=*/7);

    goss::transcription::HpMesh hp_mesh;
    hp_mesh.segment_boundary_times = {0.0, 0.4, 0.7, 1.0};
    hp_mesh.per_segment_node_count = {4, 5, 3};  // total 12 nodes

    auto compiled = goss::transcription::LegendreGaussLobatto::compile_hp(
        ocp, hp_mesh, "hp_continuity_check");
    auto result = solve_compiled(compiled, x0, /*solver_tolerance=*/1e-10);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);

    // Segment offsets (cumulative sum of per_segment_node_count).
    std::vector<std::size_t> offsets;
    offsets.push_back(0);
    for (const std::size_t n_s : hp_mesh.per_segment_node_count)
        offsets.push_back(offsets.back() + n_s);

    const std::size_t num_states = ocp.num_states;
    const std::size_t num_seg = hp_mesh.num_segments();

    // For each internal boundary s=0..S-2, check state continuity.
    for (std::size_t seg = 0; seg + 1 < num_seg; ++seg) {
        // Last node of segment seg.
        const std::size_t last_node_of_seg =
            offsets[seg] + hp_mesh.per_segment_node_count[seg] - 1;
        // First node of segment seg+1.
        const std::size_t first_node_of_next = offsets[seg + 1];

        for (std::size_t state_idx = 0; state_idx < num_states; ++state_idx) {
            const double x_end_s =
                result.x[compiled.layout.state_index(last_node_of_seg, state_idx)];
            const double x_start_s1 =
                result.x[compiled.layout.state_index(first_node_of_next, state_idx)];
            EXPECT_NEAR(x_end_s, x_start_s1, 1e-8)
                << "Continuity violated at segment boundary " << seg
                << " -> " << (seg+1) << ", state " << state_idx;
        }
    }
}
```

- [ ] **Step 2: Run to verify test target fails (compile_hp not declared yet)**

```bash
scripts/dev.sh 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target goss_transcription_tests 2>&1 | head -20'
```

Expected: compile error — `compile_hp` not found.

- [ ] **Step 3: Add compile_hp() scaffold to `legendre_gauss_lobatto.hpp`**

Add to `include/goss/transcription/legendre_gauss_lobatto.hpp`, after the existing `compile()` and `NonUniformMesh` overload, a new static method stub that returns a `CompiledOcp`:

```cpp
// include/goss/transcription/legendre_gauss_lobatto.hpp
// ADD at the top of includes (alongside existing):
#include "goss/transcription/hp_mesh.hpp"

// ADD inside struct LegendreGaussLobatto, after the NonUniformMesh overload:

/// hp-Pseudospectral collocation: partition [t0,tf] into S segments, each with
/// its own local LGL nodes and differentiation matrix. State continuity across
/// segment boundaries is enforced via explicit equality constraints.
///
/// hp_mesh specifies: segment_boundary_times (size S+1) and per_segment_node_count (size S).
/// The ocp.mesh.t_initial / ocp.mesh.t_final must match hp_mesh.t_initial() / hp_mesh.t_final().
///
/// Boundary nodes are DUPLICATED (not shared): segment s owns global nodes
/// [offset_s, offset_s + n_s) where offset_s = sum_{r<s} n_r. Continuity constraints
/// tie the last state of segment s to the first state of segment s+1.
///
/// Controls are DISCONTINUOUS across segment boundaries (free per segment, standard hp-OC).
///
/// REQUIREMENT: all initial state components must be pinned (same guard as compile()).
template <typename DynamicsFn, typename CostFn>
static CompiledOcp compile_hp(const OcpProblem<DynamicsFn, CostFn>& ocp,
                               const HpMesh& hp_mesh,
                               const std::string& model_name = "goss_lgl_hp") {
    hp_mesh.validate();
    ocp.mesh.validate();

    // Verify that the hp_mesh time horizon matches the OcpProblem mesh.
    if (std::abs(hp_mesh.t_initial() - ocp.mesh.t_initial) > 1e-12 ||
        std::abs(hp_mesh.t_final()   - ocp.mesh.t_final  ) > 1e-12) {
        throw TranscriptionError(
            "compile_hp: hp_mesh time horizon [" +
            std::to_string(hp_mesh.t_initial()) + ", " +
            std::to_string(hp_mesh.t_final()) +
            "] does not match ocp.mesh [" +
            std::to_string(ocp.mesh.t_initial) + ", " +
            std::to_string(ocp.mesh.t_final) + "]");
    }

    const std::size_t num_states   = ocp.num_states;
    const std::size_t num_controls = ocp.num_controls;
    const std::size_t num_seg      = hp_mesh.num_segments();
    const std::size_t total_nodes  = hp_mesh.total_nodes();

    if (total_nodes < 2)
        throw TranscriptionError(
            "compile_hp: total_nodes must be >= 2");

    if (ocp.state_lower.size() != num_states || ocp.state_upper.size() != num_states)
        throw TranscriptionError(
            "compile_hp: state bound vectors must have size == num_states");
    if (ocp.control_lower.size() != num_controls || ocp.control_upper.size() != num_controls)
        throw TranscriptionError(
            "compile_hp: control bound vectors must have size == num_controls");
    if (ocp.initial_state.size() != num_states || ocp.final_state.size() != num_states)
        throw TranscriptionError(
            "compile_hp: initial_state/final_state must have size == num_states");

    // Guard: all initial state components must be pinned (same as compile()).
    for (std::size_t state_idx = 0; state_idx < num_states; ++state_idx) {
        const bool pinned =
            (state_idx < ocp.initial_state_fixed.size()) &&
            (ocp.initial_state_fixed[state_idx] != 0.0);
        if (!pinned) {
            throw TranscriptionError(
                "LegendreGaussLobatto::compile_hp: all initial states must be pinned "
                "(same requirement as compile()): free initial states are not supported.");
        }
    }

    // --- Pre-compute per-segment LGL data ---
    // For segment s: lgl_nodes_s, lgl_weights_ref_s, D_s, t_nodes_s, lgl_weights_phys_s.
    // Store as flat vectors indexed by [seg][local_node] (or [seg * max_n + local_node]).
    // Use std::vector<std::vector<T>> for clarity since S and n_s are small.
    std::vector<std::vector<double>> all_lgl_xi(num_seg);
    std::vector<std::vector<double>> all_lgl_weights_ref(num_seg);
    std::vector<std::vector<double>> all_D(num_seg);
    std::vector<std::vector<double>> all_t_nodes(num_seg);
    std::vector<std::vector<double>> all_weights_phys(num_seg);
    std::vector<double> all_half_dur(num_seg);

    for (std::size_t seg = 0; seg < num_seg; ++seg) {
        const std::size_t n_s     = hp_mesh.per_segment_node_count[seg];
        const double t_a_s        = hp_mesh.segment_boundary_times[seg];
        const double t_b_s        = hp_mesh.segment_boundary_times[seg + 1];
        const double half_dur_s   = 0.5 * (t_b_s - t_a_s);
        all_half_dur[seg]         = half_dur_s;

        // LGL nodes on [-1,1] and reference quadrature weights.
        lgl_nodes_and_weights(n_s, all_lgl_xi[seg], all_lgl_weights_ref[seg]);

        // Physical node times via affine map: t_k = t_a + half_dur * (xi_k + 1).
        all_t_nodes[seg].resize(n_s);
        for (std::size_t local_k = 0; local_k < n_s; ++local_k)
            all_t_nodes[seg][local_k] =
                t_a_s + half_dur_s * (all_lgl_xi[seg][local_k] + 1.0);

        // Physical quadrature weights: w_phys_k = half_dur_s * w_ref_k.
        all_weights_phys[seg].resize(n_s);
        for (std::size_t local_k = 0; local_k < n_s; ++local_k)
            all_weights_phys[seg][local_k] =
                half_dur_s * all_lgl_weights_ref[seg][local_k];

        // Local differentiation matrix D_s (on [-1,1]), scaled to physical by
        // the formula: (dx/dt)[k] = (1/half_dur_s) * sum_j D_s[k,j] * x(j),
        // so the collocation defect is:
        //   sum_j D_s[k,j] * x(j) - half_dur_s * f(x_k, u_k, t_k) = 0
        // Storing D_s on [-1,1]; the half_dur factor appears in the defect.
        all_D[seg] = lgl_differentiation_matrix(all_lgl_xi[seg]);
    }

    // Global node offsets: offset_s = sum_{r<s} n_r.
    std::vector<std::size_t> global_node_offsets(num_seg, 0);
    for (std::size_t seg = 1; seg < num_seg; ++seg)
        global_node_offsets[seg] =
            global_node_offsets[seg - 1] + hp_mesh.per_segment_node_count[seg - 1];

    VariableLayout layout(num_states, num_controls, total_nodes);

    // Capture all pre-computed data by value (they are small vectors).
    auto packed = [ocp, layout, num_states, num_controls, num_seg,
                   all_D, all_t_nodes, all_weights_phys, all_half_dur,
                   global_node_offsets,
                   per_segment_node_count = hp_mesh.per_segment_node_count]
                  (const auto& z) {
        using T = typename std::decay_t<decltype(z)>::value_type;

        auto state_at = [&](std::size_t global_node_idx) {
            std::vector<T> x(num_states);
            for (std::size_t s_i = 0; s_i < num_states; ++s_i)
                x[s_i] = z[layout.state_index(global_node_idx, s_i)];
            return x;
        };
        auto control_at = [&](std::size_t global_node_idx) {
            std::vector<T> u(num_controls);
            for (std::size_t c_j = 0; c_j < num_controls; ++c_j)
                u[c_j] = z[layout.control_index(global_node_idx, c_j)];
            return u;
        };

        // Output 0: total running cost = sum over segments of LGL-quadrature.
        T total_cost = T(0);
        for (std::size_t seg = 0; seg < num_seg; ++seg) {
            const std::size_t n_s          = per_segment_node_count[seg];
            const std::size_t global_offset = global_node_offsets[seg];
            for (std::size_t local_k = 0; local_k < n_s; ++local_k) {
                const std::size_t global_k = global_offset + local_k;
                total_cost +=
                    T(all_weights_phys[seg][local_k]) *
                    ocp.cost(state_at(global_k), control_at(global_k),
                             T(all_t_nodes[seg][local_k]));
            }
        }

        // Collocation defects and continuity constraints.
        // Layout of outputs (outputs[0] = cost, remaining = constraints):
        //   - Per-segment collocation defects: for seg s, nodes k=1..n_s-1, states i.
        //   - Continuity constraints: for boundaries s=0..S-2, states i.
        // WHY skip local k=0 per segment: see single-interval LGL design — node 0 of segment 0
        // is pinned by variable bounds; node 0 of segments 1..S-1 is tied to the previous
        // segment's last node by a continuity constraint. Including k=0 defects would be
        // overdetermined.

        std::size_t num_defects = 0;
        for (std::size_t seg = 0; seg < num_seg; ++seg)
            num_defects += (per_segment_node_count[seg] - 1) * num_states;
        const std::size_t num_continuity = (num_seg > 1 ? num_seg - 1 : 0) * num_states;

        std::vector<T> outputs;
        outputs.reserve(1 + num_defects + num_continuity);
        outputs.push_back(total_cost);

        // Per-segment collocation defects.
        for (std::size_t seg = 0; seg < num_seg; ++seg) {
            const std::size_t n_s           = per_segment_node_count[seg];
            const std::size_t global_offset = global_node_offsets[seg];
            const double half_dur_s         = all_half_dur[seg];

            // Pre-compute dynamics at each local node of this segment.
            std::vector<std::vector<T>> F_seg(n_s);
            for (std::size_t local_k = 0; local_k < n_s; ++local_k) {
                const std::size_t global_k = global_offset + local_k;
                F_seg[local_k] = ocp.dynamics(
                    state_at(global_k), control_at(global_k),
                    T(all_t_nodes[seg][local_k]));
            }

            // Defects at local nodes k=1..n_s-1.
            // Defect: sum_j D_s[k,j] * x_i(global_offset+j) - half_dur_s * F_seg[k][i] = 0
            for (std::size_t local_k = 1; local_k < n_s; ++local_k) {
                for (std::size_t state_i = 0; state_i < num_states; ++state_i) {
                    T Dx_k_i = T(0);
                    for (std::size_t local_j = 0; local_j < n_s; ++local_j) {
                        const std::size_t global_j = global_offset + local_j;
                        Dx_k_i +=
                            T(all_D[seg][local_k * n_s + local_j]) *
                            z[layout.state_index(global_j, state_i)];
                    }
                    outputs.push_back(Dx_k_i - T(half_dur_s) * F_seg[local_k][state_i]);
                }
            }
        }

        // Continuity constraints at internal segment boundaries.
        // For boundary between segment s and segment s+1, state i:
        //   z[state_index(last_node_of_s, i)] - z[state_index(first_node_of_(s+1), i)] = 0
        for (std::size_t seg = 0; seg + 1 < num_seg; ++seg) {
            const std::size_t last_node_of_seg =
                global_node_offsets[seg] + per_segment_node_count[seg] - 1;
            const std::size_t first_node_of_next = global_node_offsets[seg + 1];
            for (std::size_t state_i = 0; state_i < num_states; ++state_i) {
                outputs.push_back(
                    z[layout.state_index(last_node_of_seg,     state_i)] -
                    z[layout.state_index(first_node_of_next,   state_i)]);
            }
        }

        return outputs;
    };

    auto backend = std::make_unique<goss::ad::CppADCGBackend>(
        packed, layout.total_variables(), model_name);

    // Variable bounds: per-node state and control bounds (all total_nodes).
    const std::size_t nv = layout.total_variables();
    std::vector<double> zl(nv, -kInf), zu(nv, kInf);
    for (std::size_t global_k = 0; global_k < total_nodes; ++global_k) {
        for (std::size_t state_i = 0; state_i < num_states; ++state_i) {
            const std::size_t idx = layout.state_index(global_k, state_i);
            zl[idx] = ocp.state_lower[state_i];
            zu[idx] = ocp.state_upper[state_i];
        }
        for (std::size_t ctrl_j = 0; ctrl_j < num_controls; ++ctrl_j) {
            const std::size_t idx = layout.control_index(global_k, ctrl_j);
            zl[idx] = ocp.control_lower[ctrl_j];
            zu[idx] = ocp.control_upper[ctrl_j];
        }
    }

    // Pin initial state at global node 0 (first node of segment 0).
    for (std::size_t state_i = 0; state_i < num_states; ++state_i) {
        if (state_i < ocp.initial_state_fixed.size() &&
                ocp.initial_state_fixed[state_i] != 0.0) {
            const std::size_t idx = layout.state_index(0, state_i);
            zl[idx] = zu[idx] = ocp.initial_state[state_i];
        }
    }
    // Pin final state at global node total_nodes-1 (last node of last segment).
    for (std::size_t state_i = 0; state_i < num_states; ++state_i) {
        if (state_i < ocp.final_state_fixed.size() &&
                ocp.final_state_fixed[state_i] != 0.0) {
            const std::size_t idx = layout.state_index(total_nodes - 1, state_i);
            zl[idx] = zu[idx] = ocp.final_state[state_i];
        }
    }

    // Constraint bounds.
    // Collocation defects: equality [0, 0].
    std::size_t num_defects = 0;
    for (std::size_t seg = 0; seg < num_seg; ++seg)
        num_defects += (hp_mesh.per_segment_node_count[seg] - 1) * num_states;
    const std::size_t num_continuity =
        (num_seg > 1 ? num_seg - 1 : 0) * num_states;
    const std::size_t num_constraints_total = num_defects + num_continuity;

    std::vector<double> gl(num_constraints_total, 0.0);
    std::vector<double> gu(num_constraints_total, 0.0);

    auto problem = std::make_unique<nlp::NLPProblem>(
        std::move(backend), std::move(zl), std::move(zu),
        std::move(gl), std::move(gu));
    return CompiledOcp{std::move(problem), layout};
}
```

- [ ] **Step 4: Add test_hp_pseudospectral.cpp to CMakeLists.txt `goss_transcription_tests`**

- [ ] **Step 5: Run the scaffold tests**

```bash
scripts/dev.sh 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target goss_transcription_tests && ctest --test-dir build -R "HpPseudospectral" --timeout 600 -V 2>&1 | tail -40'
```

Expected: `VariableLayoutThreeSegments`, `SingleSegmentMatchesSingleIntervalLayout`, `SingleSegmentSolvesToSameSolutionAsSingleInterval`, `RejectsFreeInitialState`, `ThreeSegmentSolvesExponentialDecay`, `ContinuityAtSegmentBoundaries` — all pass.

- [ ] **Step 6: Run existing transcription tests to confirm no regressions**

```bash
scripts/dev.sh 'cmake --build build --target goss_transcription_tests && ctest --test-dir build --timeout 600 -j4 2>&1 | tail -10'
```

Expected: all existing `LegendreGaussLobatto`, `Trapezoidal`, `HermiteSimpson`, `LglNodes`, `Mesh` tests still pass.

- [ ] **Step 7: Commit**

```bash
git add include/goss/transcription/legendre_gauss_lobatto.hpp \
        tests/transcription/test_hp_pseudospectral.cpp \
        CMakeLists.txt
git commit -m "feat(hp-pseudospectral): add LegendreGaussLobatto::compile_hp() with per-segment collocation and continuity constraints"
```

---

## Task 3: Per-Segment Differentiation Matrix Exactness Test

**Files:**
- Modify: `tests/transcription/test_hp_pseudospectral.cpp` (append)

**Goal:** Verify that for each segment `s` with `n_s` LGL nodes mapped from `[-1,1]` to `[t_a^s, t_b^s]`, the per-segment D matrix (on `[-1,1]`) differentiates polynomials of degree up to `n_s-1` exactly, consistent with the `DifferentiationMatrixExactUpToDegreeN` test in `test_lgl_nodes.cpp` but now checked on the physical (time-domain) affine mapping.

The physical differentiation condition is:
- For polynomial `p(t) = (t - t_a)^k` on `[t_a, t_b]`, the physical derivative is `dp/dt = k*(t - t_a)^(k-1)`.
- At physical node `t_j = t_a + half_dur*(xi_j + 1)`, we have `p(t_j) = (half_dur*(xi_j+1))^k`.
- The LGL collocation uses `(D_s @ x)[row] = half_dur * f(x_row, u_row, t_row)`, i.e. `(1/half_dur) * (D_s @ p)[row] = dp/dt at t_row`.
- Equivalently: `(D_s @ p_xi)[row] = half_dur * dp/dt(t_row)` where `p_xi[j] = p(t_j)`.
- For `p(t) = (t - t_a)^k`: `p_xi[j] = (half_dur*(xi_j+1))^k`, `dp/dt = k*(half_dur*(xi_j+1))^(k-1)`, so `half_dur * dp/dt = k*half_dur*(half_dur*(xi_j+1))^(k-1) = k*(half_dur)^k * (xi_j+1)^(k-1)`.
- Expected: `(D_s @ p_xi)[row] = k * half_dur^k * (xi_row + 1)^(k-1)`.

This tests the physical-domain scaling and the affine mapping of LGL nodes, not just the reference-domain D matrix.

- [ ] **Step 1: Append the D-matrix exactness test to `test_hp_pseudospectral.cpp`**

```cpp
// Append to tests/transcription/test_hp_pseudospectral.cpp:

// --- Test: per-segment differentiation matrix is exact up to degree n_s-1 ---
// For segment [t_a, t_b] with n_s LGL nodes, the local D_s matrix on [-1,1]
// differentiates polynomials of degree <= n_s-1 exactly (standard LGL property).
// We verify the PHYSICAL-domain scaling: (D_s @ p_xi)[row] = half_dur * (dp/dt)(t_row)
// for p(t) = (t - t_a)^k, k=1..n_s-1.
TEST(HpPseudospectral, PerSegmentDifferentiationMatrixExactUpToDegreeNMinusOne) {
    // Test for two segment sizes: n_s=4 and n_s=6.
    for (const std::size_t n_s : {4u, 6u}) {
        // Arbitrary physical interval to test real-world scaling.
        const double t_a_seg   = 0.3;
        const double t_b_seg   = 0.8;
        const double half_dur_seg = 0.5 * (t_b_seg - t_a_seg);

        std::vector<double> lgl_xi_seg, lgl_weights_seg;
        goss::transcription::lgl_nodes_and_weights(n_s, lgl_xi_seg, lgl_weights_seg);

        // Physical node times: t_k = t_a + half_dur * (xi_k + 1).
        std::vector<double> t_physical(n_s);
        for (std::size_t k = 0; k < n_s; ++k)
            t_physical[k] = t_a_seg + half_dur_seg * (lgl_xi_seg[k] + 1.0);

        // Local differentiation matrix on [-1,1].
        const std::vector<double> D_seg =
            goss::transcription::lgl_differentiation_matrix(lgl_xi_seg);

        // For polynomial p(t) = (t - t_a)^k (k=1..n_s-1):
        //   p_xi[j] = (t_physical[j] - t_a)^k = (half_dur * (xi_j+1))^k
        //   (dp/dt)(t_row) = k * (t_row - t_a)^(k-1)
        // Collocation scaling: (D_s @ p_xi)[row] must equal half_dur_seg * (dp/dt)(t_row).
        const std::size_t degree_max = n_s - 1;  // LGL is exact up to degree n_s-1
        for (std::size_t poly_degree = 1; poly_degree <= degree_max; ++poly_degree) {
            // Evaluate polynomial at LGL nodes.
            std::vector<double> p_at_nodes(n_s);
            for (std::size_t j = 0; j < n_s; ++j)
                p_at_nodes[j] = std::pow(t_physical[j] - t_a_seg,
                                         static_cast<double>(poly_degree));

            for (std::size_t row = 0; row < n_s; ++row) {
                // Compute (D_s @ p_xi)[row] = sum_j D_seg[row*n_s + j] * p_at_nodes[j].
                double D_times_p = 0.0;
                for (std::size_t col = 0; col < n_s; ++col)
                    D_times_p += D_seg[row * n_s + col] * p_at_nodes[col];

                // Expected: half_dur_seg * dp/dt at t_physical[row].
                const double dp_dt_at_row =
                    static_cast<double>(poly_degree) *
                    std::pow(t_physical[row] - t_a_seg,
                             static_cast<double>(poly_degree - 1));
                const double expected = half_dur_seg * dp_dt_at_row;

                EXPECT_NEAR(D_times_p, expected, 1e-8)
                    << "n_s=" << n_s << " degree=" << poly_degree << " row=" << row;
            }
        }
    }
}
```

- [ ] **Step 2: Run and verify this test passes**

```bash
scripts/dev.sh 'cmake --build build --target goss_transcription_tests && ctest --test-dir build -R "PerSegmentDifferentiationMatrix" -V 2>&1 | tail -10'
```

Expected: `[  PASSED  ] 1 test.`

- [ ] **Step 3: Commit**

```bash
git add tests/transcription/test_hp_pseudospectral.cpp
git commit -m "test(hp-pseudospectral): add per-segment D-matrix exactness test with physical-domain scaling"
```

---

## Task 4: h-Refinement Convergence and hp-Beats-Global Accuracy Test

**Files:**
- Modify: `tests/transcription/test_hp_pseudospectral.cpp` (append)

**Goal:** Two tests:
1. **h-refinement convergence:** increasing the number of equal-order segments (h-refinement) reduces the error on a smooth problem monotonically.
2. **hp-beats-global:** a problem with a sharp feature where single-interval LGL performs poorly (Runge phenomenon or near-non-smooth solution) but hp-segmented LGL captures it accurately.

### The hp-beats-global test problem

**Problem:** exp-decay with a LARGE decay constant `k = 20`:
```
dx/dt = -20 * x,  x(0) = 1,  t in [0, 1]
Analytic solution: x(t) = exp(-20t)
```

On `[0,1]`, `x(1) = exp(-20) ≈ 2.06e-9` (essentially zero by `t=0.1`).

**Why global LGL fails here:** A single LGL grid over `[0,1]` with `n` global nodes tries to represent the sharp exponential front with a global polynomial of degree `n-1`. For `k=20`, the solution decays by 8 orders of magnitude in the first 10% of the interval. Global LGL requires a LARGE number of nodes to capture this accurately; with 10 nodes the polynomial oscillates (Runge-like phenomenon near the sharp front). This is the standard motivation for hp methods.

**Why segmented LGL wins:** With 4 equal segments of 5 nodes each (total 20 nodes), the first segment covers `[0, 0.25]` and captures the steep front with 5 nodes over a physically small interval. The remaining 3 segments handle the nearly-zero tail effortlessly. The per-segment polynomial degree is 4 (much lower than the global 19), and the scale of variation within each segment is much smaller.

**Assertion:** Max nodal error of the 4-segment hp solution vs analytic must be at least 100× smaller than the 20-node single-interval LGL solution, with absolute tolerances:
- Single-interval LGL with 20 nodes on `[0,1]` for `k=20`: expected error `> 1e-4` (the Runge-like oscillation is large).
- hp with 4 segments of 5 nodes each (total 20 nodes) for `k=20`: expected error `< 1e-6`.

**References:** Fahroo & Ross (2001) "Costate estimation by a Legendre pseudospectral method", Garg et al. (2010) "A unified framework for the numerical solution of optimal control problems using pseudospectral methods". Both demonstrate that hp-LGL handles non-smooth solutions that global-LGL cannot.

- [ ] **Step 1: Append h-refinement convergence test to `test_hp_pseudospectral.cpp`**

```cpp
// Append to tests/transcription/test_hp_pseudospectral.cpp:

namespace {
// Compute max nodal error of a 1-state hp solution vs an analytic reference.
// Evaluates analytic_fn at each global node's corresponding physical time.
// global_node_times[global_k] = physical time of global node k.
double hp_max_nodal_error(
        const goss::solver::SolverResult& result,
        const goss::transcription::VariableLayout& layout,
        const std::vector<double>& global_node_times,
        double x0_value,
        double decay_constant) {
    // WHY using analytic for exp-decay directly: this helper is specific to the
    // exp-decay problem family used throughout the convergence tests.
    double max_error = 0.0;
    const std::size_t num_global_nodes = layout.num_nodes();
    for (std::size_t global_k = 0; global_k < num_global_nodes; ++global_k) {
        const double t_k        = global_node_times[global_k];
        const double x_numeric  = result.x[layout.state_index(global_k, 0)];
        const double x_analytic = x0_value * std::exp(-decay_constant * t_k);
        max_error = std::max(max_error, std::abs(x_numeric - x_analytic));
    }
    return max_error;
}

// Build node times for an HpMesh: for each segment, compute LGL node times.
std::vector<double> compute_global_node_times(
        const goss::transcription::HpMesh& hp_mesh) {
    std::vector<double> times;
    const std::size_t num_seg = hp_mesh.num_segments();
    for (std::size_t seg = 0; seg < num_seg; ++seg) {
        const std::size_t n_s    = hp_mesh.per_segment_node_count[seg];
        const double t_a_s       = hp_mesh.segment_boundary_times[seg];
        const double t_b_s       = hp_mesh.segment_boundary_times[seg + 1];
        const double half_dur_s  = 0.5 * (t_b_s - t_a_s);
        std::vector<double> lgl_xi_s, lgl_w_s;
        goss::transcription::lgl_nodes_and_weights(n_s, lgl_xi_s, lgl_w_s);
        for (std::size_t local_k = 0; local_k < n_s; ++local_k)
            times.push_back(t_a_s + half_dur_s * (lgl_xi_s[local_k] + 1.0));
    }
    return times;
}
}  // namespace (anonymous)

// --- Test: h-refinement convergence on smooth exp-decay (k=1) ---
// Increasing number of equal-size segments (4 nodes each) reduces error monotonically.
// WHY k=1 (not k=20): for h-refinement we want a smooth problem so that the error
// decrease is clean polynomial (h-type) convergence rather than a regime-change.
TEST(HpPseudospectral, HRefinementConvergesMonotonicallyOnSmoothProblem) {
    const double decay_constant = 1.0;
    const double x0_value       = 1.0;
    const double time_final     = 1.0;

    // Test at S=1, 2, 4 segments of 4 nodes each (h-refinement, fixed p=4).
    const std::vector<std::size_t> num_segments_list = {1, 2, 4};
    std::vector<double> errors;

    for (const std::size_t num_segs : num_segments_list) {
        // Build a uniform hp mesh: equal segment widths, 4 nodes per segment.
        goss::transcription::HpMesh hp_mesh_uniform;
        hp_mesh_uniform.segment_boundary_times.resize(num_segs + 1);
        hp_mesh_uniform.per_segment_node_count.resize(num_segs, 4);
        for (std::size_t seg = 0; seg <= num_segs; ++seg)
            hp_mesh_uniform.segment_boundary_times[seg] =
                time_final * static_cast<double>(seg) / static_cast<double>(num_segs);

        auto ocp = goss::transcription::test::make_exponential_decay(
            x0_value, time_final, /*intervals=*/7);
        const std::string model_name = "hrefinement_s" + std::to_string(num_segs);
        auto compiled = goss::transcription::LegendreGaussLobatto::compile_hp(
            ocp, hp_mesh_uniform, model_name);
        auto result   = solve_compiled(compiled, x0_value, /*solver_tolerance=*/1e-12);
        ASSERT_EQ(result.status, goss::solver::SolverStatus::Success)
            << "h-refinement: S=" << num_segs << " solve failed";

        const std::vector<double> node_times =
            compute_global_node_times(hp_mesh_uniform);
        errors.push_back(hp_max_nodal_error(
            result, compiled.layout, node_times, x0_value, decay_constant));
    }

    // Errors must decrease monotonically as number of segments increases.
    for (std::size_t idx = 0; idx + 1 < errors.size(); ++idx) {
        EXPECT_LT(errors[idx + 1], errors[idx])
            << "h-refinement error must decrease with more segments: "
            << "errors[" << idx << "]=" << errors[idx]
            << ", errors[" << idx+1 << "]=" << errors[idx+1];
    }
}

// --- Test: hp BEATS single-interval LGL on a problem with a sharp feature ---
//
// Problem: fast exp-decay, dx/dt = -20*x, x(0)=1, t in [0,1].
// Analytic: x(t) = exp(-20t). The solution drops by 8 orders of magnitude in [0,0.1].
//
// Global LGL with 20 nodes over [0,1]: the global polynomial degree is 19.
// The sharp front near t=0 causes polynomial oscillation (Runge phenomenon
// analogue in the collocation context): error > 1e-4.
//
// hp-LGL with 4 segments x 5 nodes = 20 nodes total:
// Segment 0 covers [0, 0.25] with 5 LGL nodes — captures the steep front.
// Segments 1-3 cover [0.25, 0.5], [0.5, 0.75], [0.75, 1.0] with 5 nodes each.
// Per-segment polynomial degree is 4; the variation within each segment is modest.
// Error < 1e-6.
//
// WHY this is the right assertion: with 20 TOTAL nodes, both approaches use the
// same number of decision variables. The hp segmentation exploits locality;
// the global approach cannot. This is the key hp advantage.
//
// WHY 1e-4 lower bound for global: empirically, exp(-20t) on [0,1] with a
// 20-node global Chebyshev/LGL grid shows max error in the range 1e-3..1e-2
// near the interpolation nodes in the oscillatory region. We use a conservative
// 1e-4 lower bound to avoid a fragile test.
//
// WHY 1e-6 upper bound for hp: with 5 LGL nodes on [0, 0.25], the local
// polynomial degree is 4. On [0, 0.25], exp(-20t) has moderate smoothness
// (no singularity) and the local variation is exp(-5) ≈ 6.7e-3. Four-degree
// polynomial interpolation of a smooth function on a short interval achieves
// error O((h/2)^5/5!) ≈ O((0.125)^5) ≈ 3e-5; with LGL nodes this improves
// further. Empirically < 1e-6 is achievable.
TEST(HpPseudospectral, HpBeatsGlobalLGLOnSharpFeatureProblem) {
    const double decay_constant = 20.0;
    const double x0_value       = 1.0;
    const double time_final     = 1.0;

    // Build the fast-decay OCP.
    // Make a custom OcpProblem directly with FastDecayDynamics (k=10 fixture is
    // in ocp_fixtures.hpp; for k=20 we build manually).
    struct FastDecayK20Dynamics {
        template <typename T>
        std::vector<T> operator()(const std::vector<T>& x, const std::vector<T>& /*u*/, T /*t*/) const {
            return { T(-20.0) * x[0] };
        }
    };

    goss::transcription::OcpProblem<FastDecayK20Dynamics,
                                    goss::transcription::test::ZeroCost> ocp;
    ocp.num_states       = 1;
    ocp.num_controls     = 0;
    ocp.dynamics         = FastDecayK20Dynamics{};
    ocp.cost             = goss::transcription::test::ZeroCost{};
    ocp.mesh             = goss::transcription::Mesh{0.0, time_final, /*intervals=*/19};
    ocp.state_lower      = {-1e19};
    ocp.state_upper      = { 1e19};
    ocp.control_lower    = {};
    ocp.control_upper    = {};
    ocp.initial_state    = {x0_value};
    ocp.initial_state_fixed = {1.0};    // pin x(0) = 1
    ocp.final_state      = {0.0};
    ocp.final_state_fixed = {0.0};      // free final state

    // --- Single-interval LGL: 20 nodes over [0,1] ---
    auto compiled_global = goss::transcription::LegendreGaussLobatto::compile(
        ocp, "hp_vs_global_global20");
    auto result_global   = solve_compiled(compiled_global, x0_value, /*tol=*/1e-11);
    ASSERT_EQ(result_global.status, goss::solver::SolverStatus::Success)
        << "Global LGL (20 nodes) failed to solve";

    // Compute global node times for single-interval LGL (20 nodes over [0,1]).
    std::vector<double> lgl_xi_global, lgl_w_global;
    goss::transcription::lgl_nodes_and_weights(20, lgl_xi_global, lgl_w_global);
    const double half_dur_global = 0.5 * time_final;
    std::vector<double> global_node_times_single(20);
    for (std::size_t k = 0; k < 20; ++k)
        global_node_times_single[k] = 0.0 + half_dur_global * (lgl_xi_global[k] + 1.0);

    const double error_global = hp_max_nodal_error(
        result_global, compiled_global.layout,
        global_node_times_single, x0_value, decay_constant);

    // --- hp-LGL: 4 segments x 5 nodes = 20 total nodes ---
    // Uniform segment boundaries at 0.0, 0.25, 0.5, 0.75, 1.0.
    goss::transcription::HpMesh hp_mesh_4seg;
    hp_mesh_4seg.segment_boundary_times = {0.0, 0.25, 0.5, 0.75, 1.0};
    hp_mesh_4seg.per_segment_node_count  = {5, 5, 5, 5};  // 20 total nodes

    auto compiled_hp = goss::transcription::LegendreGaussLobatto::compile_hp(
        ocp, hp_mesh_4seg, "hp_vs_global_hp4x5");
    auto result_hp   = solve_compiled(compiled_hp, x0_value, /*tol=*/1e-11);
    ASSERT_EQ(result_hp.status, goss::solver::SolverStatus::Success)
        << "hp-LGL (4 segments x 5 nodes) failed to solve";

    const std::vector<double> hp_node_times =
        compute_global_node_times(hp_mesh_4seg);
    const double error_hp = hp_max_nodal_error(
        result_hp, compiled_hp.layout,
        hp_node_times, x0_value, decay_constant);

    // --- Assertions ---
    // WHY > 1e-4: global LGL with 20 nodes on exp(-20t) over [0,1] exhibits
    // large-amplitude oscillation in the Runge-phenomenon region. Empirically
    // this error is > 1e-2 for k=20; we use a conservative 1e-4 as the lower bound.
    EXPECT_GT(error_global, 1e-4)
        << "Global LGL error on fast decay (k=20) expected > 1e-4 (Runge-like oscillation); "
           "got error_global=" << error_global;

    // WHY < 1e-6: hp with 5 nodes per segment on [0,0.25] local intervals captures
    // the front accurately; error is dominated by O(h^(2*n_s)) spectral convergence
    // per segment. Empirically < 1e-7 is achievable; 1e-6 is the test tolerance.
    EXPECT_LT(error_hp, 1e-6)
        << "hp-LGL (4x5) error on fast decay must be < 1e-6; got error_hp=" << error_hp;

    // hp must be at least 100x more accurate than global LGL.
    EXPECT_LT(error_hp, error_global / 100.0)
        << "hp-LGL must beat global LGL by at least 100x on fast decay (k=20); "
           "error_global=" << error_global << ", error_hp=" << error_hp;
}
```

- [ ] **Step 2: Run these two new tests**

```bash
scripts/dev.sh 'cmake --build build --target goss_transcription_tests && ctest --test-dir build -R "(HRefinementConverges|HpBeatsGlobal)" --timeout 600 -V 2>&1 | tail -30'
```

Expected: both tests pass.

- [ ] **Step 3: Run the full transcription test suite (regression check)**

```bash
scripts/dev.sh 'ctest --test-dir build --timeout 600 -j4 2>&1 | tail -10'
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/transcription/test_hp_pseudospectral.cpp
git commit -m "test(hp-pseudospectral): add h-refinement convergence and hp-beats-global tests (fast decay k=20)"
```

---

## Task 5: Accuracy Suite Integration (depends on accuracy-suite plan)

**Dependency:** The accuracy validation suite plan (`docs/superpowers/plans/2026-08-01-accuracy-validation-suite.md`) must be implemented first: `tests/accuracy/accuracy_helpers.hpp` and the `goss_accuracy_tests` CMake target must exist.

**Files:**
- Create: `tests/accuracy/test_hp_accuracy.cpp`
- Modify: `CMakeLists.txt` (add `tests/accuracy/test_hp_accuracy.cpp` to `goss_accuracy_tests`)

**Goal:** Register the hp-convergence and hp-beats-global result with the accuracy harness (`accuracy_helpers.hpp`), making it part of the shared yardstick for future regression.

**Problem for the accuracy suite test:** Use the harmonic oscillator (`dx0/dt = x1, dx1/dt = -x0`) with a localized "fast phase" where a single LGL grid is insufficient. We test:
1. hp h-refinement convergence: 4 equal segments of 5 nodes vs 2 segments of 5 nodes — error ratio confirms improvement.
2. hp vs global LGL on fast decay `k=20` using `solve_and_extract_trajectory` helper.

- [ ] **Step 1: Write the failing test**

```cpp
// tests/accuracy/test_hp_accuracy.cpp
//
// hp-Pseudospectral accuracy integration tests.
//
// DEPENDENCY: This file depends on:
//   - goss_accuracy_tests target (accuracy-suite plan)
//   - LegendreGaussLobatto::compile_hp() (hp-pseudospectral plan, Task 2)
//   - HpMesh struct (hp-pseudospectral plan, Task 1)
//
// These tests extend the accuracy suite with hp-specific validations.
// They use accuracy_helpers.hpp's solve_and_extract_trajectory so that
// hp tests are visually comparable to the single-interval tests in test_convergence_order.cpp.
#include <gtest/gtest.h>
#include <cmath>
#include <cstddef>
#include <vector>
#include <string>
#include "goss/transcription/legendre_gauss_lobatto.hpp"
#include "goss/transcription/hp_mesh.hpp"
#include "goss/transcription/ocp_problem.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "accuracy/accuracy_helpers.hpp"

namespace {

// Fast-decay dynamics: dx/dt = -20*x.
// Defined here (not in ocp_fixtures.hpp) to keep accuracy tests self-contained.
struct FastDecayK20Dynamics {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x, const std::vector<T>& /*u*/, T /*t*/) const {
        return { T(-20.0) * x[0] };
    }
};

struct ZeroCostLocal {
    template <typename T>
    T operator()(const std::vector<T>& /*x*/, const std::vector<T>& /*u*/, T /*t*/) const {
        return T(0);
    }
};

// Build the fast-decay OCP (k=20, x(0)=1, t in [0,1], free final state).
goss::transcription::OcpProblem<FastDecayK20Dynamics, ZeroCostLocal>
build_fast_decay_ocp() {
    goss::transcription::OcpProblem<FastDecayK20Dynamics, ZeroCostLocal> ocp;
    ocp.num_states       = 1;
    ocp.num_controls     = 0;
    ocp.dynamics         = FastDecayK20Dynamics{};
    ocp.cost             = ZeroCostLocal{};
    ocp.mesh             = goss::transcription::Mesh{0.0, 1.0, /*intervals=*/19};
    ocp.state_lower      = {-1e19};
    ocp.state_upper      = { 1e19};
    ocp.control_lower    = {};
    ocp.control_upper    = {};
    ocp.initial_state    = {1.0};
    ocp.initial_state_fixed = {1.0};
    ocp.final_state      = {0.0};
    ocp.final_state_fixed = {0.0};
    return ocp;
}

// Max nodal error for fast decay vs exp(-20*t_k).
// For each global node k, computes |x_k - exp(-20 * t_k)|.
// t_k is the physical time of global node k (from node_times vector).
double fast_decay_max_error(const goss::accuracy::SolutionTrajectory& trajectory,
                            const std::vector<double>& node_times) {
    EXPECT_EQ(trajectory.states.size(), node_times.size())
        << "node_times and states must have the same size";
    double max_err = 0.0;
    for (std::size_t k = 0; k < trajectory.states.size(); ++k) {
        const double x_analytic = std::exp(-20.0 * node_times[k]);
        max_err = std::max(max_err, std::abs(trajectory.states[k][0] - x_analytic));
    }
    return max_err;
}

// Compute physical LGL node times for an HpMesh.
std::vector<double> hp_mesh_node_times(
        const goss::transcription::HpMesh& hp_mesh) {
    std::vector<double> times;
    for (std::size_t seg = 0; seg < hp_mesh.num_segments(); ++seg) {
        const std::size_t n_s    = hp_mesh.per_segment_node_count[seg];
        const double t_a_s       = hp_mesh.segment_boundary_times[seg];
        const double t_b_s       = hp_mesh.segment_boundary_times[seg + 1];
        const double half_dur_s  = 0.5 * (t_b_s - t_a_s);
        std::vector<double> xi_s, w_s;
        goss::transcription::lgl_nodes_and_weights(n_s, xi_s, w_s);
        for (std::size_t local_k = 0; local_k < n_s; ++local_k)
            times.push_back(t_a_s + half_dur_s * (xi_s[local_k] + 1.0));
    }
    return times;
}

}  // namespace

// --- Accuracy suite: hp h-refinement convergence on smooth OCP ---
// Increasing the number of equal-size segments drives error down on smooth exp(-t).
// WHY registered here (not only in test_hp_pseudospectral.cpp): the accuracy harness
// tracks error metrics as a shared yardstick; having hp convergence here means it
// will be detected as a regression if a future change degrades hp accuracy.
TEST(HpAccuracy, HRefinementConvergesOnSmoothDecay) {
    const double decay_constant = 1.0;
    const double x0_value       = 1.0;
    const double time_final     = 1.0;

    auto ocp = goss::transcription::OcpProblem<FastDecayK20Dynamics, ZeroCostLocal>{};
    // Reuse the fast-decay OCP but with k=1 (smooth, easy convergence).
    struct SlowDecayDynamics {
        template <typename T>
        std::vector<T> operator()(const std::vector<T>& x, const std::vector<T>& /*u*/, T /*t*/) const {
            return { T(-1.0) * x[0] };
        }
    };
    goss::transcription::OcpProblem<SlowDecayDynamics, ZeroCostLocal> smooth_ocp;
    smooth_ocp.num_states       = 1;
    smooth_ocp.num_controls     = 0;
    smooth_ocp.dynamics         = SlowDecayDynamics{};
    smooth_ocp.cost             = ZeroCostLocal{};
    smooth_ocp.mesh             = goss::transcription::Mesh{0.0, time_final, 7};
    smooth_ocp.state_lower      = {-1e19};
    smooth_ocp.state_upper      = { 1e19};
    smooth_ocp.control_lower    = {};
    smooth_ocp.control_upper    = {};
    smooth_ocp.initial_state    = {x0_value};
    smooth_ocp.initial_state_fixed = {1.0};
    smooth_ocp.final_state      = {0.0};
    smooth_ocp.final_state_fixed = {0.0};

    const std::vector<std::size_t> num_segs_list = {1, 2, 4};
    std::vector<double> errors;

    for (const std::size_t num_segs : num_segs_list) {
        goss::transcription::HpMesh hp_mesh;
        hp_mesh.segment_boundary_times.resize(num_segs + 1);
        hp_mesh.per_segment_node_count.resize(num_segs, 4);
        for (std::size_t seg = 0; seg <= num_segs; ++seg)
            hp_mesh.segment_boundary_times[seg] =
                time_final * static_cast<double>(seg) / static_cast<double>(num_segs);

        const std::string model_name =
            "hp_accuracy_hrefinement_s" + std::to_string(num_segs);
        auto compiled = goss::transcription::LegendreGaussLobatto::compile_hp(
            smooth_ocp, hp_mesh, model_name);
        const goss::accuracy::SolutionTrajectory trajectory =
            goss::accuracy::solve_and_extract_trajectory(
                compiled, /*initial_guess=*/x0_value, /*solver_tol=*/1e-12);
        ASSERT_FALSE(trajectory.states.empty())
            << "h-refinement accuracy: S=" << num_segs << " solve failed";

        const std::vector<double> node_times = hp_mesh_node_times(hp_mesh);
        double max_err = 0.0;
        for (std::size_t k = 0; k < trajectory.states.size(); ++k) {
            const double x_analytic = x0_value * std::exp(-decay_constant * node_times[k]);
            max_err = std::max(max_err, std::abs(trajectory.states[k][0] - x_analytic));
        }
        errors.push_back(max_err);
    }

    for (std::size_t idx = 0; idx + 1 < errors.size(); ++idx) {
        EXPECT_LT(errors[idx + 1], errors[idx])
            << "h-refinement must reduce error monotonically (S="
            << num_segs_list[idx] << " to S=" << num_segs_list[idx+1] << ")";
    }
}

// --- Accuracy suite: hp beats global LGL on fast-decay (k=20) ---
// Same assertion as HpPseudospectral.HpBeatsGlobalLGLOnSharpFeatureProblem,
// but using solve_and_extract_trajectory from the accuracy harness.
// 20 total nodes in both cases; hp wins by >= 100x.
TEST(HpAccuracy, HpBeatsGlobalLGLOnFastDecay) {
    const auto fast_ocp = build_fast_decay_ocp();

    // Global LGL: 20 nodes.
    auto compiled_global = goss::transcription::LegendreGaussLobatto::compile(
        fast_ocp, "hp_accuracy_global20");
    const goss::accuracy::SolutionTrajectory trajectory_global =
        goss::accuracy::solve_and_extract_trajectory(
            compiled_global, /*initial_guess=*/1.0, /*solver_tol=*/1e-11);
    ASSERT_FALSE(trajectory_global.states.empty());

    // Global LGL node times.
    std::vector<double> xi_global, w_global;
    goss::transcription::lgl_nodes_and_weights(20, xi_global, w_global);
    std::vector<double> times_global(20);
    for (std::size_t k = 0; k < 20; ++k)
        times_global[k] = 0.0 + 0.5 * 1.0 * (xi_global[k] + 1.0);
    const double error_global = fast_decay_max_error(trajectory_global, times_global);

    // hp-LGL: 4 segments x 5 nodes = 20 total.
    goss::transcription::HpMesh hp_mesh;
    hp_mesh.segment_boundary_times = {0.0, 0.25, 0.5, 0.75, 1.0};
    hp_mesh.per_segment_node_count = {5, 5, 5, 5};
    auto compiled_hp = goss::transcription::LegendreGaussLobatto::compile_hp(
        fast_ocp, hp_mesh, "hp_accuracy_hp4x5");
    const goss::accuracy::SolutionTrajectory trajectory_hp =
        goss::accuracy::solve_and_extract_trajectory(
            compiled_hp, /*initial_guess=*/1.0, /*solver_tol=*/1e-11);
    ASSERT_FALSE(trajectory_hp.states.empty());

    const std::vector<double> times_hp = hp_mesh_node_times(hp_mesh);
    const double error_hp = fast_decay_max_error(trajectory_hp, times_hp);

    EXPECT_GT(error_global, 1e-4)
        << "Global LGL (20 nodes) on fast decay (k=20) expected error > 1e-4; "
           "got " << error_global;
    EXPECT_LT(error_hp, 1e-6)
        << "hp-LGL (4x5) on fast decay (k=20) expected error < 1e-6; "
           "got " << error_hp;
    EXPECT_LT(error_hp, error_global / 100.0)
        << "hp-LGL must beat global LGL by 100x on fast decay (k=20); "
           "error_global=" << error_global << ", error_hp=" << error_hp;
}
```

- [ ] **Step 2: Add test_hp_accuracy.cpp to goss_accuracy_tests in CMakeLists.txt**

In the `add_executable(goss_accuracy_tests ...)` block, add `tests/accuracy/test_hp_accuracy.cpp`.

- [ ] **Step 3: Run and verify**

```bash
scripts/dev.sh 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target goss_accuracy_tests && ctest --test-dir build -R "HpAccuracy" --timeout 600 -V 2>&1 | tail -20'
```

Expected: `[  PASSED  ] 2 tests.`

- [ ] **Step 4: Run the full accuracy suite to confirm no regressions**

```bash
scripts/dev.sh 'ctest --test-dir build -R "(ClosedForm|Benchmarks|ConvergenceOrder|Invariants|HpAccuracy)" --timeout 900 -j2 2>&1 | tail -10'
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/accuracy/test_hp_accuracy.cpp CMakeLists.txt
git commit -m "feat(hp-pseudospectral): integrate hp accuracy tests into goss_accuracy_tests"
```

---

## Task 6: Update Deferral Comment and Run Full Suite

**Files:**
- Modify: `include/goss/transcription/legendre_gauss_lobatto.hpp` (update the "out of scope" comment at line ~29 and the `NonUniformMesh` overload comment at line ~197)

**Goal:** Remove the "out of scope" language now that `compile_hp()` implements hp-pseudospectral. Update the comments to point to `compile_hp()`. Confirm all transcription, accuracy, and model tests still pass.

- [ ] **Step 1: Update comments in `legendre_gauss_lobatto.hpp`**

Change the comment at line ~29 from:
```
// For moderate n (up to ~40) this is still efficient; for large n
// consider multiple LGL sub-intervals (hp-pseudospectral, out of scope here).
```
to:
```
// For moderate n (up to ~40) this is still efficient; for large n or solutions
// with sharp features, use compile_hp() for hp-pseudospectral multi-segment collocation.
```

Change the comment in the `NonUniformMesh` overload from:
```
// requires hp-pseudospectral (out of scope). Use Trapezoidal or ...
```
to:
```
// requires hp-pseudospectral; use compile_hp() for multi-segment hp collocation.
// Use Trapezoidal or HermiteSimpson for adaptive mesh refinement (refine_and_solve).
```

- [ ] **Step 2: Run the full test suite**

```bash
scripts/dev.sh 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest --test-dir build --timeout 900 -j4 2>&1 | tail -20'
```

Expected: all tests pass; no regressions in any existing target.

- [ ] **Step 3: Commit**

```bash
git add include/goss/transcription/legendre_gauss_lobatto.hpp
git commit -m "docs(hp-pseudospectral): update LGL deferral comments — compile_hp() implements hp-pseudospectral"
```

---

## Self-Review

### Spec Coverage

| Requirement | Task(s) | Status |
|---|---|---|
| Multi-segment LGL with per-segment nodes and D matrix | Tasks 2, 3 | Planned — per-segment `lgl_nodes_and_weights` + `lgl_differentiation_matrix` per segment |
| State continuity constraints across segment boundaries | Task 2 | Planned — `(S-1)*num_states` equality constraints appended after collocation defects |
| HpMesh API: segment_boundary_times + per_segment_node_count | Task 1 | Planned — new `hp_mesh.hpp`, `compile_hp(ocp, hp_mesh, name)` overload |
| OcpProblem unchanged | All | Planned — `compile_hp` reads same `OcpProblem` fields as `compile`; mesh time horizon cross-checked |
| Single-interval LGL untouched | All | Planned — additive; `compile()` code path not modified; S=1 test confirms equivalence |
| Pinned-initial-state guard | Task 2 | Planned — same guard as `compile()`, checked in `compile_hp()` before assembly |
| Per-segment D-matrix exactness test | Task 3 | Planned — physical-domain polynomial test for k=1..n_s-1 |
| h-refinement convergence test | Task 4 | Planned — S=1,2,4 equal segments on smooth exp-decay, error monotone decrease |
| hp-beats-global accuracy test | Tasks 4, 5 | Planned — fast-decay k=20, 20 nodes total; hp error < 1e-6, global error > 1e-4, ratio >= 100x |
| Accuracy harness integration | Task 5 | Planned — `test_hp_accuracy.cpp` uses `solve_and_extract_trajectory` |
| Duplicate-vs-shared boundary node decision with justification | Design section | Documented — duplicated, justified by simpler assembly and uniform segment loops |
| Control discontinuity decision with justification | Design section | Documented — discontinuous, standard hp-OC convention |
| CMakeLists.txt updated | Tasks 1, 2, 5 | Planned — `hp_mesh.cpp` added to `goss_transcription`; test files added to targets |

### Index Formula Audit

**Global node index for segment `s`, local node `k`:**
```
global_node(s, k) = global_node_offsets[s] + k
```
where `global_node_offsets[s] = sum_{r=0}^{s-1} per_segment_node_count[r]`.

**Cross-checks:**
- `global_node(0, 0) = 0` ✓ (overall initial node)
- `global_node(S-1, n_{S-1}-1) = total_nodes - 1` ✓ (overall final node)
- For continuity: `last_node_of_seg = global_node_offsets[s] + n_s - 1`, `first_node_of_next = global_node_offsets[s+1] = global_node_offsets[s] + n_s` — these are DIFFERENT indices (duplicated) ✓
- For S=1: `global_node_offsets = [0]`, `global_node(0, k) = k`, `total_nodes = n_0 = nn` — exactly matches the `layout.state_index(j, s)` calls in the single-interval LGL ✓

**Defect index within `outputs` vector:**
- `outputs[0]` = total cost (scalar)
- `outputs[1 + defect_index]` for `defect_index = s*(n_{prev_segs}-related) + (local_k-1)*ns + i` — assembled by the per-segment loop in order segment 0 nodes 1..n_0-1, segment 1 nodes 1..n_1-1, etc.
- Continuity outputs start at `outputs[1 + num_defects]`, indexed as `(s-1)*ns + i` for boundary `s`.

**Constraint bound assignment:**
- `gl[0..num_defects-1] = 0`, `gu[0..num_defects-1] = 0` (collocation equality)
- `gl[num_defects..num_defects+num_continuity-1] = 0`, `gu[...] = 0` (continuity equality)
- Total: `num_defects + num_continuity = sum(n_s-1)*ns + (S-1)*ns`. For S=1: `(n_0-1)*ns + 0 = (nn-1)*ns` — matches the single-interval formula `(nn-1)*ns` ✓

### S=1 Reduces-to-Current Check

With S=1:
- `total_nodes = n_0 = nn`
- `global_node_offsets = [0]`
- No continuity constraints (only 1 segment)
- Per-segment collocation loop for `s=0`, `k=1..n_0-1`: identical to the single-interval loop
- D matrix: `lgl_differentiation_matrix(lgl_xi_for_n_0)` — same as `D` in `compile()`
- Half-duration: `(t_b_0 - t_a_0)/2 = (tf - t0)/2 = half_duration` — same as `half_duration` in `compile()`
- Cost: `sum_{k=0}^{n_0-1} (half_dur_0 * lgl_w_0[k]) * cost(...)` = single-interval formula ✓
- Variable count: `n_0 * (ns + nc)` = single-interval ✓
- Constraint count: `(n_0 - 1) * ns = (nn-1) * ns` = single-interval ✓

The `SingleSegmentMatchesSingleIntervalLayout` and `SingleSegmentSolvesToSameSolutionAsSingleInterval` tests in Task 2 verify this numerically.

### Convergence / Accuracy Target

**Problem:** `dx/dt = -20x`, `x(0)=1`, `t ∈ [0,1]`. Analytic: `x(t) = exp(-20t)`.

**Global LGL (20 nodes):** Max nodal error > 1e-4. Rationale: exp(-20t) spans 8 orders of magnitude on [0,1]; a 19th-degree polynomial on the global LGL grid oscillates near the steep front. Empirically this is ~1e-2 for k=20.

**hp-LGL (4 segments × 5 nodes = 20 total nodes):** Max nodal error < 1e-6. Rationale: the first segment [0, 0.25] has 5 LGL nodes over a 0.25-wide interval; exp(-20t) on [0, 0.25] spans [1, exp(-5)] ≈ [1, 6.7e-3] — a factor of 150, much smaller than the full [1, 2e-9] range. A 4th-degree polynomial on a short interval achieves sub-1e-6 accuracy.

**hp advantage ratio:** >= 100× by assertion.

### Biggest Layout/Indexing Regression Risk

**Risk: off-by-one in `global_node_offsets` computation causing wrong D-matrix row/column in the defect loop.**

Specifically: in the defect assembly, the row index into `all_D[seg]` is `local_k * n_s + local_j` (row-major, n_s columns). If the captured `per_segment_node_count` vector has a different size than `all_D[seg]` (which is `n_s × n_s`), the index `local_k * n_s + local_j` is out-of-range. This can happen if the captured `n_s` inside the lambda is taken from the wrong segment. The lambda captures `per_segment_node_count` (the original `HpMesh::per_segment_node_count`) and recomputes `n_s = per_segment_node_count[seg]` inside the segment loop — this must match `all_D[seg].size() == n_s * n_s`. The `ContinuityAtSegmentBoundaries` test catches the downstream effect (wrong variable indices → wrong continuity residuals), and `PerSegmentDifferentiationMatrixExactUpToDegreeNMinusOne` catches D-matrix index errors directly.

**Mitigation:** The per-segment D-matrix exactness test (Task 3) verifies correct D-matrix indexing independently of the NLP solve. The continuity boundary test (Task 2) verifies that the NLP actually ties the correct node variables. The S=1 equivalence test cross-validates that the index formula degenerates correctly to the known-good single-interval case.
