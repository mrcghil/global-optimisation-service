# Transcription Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `transcription/` layer — a `Transcription` interface plus `Trapezoidal` and `HermiteSimpson` local collocation schemes that compile a continuous optimal-control problem (`OcpProblem`) into a `goss::nlp::NLPProblem` by packing states/controls at mesh nodes into one decision vector and writing defect constraints between nodes.

**Architecture:** `NLPProblem` is backed by ONE templated functor recorded by `CppADCGBackend`. So transcription's job is to synthesize one packed functor `F(z)`: `z` = states and controls at every mesh node flattened into a flat vector; output 0 = cost; outputs 1..m = all constraints (dynamics defects between nodes + boundary constraints). A `Transcription` scheme defines HOW defects are written. Trapezoidal: `x_{k+1}-x_k-(h/2)(f_k+f_{k+1})=0`. Hermite-Simpson: adds a midpoint state with an interpolation defect and Simpson quadrature. Both are LOCAL (banded sparsity), matching the sparse AD/solver stack. The continuous problem is described by a minimal `OcpProblem` struct owned by this layer (the full `model/` DSL is a later layer that will produce `OcpProblem` — do NOT build the DSL here).

**Tech Stack:** C++17, the merged `goss::ad` + `goss::nlp` + `goss::solver` layers, GoogleTest, CMake, containerized build via `scripts/dev.sh`.

## Global Constraints

- Language: **C++17**.
- Transcription produces a `goss::nlp::NLPProblem` via its constructor `NLPProblem(std::unique_ptr<ad::ADBackend>, var_lb, var_ub, con_lb, con_ub)`. The packed functor is recorded by `goss::ad::CppADCGBackend`.
- **Packing convention (fixed, matches NLPProblem):** functor output 0 = cost (objective); outputs 1..m = constraints. Decision vector layout `z` is fixed and documented once: for N intervals (N+1 nodes), states then controls per node — the EXACT layout is defined in Task 2 and every scheme + test uses it identically.
- The `Transcription` interface MUST depend only on `OcpProblem` + `goss::nlp`/`goss::ad` types — it is scheme-agnostic. Concrete schemes (`Trapezoidal`, `HermiteSimpson`) implement it.
- `OcpProblem` carries a **templated dynamics functor** `dx/dt = f(x, u, t)` and a **templated cost** so the SAME code records under `CppAD::AD<CG<double>>` and evaluates under `double`/`std::complex<double>`. Dynamics/cost use the ADL-friendly pattern (`using std::sin;`) like the ad-layer fixtures.
- Local schemes only (v1): Trapezoidal (O(h²)) and Hermite-Simpson (O(h⁴)). Pseudospectral/mesh-refinement are OUT of scope.
- **Definitive correctness test = convergence order:** on an analytically-integrable ODE, refine the mesh and confirm the solution error shrinks at the scheme's theoretical rate (trap O(h²), HS O(h⁴)). A wrong order means a wrong scheme.
- Error handling per org standard: a `TranscriptionError` exception for setup errors (dimension mismatch, empty mesh, etc.) with meaningful messages. No silent catch-all.
- Coding standards: verbose descriptive variable names; type annotations everywhere; comments explain WHY.
- Container-first build: all `cmake`/`ctest` run inside the container via `scripts/dev.sh '<command>'`. Never build on the host.
- Test framework: **GoogleTest**. Tests solve the transcribed NLP with the merged `IpoptSolver` and compare against analytic solutions.

---

## File Structure

- `include/goss/transcription/errors.hpp` — `TranscriptionError : std::runtime_error`.
- `include/goss/transcription/ocp_problem.hpp` — `OcpProblem`: the minimal continuous optimal-control description (dims, templated dynamics + cost, state/control bounds, boundary conditions, mesh). Header-only (templated functors).
- `include/goss/transcription/variable_layout.hpp` — `VariableLayout`: computes flat-vector indices for state i / control j at node k; the single source of truth for `z` packing. Header-only.
- `include/goss/transcription/transcription.hpp` — abstract `Transcription` interface (`compile(const OcpProblem&) -> std::unique_ptr<nlp::NLPProblem>`).
- `include/goss/transcription/trapezoidal.hpp` / `src/transcription/trapezoidal.cpp` — `Trapezoidal` scheme.
- `include/goss/transcription/hermite_simpson.hpp` / `src/transcription/hermite_simpson.cpp` — `HermiteSimpson` scheme.
- `tests/transcription/ocp_fixtures.hpp` — analytic OCP fixtures (ODEs with known closed-form solutions).
- `tests/transcription/test_variable_layout.cpp` — layout index tests.
- `tests/transcription/test_trapezoidal.cpp` — trapezoidal defect + solve + convergence-order tests.
- `tests/transcription/test_hermite_simpson.cpp` — HS defect + solve + convergence-order tests.
- `tests/transcription/test_scheme_agreement.cpp` — both schemes converge to the same solution.
- `CMakeLists.txt` — `goss_transcription` lib + `goss_transcription_tests`.

---

### Task 1: Scaffold transcription target + TranscriptionError

**Files:**
- Create: `include/goss/transcription/errors.hpp`
- Modify: `CMakeLists.txt`
- Create: `tests/transcription/test_variable_layout.cpp` (smoke portion only in this task)

**Interfaces:**
- Consumes: nothing new.
- Produces: `class goss::transcription::TranscriptionError : public std::runtime_error` (ctor from `const std::string&`); a `goss_transcription` static library target and a `goss_transcription_tests` executable wired into ctest. `goss_transcription` links `goss_nlp goss_ad` PUBLIC (it produces NLPProblems from CppADCGBackends).

- [ ] **Step 1: Write the failing smoke test**

```cpp
// tests/transcription/test_variable_layout.cpp
#include <gtest/gtest.h>
#include "goss/transcription/errors.hpp"

TEST(TranscriptionError, IsThrowable) {
    EXPECT_THROW(throw goss::transcription::TranscriptionError("boom"),
                 goss::transcription::TranscriptionError);
}
```

- [ ] **Step 2: Write the header**

```cpp
// include/goss/transcription/errors.hpp
#pragma once
#include <stdexcept>
#include <string>
namespace goss::transcription {
class TranscriptionError : public std::runtime_error {
 public:
    explicit TranscriptionError(const std::string& message) : std::runtime_error(message) {}
};
}  // namespace goss::transcription
```

- [ ] **Step 3: Wire CMake**

Append to `CMakeLists.txt` (after the `goss_solver_tests`/solver block):

```cmake
# ---- Transcription layer ----
add_library(goss_transcription STATIC
  src/transcription/trapezoidal.cpp
  src/transcription/hermite_simpson.cpp)
target_include_directories(goss_transcription PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(goss_transcription PUBLIC goss_nlp goss_ad
  PRIVATE goss_ad_impl cppadcg $<$<BOOL:${CPPAD_LIB}>:${CPPAD_LIB}>)

add_executable(goss_transcription_tests
  tests/transcription/test_variable_layout.cpp
  tests/transcription/test_trapezoidal.cpp
  tests/transcription/test_hermite_simpson.cpp
  tests/transcription/test_scheme_agreement.cpp)
target_include_directories(goss_transcription_tests PRIVATE
  ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(goss_transcription_tests PRIVATE
  goss_transcription goss_nlp goss_ad goss_ad_impl goss_solver
  goss_ipopt_iface goss_nlopt_iface cppadcg
  $<$<BOOL:${CPPAD_LIB}>:${CPPAD_LIB}> GTest::gtest_main)
gtest_discover_tests(goss_transcription_tests)
```

`goss_transcription` references `src/transcription/trapezoidal.cpp` and `hermite_simpson.cpp` (Tasks 4, 6) and the test target references test files not yet written (Tasks 2-7). To keep the build green NOW, create minimal placeholders: the two .cpp files each with only a comment + `#include` of their (future) header is premature — instead, for THIS task, temporarily list only `tests/transcription/test_variable_layout.cpp` in the test executable and create the two src .cpp as `// placeholder` files containing just a comment (no include). Add the other test files to the executable in the tasks that create them. (Simplest: create both `src/transcription/*.cpp` as comment-only placeholders now; add each real test file to CMake in its own task.)

- [ ] **Step 4: Create placeholder src files + trim the test list**

Create `src/transcription/trapezoidal.cpp` and `src/transcription/hermite_simpson.cpp` each containing only:
```cpp
// Implemented in a later task.
```
In the `goss_transcription_tests` `add_executable`, for THIS task list only `tests/transcription/test_variable_layout.cpp`. Later tasks add their test files.

- [ ] **Step 5: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake -S . -B build && cmake --build build && ctest --test-dir build -R TranscriptionError --output-on-failure'`
Expected: PASS. Confirm no regression: `scripts/dev.sh 'ctest --test-dir build --output-on-failure'` (44 prior + 1 new = 45).

- [ ] **Step 6: Commit**

```bash
git add include/goss/transcription/errors.hpp src/transcription/ tests/transcription/ CMakeLists.txt
git commit -m "build: scaffold transcription layer target with TranscriptionError"
```

---

### Task 2: VariableLayout — the flat decision-vector index map

**Files:**
- Create: `include/goss/transcription/variable_layout.hpp`
- Modify: `tests/transcription/test_variable_layout.cpp`

**Interfaces:**
- Consumes: `TranscriptionError`.
- Produces: `class goss::transcription::VariableLayout`.
  - Construct: `VariableLayout(std::size_t num_states, std::size_t num_controls, std::size_t num_nodes)` (num_nodes = N+1 for N intervals). Throws `TranscriptionError` if num_states==0 or num_nodes<2.
  - **Fixed layout:** `z` is grouped by node, states before controls within each node:
    `z = [ x_0(0..ns-1), u_0(0..nc-1), x_1(...), u_1(...), ..., x_{Nn-1}, u_{Nn-1} ]`
    where ns=num_states, nc=num_controls, Nn=num_nodes.
  - `std::size_t total_variables() const;` → `num_nodes * (num_states + num_controls)`.
  - `std::size_t state_index(std::size_t node, std::size_t state) const;` → flat index of state `state` at node `node`. Throws on out-of-range.
  - `std::size_t control_index(std::size_t node, std::size_t control) const;` → flat index. Throws on out-of-range.
  - Accessors `num_states()`, `num_controls()`, `num_nodes()`.

- [ ] **Step 1: Write the failing tests**

```cpp
// append to tests/transcription/test_variable_layout.cpp
#include "goss/transcription/variable_layout.hpp"

TEST(VariableLayout, ComputesTotalVariables) {
    goss::transcription::VariableLayout layout(2, 1, 3);  // 2 states, 1 control, 3 nodes
    EXPECT_EQ(layout.total_variables(), 3u * (2u + 1u));   // 9
}

TEST(VariableLayout, StateAndControlIndicesAreNodeGrouped) {
    goss::transcription::VariableLayout layout(2, 1, 3);
    // node 0: x0=0, x1=1, u0=2 ; node 1: x0=3, x1=4, u0=5 ; node 2: 6,7,8
    EXPECT_EQ(layout.state_index(0, 0), 0u);
    EXPECT_EQ(layout.state_index(0, 1), 1u);
    EXPECT_EQ(layout.control_index(0, 0), 2u);
    EXPECT_EQ(layout.state_index(1, 0), 3u);
    EXPECT_EQ(layout.control_index(1, 0), 5u);
    EXPECT_EQ(layout.state_index(2, 1), 7u);
}

TEST(VariableLayout, RejectsBadDimensions) {
    EXPECT_THROW(goss::transcription::VariableLayout(0, 1, 3), goss::transcription::TranscriptionError);
    EXPECT_THROW(goss::transcription::VariableLayout(2, 1, 1), goss::transcription::TranscriptionError);
}

TEST(VariableLayout, ZeroControlsIsAllowed) {
    goss::transcription::VariableLayout layout(2, 0, 4);  // states only
    EXPECT_EQ(layout.total_variables(), 8u);
    EXPECT_EQ(layout.state_index(3, 1), 7u);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `scripts/dev.sh 'cmake --build build 2>&1 | tail -20'`
Expected: FAIL — variable_layout.hpp not found.

- [ ] **Step 3: Write the header**

```cpp
// include/goss/transcription/variable_layout.hpp
#pragma once
#include <cstddef>
#include <string>
#include "goss/transcription/errors.hpp"

namespace goss::transcription {

/// Maps (node, state|control) coordinates to flat decision-vector indices.
/// Layout is grouped by node, states before controls within each node:
///   z = [ x_0, u_0, x_1, u_1, ..., x_{Nn-1}, u_{Nn-1} ]
/// This is the single source of truth for the z packing; every scheme and
/// test must use it so indices never diverge.
class VariableLayout {
 public:
    VariableLayout(std::size_t num_states, std::size_t num_controls, std::size_t num_nodes)
        : num_states_(num_states), num_controls_(num_controls), num_nodes_(num_nodes) {
        if (num_states_ == 0) {
            throw TranscriptionError("VariableLayout: num_states must be >= 1");
        }
        if (num_nodes_ < 2) {
            throw TranscriptionError("VariableLayout: num_nodes must be >= 2 (need >= 1 interval)");
        }
    }

    std::size_t num_states() const { return num_states_; }
    std::size_t num_controls() const { return num_controls_; }
    std::size_t num_nodes() const { return num_nodes_; }
    std::size_t variables_per_node() const { return num_states_ + num_controls_; }
    std::size_t total_variables() const { return num_nodes_ * variables_per_node(); }

    std::size_t state_index(std::size_t node, std::size_t state) const {
        if (node >= num_nodes_) throw TranscriptionError("VariableLayout::state_index: node out of range");
        if (state >= num_states_) throw TranscriptionError("VariableLayout::state_index: state out of range");
        return node * variables_per_node() + state;
    }

    std::size_t control_index(std::size_t node, std::size_t control) const {
        if (node >= num_nodes_) throw TranscriptionError("VariableLayout::control_index: node out of range");
        if (control >= num_controls_) throw TranscriptionError("VariableLayout::control_index: control out of range");
        return node * variables_per_node() + num_states_ + control;
    }

 private:
    std::size_t num_states_;
    std::size_t num_controls_;
    std::size_t num_nodes_;
};

}  // namespace goss::transcription
```

- [ ] **Step 4: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake --build build && ctest --test-dir build -R VariableLayout --output-on-failure'`
Expected: all 4 VariableLayout tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/goss/transcription/variable_layout.hpp tests/transcription/test_variable_layout.cpp
git commit -m "feat: VariableLayout flat decision-vector index map"
```

---
### Task 3: OcpProblem description + Transcription interface

**Files:**
- Create: `include/goss/transcription/ocp_problem.hpp`
- Create: `include/goss/transcription/transcription.hpp`
- Create: `tests/transcription/ocp_fixtures.hpp`
- Modify: `tests/transcription/test_variable_layout.cpp` (add an OcpProblem construction assertion)

**Interfaces:**
- Consumes: `VariableLayout`, `TranscriptionError`, `goss::nlp::NLPProblem`.
- Produces:
  - `struct goss::transcription::Mesh { double t_initial; double t_final; std::size_t num_intervals; };` with `num_nodes() const { return num_intervals + 1; }` and `interval_width() const { return (t_final - t_initial) / num_intervals; }` (uniform mesh v1). Throws via a `validate()` if num_intervals==0 or t_final<=t_initial.
  - `template <typename DynamicsFn, typename CostFn> struct goss::transcription::OcpProblem` holding:
    - `std::size_t num_states; std::size_t num_controls;`
    - `DynamicsFn dynamics;` — callable `template<typename T> std::vector<T> operator()(const std::vector<T>& x, const std::vector<T>& u, T t) const;` returning dx/dt (size num_states).
    - `CostFn cost;` — callable `template<typename T> T operator()(const std::vector<T>& x, const std::vector<T>& u, T t) const;` the RUNNING cost (integrated over time). v1 uses a running (Lagrange) cost integrated by the scheme's quadrature; terminal (Mayer) cost is out of scope for v1.
    - `Mesh mesh;`
    - `std::vector<double> state_lower, state_upper;` (size num_states; ±2e19 for free) applied at every node.
    - `std::vector<double> control_lower, control_upper;` (size num_controls) applied at every node.
    - `std::vector<double> initial_state; std::vector<double> initial_state_fixed;` — for each state, if `initial_state_fixed[i]!=0`, node 0's state i is pinned to `initial_state[i]` (boundary condition); size num_states.
    - `std::vector<double> final_state; std::vector<double> final_state_fixed;` — analogous for the last node.
  - Abstract `class goss::transcription::Transcription`:
    - protected default ctor; deleted copy/move; `virtual ~Transcription() = default;`
    - `template`-free interface is impossible (OcpProblem is templated), so the interface is a template method pattern: each concrete scheme is a class template `template<typename DynamicsFn, typename CostFn>` OR — simpler and chosen here — each scheme exposes a free/templated `compile` function. **Decision:** schemes are provided as **templated free functions** in their own namespaces, plus a thin non-template registration is NOT needed for v1. So `transcription.hpp` defines only shared helpers + the `kInf` constant + a `CompiledOcp` return bundle. Concrete schemes (Tasks 4, 6) provide `template<...> std::unique_ptr<nlp::NLPProblem> Trapezoidal::compile(const OcpProblem<...>&)`.
    - Define `constexpr double kInf = 2e19;` (matches NLPProblem free-bound convention).
    - Define `struct CompiledOcp { std::unique_ptr<nlp::NLPProblem> problem; VariableLayout layout; };` so callers get both the NLP and the layout to unpack solutions.
  - Fixture `goss::transcription::test::ExponentialDecay`: dynamics `dx/dt = -x` (1 state, 0 controls), cost `0` (feasibility problem), analytic solution `x(t) = x0 * exp(-t)`. Plus `goss::transcription::test::HarmonicOscillator`: 2 states `dx0=x1, dx1=-x0` (0 controls), analytic `x0(t)=x0(0)cos t + x1(0)sin t`.

**Design note:** Because `OcpProblem` is templated on the functor types, the `compile` entry points are templated too. The abstract `Transcription` base with a virtual `compile` cannot take a templated argument, so v1 uses templated `compile` free/static functions per scheme rather than runtime polymorphism. This keeps AD recording (which needs the concrete functor type) working. Runtime scheme selection, if ever needed, is a later concern — YAGNI now. Document this clearly in transcription.hpp.

- [ ] **Step 1: Write the failing test**

```cpp
// append to tests/transcription/test_variable_layout.cpp
#include "goss/transcription/ocp_problem.hpp"
#include "transcription/ocp_fixtures.hpp"

TEST(OcpProblem, MeshComputesNodesAndWidth) {
    goss::transcription::Mesh mesh{0.0, 2.0, 4};
    EXPECT_EQ(mesh.num_nodes(), 5u);
    EXPECT_DOUBLE_EQ(mesh.interval_width(), 0.5);
}

TEST(OcpProblem, ExponentialDecayFixtureEvaluates) {
    auto ocp = goss::transcription::test::make_exponential_decay(/*x0=*/1.0, /*tf=*/1.0, /*intervals=*/10);
    EXPECT_EQ(ocp.num_states, 1u);
    EXPECT_EQ(ocp.num_controls, 0u);
    std::vector<double> x{2.0}, u{};
    auto dx = ocp.dynamics(x, u, 0.0);
    ASSERT_EQ(dx.size(), 1u);
    EXPECT_DOUBLE_EQ(dx[0], -2.0);  // dx/dt = -x
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `scripts/dev.sh 'cmake --build build 2>&1 | tail -20'`
Expected: FAIL — ocp_problem.hpp / fixtures not found.

- [ ] **Step 3: Write ocp_problem.hpp**

```cpp
// include/goss/transcription/ocp_problem.hpp
#pragma once
#include <cstddef>
#include <vector>
#include "goss/transcription/errors.hpp"

namespace goss::transcription {

struct Mesh {
    double t_initial;
    double t_final;
    std::size_t num_intervals;
    std::size_t num_nodes() const { return num_intervals + 1; }
    double interval_width() const { return (t_final - t_initial) / static_cast<double>(num_intervals); }
    void validate() const {
        if (num_intervals == 0) throw TranscriptionError("Mesh: num_intervals must be >= 1");
        if (t_final <= t_initial) throw TranscriptionError("Mesh: t_final must be > t_initial");
    }
};

template <typename DynamicsFn, typename CostFn>
struct OcpProblem {
    std::size_t num_states;
    std::size_t num_controls;
    DynamicsFn dynamics;   // template<T> vector<T> (const vector<T>& x, const vector<T>& u, T t)
    CostFn cost;           // template<T> T (const vector<T>& x, const vector<T>& u, T t) — running cost
    Mesh mesh;
    std::vector<double> state_lower;
    std::vector<double> state_upper;
    std::vector<double> control_lower;
    std::vector<double> control_upper;
    std::vector<double> initial_state;
    std::vector<double> initial_state_fixed;   // nonzero => pin node 0 state i
    std::vector<double> final_state;
    std::vector<double> final_state_fixed;      // nonzero => pin last node state i
};

}  // namespace goss::transcription
```

- [ ] **Step 4: Write transcription.hpp (shared helpers)**

```cpp
// include/goss/transcription/transcription.hpp
#pragma once
#include <memory>
#include "goss/nlp/nlp_problem.hpp"
#include "goss/transcription/variable_layout.hpp"

namespace goss::transcription {

/// Free-bound sentinel matching NLPProblem's convention.
constexpr double kInf = 2e19;

/// Bundle returned by a scheme's compile(): the transcribed NLP plus the
/// layout needed to unpack a solution vector back into states/controls.
///
/// NOTE ON DESIGN: OcpProblem is templated on its dynamics/cost functor types
/// (so the SAME functor records under CppAD AD types and evaluates under
/// double). A runtime-polymorphic Transcription base with a virtual compile()
/// cannot accept a templated OcpProblem, so v1 exposes each scheme's compile()
/// as a templated static function (Trapezoidal::compile, HermiteSimpson::compile)
/// rather than through virtual dispatch. Runtime scheme selection is not needed
/// yet (YAGNI); if required later, a type-erased OcpProblem wrapper can be added.
struct CompiledOcp {
    std::unique_ptr<nlp::NLPProblem> problem;
    VariableLayout layout;
};

}  // namespace goss::transcription
```

- [ ] **Step 5: Write ocp_fixtures.hpp**

```cpp
// tests/transcription/ocp_fixtures.hpp
#pragma once
#include <cmath>
#include <cstddef>
#include <vector>
#include "goss/transcription/ocp_problem.hpp"

namespace goss::transcription::test {

// dx/dt = -x ; analytic x(t) = x0 * exp(-t). 1 state, 0 controls, zero cost.
struct ExpDecayDynamics {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x, const std::vector<T>& /*u*/, T /*t*/) const {
        return { -x[0] };
    }
};
struct ZeroCost {
    template <typename T>
    T operator()(const std::vector<T>& /*x*/, const std::vector<T>& /*u*/, T /*t*/) const {
        return T(0);
    }
};

inline auto make_exponential_decay(double x0, double tf, std::size_t intervals) {
    OcpProblem<ExpDecayDynamics, ZeroCost> ocp;
    ocp.num_states = 1;
    ocp.num_controls = 0;
    ocp.dynamics = ExpDecayDynamics{};
    ocp.cost = ZeroCost{};
    ocp.mesh = Mesh{0.0, tf, intervals};
    ocp.state_lower = { -1e19 };
    ocp.state_upper = { 1e19 };
    ocp.control_lower = {};
    ocp.control_upper = {};
    ocp.initial_state = { x0 };
    ocp.initial_state_fixed = { 1.0 };   // pin x(0) = x0
    ocp.final_state = { 0.0 };
    ocp.final_state_fixed = { 0.0 };      // free
    return ocp;
}

inline double exp_decay_solution(double x0, double t) { return x0 * std::exp(-t); }

// dx0=x1, dx1=-x0 ; analytic x0(t)=a cos t + b sin t with a=x0(0), b=x1(0). 2 states.
struct HarmonicDynamics {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x, const std::vector<T>& /*u*/, T /*t*/) const {
        return { x[1], -x[0] };
    }
};

inline auto make_harmonic(double x0_0, double x1_0, double tf, std::size_t intervals) {
    OcpProblem<HarmonicDynamics, ZeroCost> ocp;
    ocp.num_states = 2;
    ocp.num_controls = 0;
    ocp.dynamics = HarmonicDynamics{};
    ocp.cost = ZeroCost{};
    ocp.mesh = Mesh{0.0, tf, intervals};
    ocp.state_lower = { -1e19, -1e19 };
    ocp.state_upper = { 1e19, 1e19 };
    ocp.control_lower = {};
    ocp.control_upper = {};
    ocp.initial_state = { x0_0, x1_0 };
    ocp.initial_state_fixed = { 1.0, 1.0 };
    ocp.final_state = { 0.0, 0.0 };
    ocp.final_state_fixed = { 0.0, 0.0 };
    return ocp;
}

inline double harmonic_x0_solution(double a, double b, double t) {
    return a * std::cos(t) + b * std::sin(t);
}

}  // namespace goss::transcription::test
```

- [ ] **Step 6: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake --build build && ctest --test-dir build -R "OcpProblem" --output-on-failure'`
Expected: both OcpProblem tests PASS.

- [ ] **Step 7: Commit**

```bash
git add include/goss/transcription/ocp_problem.hpp include/goss/transcription/transcription.hpp tests/transcription/ocp_fixtures.hpp tests/transcription/test_variable_layout.cpp
git commit -m "feat: OcpProblem description, Transcription helpers, analytic OCP fixtures"
```

---

### Task 4: Trapezoidal scheme — packed functor + compile

**Files:**
- Create: `include/goss/transcription/trapezoidal.hpp`
- Modify: `src/transcription/trapezoidal.cpp` (may stay a comment; impl is header-templated — see note)
- Create: `tests/transcription/test_trapezoidal.cpp`
- Modify: `CMakeLists.txt` (add test_trapezoidal.cpp to the test executable)

**Interfaces:**
- Consumes: `OcpProblem`, `VariableLayout`, `CompiledOcp`, `kInf`, `goss::ad::CppADCGBackend`, `goss::nlp::NLPProblem`.
- Produces: `struct goss::transcription::Trapezoidal` with a static templated method:
  `template <typename DynamicsFn, typename CostFn> static CompiledOcp compile(const OcpProblem<DynamicsFn,CostFn>& ocp, const std::string& model_name = "goss_trap");`

**The packed functor (recorded by CppADCGBackend):** operator()(z) returns a `std::vector<T>` whose:
- **output 0 = cost:** trapezoidal quadrature of the running cost: `sum over intervals k of (h/2)*(L_k + L_{k+1})` where `L_k = ocp.cost(x_k, u_k, t_k)`. (Zero for feasibility problems.)
- **outputs 1..:** constraints, in this fixed order:
  1. **Defects:** for each interval k=0..N-1, for each state i: `x_{k+1}[i] - x_k[i] - (h/2)*(f_k[i] + f_{k+1}[i])` where `f_k = ocp.dynamics(x_k, u_k, t_k)`. (N*ns defect outputs.)
  The defect equality is enforced by giving these constraint bounds [0,0].

**Boundary conditions and bounds go into NLPProblem's bound vectors, NOT the functor:**
- Variable bounds: for every node, state i gets [state_lower[i], state_upper[i]], control j gets [control_lower[j], control_upper[j]]. For a PINNED initial/final state (initial_state_fixed[i]!=0), override that specific node-0 / node-N variable's lower==upper==the pinned value (a fixed variable — simplest, avoids an extra constraint).
- Constraint bounds: all defect constraints are equalities [0,0].

**Time at node k:** `t_k = t_initial + k*h`.

- [ ] **Step 1: Write the failing test (defect residual at the analytic solution)**

First test: the defect residuals must be ~0 when z holds a good discretization of the analytic solution — but trapezoidal is only O(h²) exact, so instead test the SOLVE: transcribe exp-decay, solve with IPOPT, compare to x0*exp(-t) at the final node.

```cpp
// tests/transcription/test_trapezoidal.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "goss/transcription/trapezoidal.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "transcription/ocp_fixtures.hpp"

TEST(Trapezoidal, SolvesExponentialDecay) {
    const double x0 = 1.0, tf = 1.0;
    const std::size_t intervals = 40;
    auto ocp = goss::transcription::test::make_exponential_decay(x0, tf, intervals);
    auto compiled = goss::transcription::Trapezoidal::compile(ocp, "trap_expdecay");

    goss::solver::IpoptSolver solver;
    // initial guess: all nodes = x0 (a crude but feasible-ish start)
    std::vector<double> z0(compiled.problem->num_variables(), x0);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);

    // final node state should match x0*exp(-tf)
    const auto& layout = compiled.layout;
    std::size_t last = layout.num_nodes() - 1;
    double x_final = result.x[layout.state_index(last, 0)];
    EXPECT_NEAR(x_final, goss::transcription::test::exp_decay_solution(x0, tf), 1e-3);
}

TEST(Trapezoidal, PinsInitialState) {
    auto ocp = goss::transcription::test::make_exponential_decay(2.0, 1.0, 20);
    auto compiled = goss::transcription::Trapezoidal::compile(ocp, "trap_pin");
    // node 0 state 0 must be pinned to 2.0 via equal var bounds
    std::size_t idx = compiled.layout.state_index(0, 0);
    EXPECT_DOUBLE_EQ(compiled.problem->variable_lower_bounds()[idx], 2.0);
    EXPECT_DOUBLE_EQ(compiled.problem->variable_upper_bounds()[idx], 2.0);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `scripts/dev.sh 'cmake --build build 2>&1 | tail -20'`
Expected: FAIL — trapezoidal.hpp not found (after adding test_trapezoidal.cpp to CMake).

- [ ] **Step 3: Write trapezoidal.hpp**

```cpp
// include/goss/transcription/trapezoidal.hpp
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "goss/ad/cppadcg_backend.hpp"
#include "goss/nlp/nlp_problem.hpp"
#include "goss/transcription/ocp_problem.hpp"
#include "goss/transcription/transcription.hpp"
#include "goss/transcription/variable_layout.hpp"

namespace goss::transcription {

struct Trapezoidal {
    template <typename DynamicsFn, typename CostFn>
    static CompiledOcp compile(const OcpProblem<DynamicsFn, CostFn>& ocp,
                               const std::string& model_name = "goss_trap") {
        ocp.mesh.validate();
        const std::size_t ns = ocp.num_states;
        const std::size_t nc = ocp.num_controls;
        const std::size_t nn = ocp.mesh.num_nodes();
        const std::size_t ni = ocp.mesh.num_intervals;
        const double t0 = ocp.mesh.t_initial;
        const double h = ocp.mesh.interval_width();
        VariableLayout layout(ns, nc, nn);

        // The packed functor: captures ocp by value (functors are cheap), layout by value.
        auto packed = [ocp, layout, ns, nc, nn, ni, t0, h](const auto& z) {
            using T = typename std::decay_t<decltype(z)>::value_type;
            std::vector<T> outputs;
            outputs.reserve(1 + ni * ns);

            // Helper lambdas to slice x_k, u_k from z.
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

            // Output 0: trapezoidal quadrature of running cost.
            T cost = T(0);
            for (std::size_t k = 0; k < nn; ++k) {
                T tk = T(t0 + static_cast<double>(k) * h);
                T Lk = ocp.cost(state_at(k), control_at(k), tk);
                // trapezoid weights: endpoints h/2, interior h
                T weight = (k == 0 || k == nn - 1) ? T(h / 2.0) : T(h);
                cost += weight * Lk;
            }
            outputs.push_back(cost);

            // Outputs 1..: defects per interval per state.
            for (std::size_t k = 0; k < ni; ++k) {
                T tk = T(t0 + static_cast<double>(k) * h);
                T tk1 = T(t0 + static_cast<double>(k + 1) * h);
                auto xk = state_at(k);
                auto xk1 = state_at(k + 1);
                auto fk = ocp.dynamics(xk, control_at(k), tk);
                auto fk1 = ocp.dynamics(xk1, control_at(k + 1), tk1);
                for (std::size_t i = 0; i < ns; ++i) {
                    outputs.push_back(xk1[i] - xk[i] - T(h / 2.0) * (fk[i] + fk1[i]));
                }
            }
            return outputs;
        };

        auto backend = std::make_unique<goss::ad::CppADCGBackend>(
            packed, layout.total_variables(), model_name);

        // Bounds.
        const std::size_t nv = layout.total_variables();
        std::vector<double> zl(nv, -kInf), zu(nv, kInf);
        for (std::size_t k = 0; k < nn; ++k) {
            for (std::size_t i = 0; i < ns; ++i) {
                std::size_t idx = layout.state_index(k, i);
                zl[idx] = ocp.state_lower[i];
                zu[idx] = ocp.state_upper[i];
            }
            for (std::size_t j = 0; j < nc; ++j) {
                std::size_t idx = layout.control_index(k, j);
                zl[idx] = ocp.control_lower[j];
                zu[idx] = ocp.control_upper[j];
            }
        }
        // Pin fixed boundary states via equal bounds.
        for (std::size_t i = 0; i < ns; ++i) {
            if (!ocp.initial_state_fixed.empty() && ocp.initial_state_fixed[i] != 0.0) {
                std::size_t idx = layout.state_index(0, i);
                zl[idx] = zu[idx] = ocp.initial_state[i];
            }
            if (!ocp.final_state_fixed.empty() && ocp.final_state_fixed[i] != 0.0) {
                std::size_t idx = layout.state_index(nn - 1, i);
                zl[idx] = zu[idx] = ocp.final_state[i];
            }
        }

        // Constraint bounds: all defects are equalities [0,0].
        const std::size_t num_defects = ni * ns;
        std::vector<double> gl(num_defects, 0.0), gu(num_defects, 0.0);

        auto problem = std::make_unique<nlp::NLPProblem>(
            std::move(backend), std::move(zl), std::move(zu), std::move(gl), std::move(gu));
        return CompiledOcp{std::move(problem), layout};
    }
};

}  // namespace goss::transcription
```

**Implementer note:** the packed lambda uses `const auto& z` + `typename std::decay_t<decltype(z)>::value_type` so it works for both `std::vector<double>` and `std::vector<CppAD::AD<CG<double>>>` — CppADCGBackend records by instantiating it with the AD vector type. Confirm CppADCGBackend accepts a generic lambda functor (it takes `template<typename F>`), which it does. `src/transcription/trapezoidal.cpp` can remain a comment-only placeholder since the implementation is header-templated; keep it in CMake so the lib has a TU, OR remove it from `add_library` and make goss_transcription INTERFACE. SIMPLEST: keep trapezoidal.cpp as a one-line TU `#include "goss/transcription/trapezoidal.hpp"` so the static lib is non-empty and the header compiles standalone.

- [ ] **Step 4: Make trapezoidal.cpp a real TU + add test to CMake**

Set `src/transcription/trapezoidal.cpp` to:
```cpp
#include "goss/transcription/trapezoidal.hpp"
// Header-templated scheme; this TU ensures the header compiles standalone.
```
Add `tests/transcription/test_trapezoidal.cpp` to the `goss_transcription_tests` `add_executable` list in CMakeLists.txt.

- [ ] **Step 5: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake -S . -B build && cmake --build build && ctest --test-dir build -R "Trapezoidal" --output-on-failure'`
Expected: SolvesExponentialDecay PASS (x_final ≈ 0.3679 within 1e-3) + PinsInitialState PASS. If the solve fails, check: constraint count = ni*ns; the pinned initial state; the initial guess feasibility.

- [ ] **Step 6: Commit**

```bash
git add include/goss/transcription/trapezoidal.hpp src/transcription/trapezoidal.cpp tests/transcription/test_trapezoidal.cpp CMakeLists.txt
git commit -m "feat: Trapezoidal collocation compiles OCP to NLP, solves exp-decay"
```

---
### Task 5: Trapezoidal convergence-order test (O(h²)) — the definitive correctness check

**Files:**
- Modify: `tests/transcription/test_trapezoidal.cpp`

**Interfaces:**
- Consumes: `Trapezoidal::compile`, `IpoptSolver`, harmonic + exp-decay fixtures.
- Produces: a test that refines the mesh and confirms the max nodal error shrinks at ~O(h²). No production code.

**Method:** For a sequence of interval counts (e.g. 10, 20, 40), solve exp-decay, measure the max error over all nodes vs the analytic solution, and confirm the error roughly QUARTERS when h HALVES (order 2: error ∝ h², so halving h → error × 1/4). Assert the observed order (from `log(e1/e2)/log(2)`) is ≥ 1.8 (allowing solver/tolerance slack).

- [ ] **Step 1: Write the convergence test**

```cpp
// append to tests/transcription/test_trapezoidal.cpp
#include <vector>

namespace {
// Solve exp-decay at a given interval count, return max nodal error vs analytic.
double trap_max_error(std::size_t intervals) {
    const double x0 = 1.0, tf = 1.0;
    auto ocp = goss::transcription::test::make_exponential_decay(x0, tf, intervals);
    auto compiled = goss::transcription::Trapezoidal::compile(
        ocp, "trap_conv_" + std::to_string(intervals));
    goss::solver::IpoptSolver solver;
    std::vector<double> z0(compiled.problem->num_variables(), x0);
    auto result = solver.solve(*compiled.problem, z0);
    if (result.status != goss::solver::SolverStatus::Success) return 1e9;
    const auto& layout = compiled.layout;
    const double h = tf / static_cast<double>(intervals);
    double max_err = 0.0;
    for (std::size_t k = 0; k < layout.num_nodes(); ++k) {
        double xk = result.x[layout.state_index(k, 0)];
        double exact = goss::transcription::test::exp_decay_solution(x0, k * h);
        max_err = std::max(max_err, std::abs(xk - exact));
    }
    return max_err;
}
}  // namespace

TEST(Trapezoidal, ConvergesAtSecondOrder) {
    double e_coarse = trap_max_error(10);
    double e_fine = trap_max_error(20);
    double e_finer = trap_max_error(40);
    ASSERT_LT(e_finer, e_fine);
    ASSERT_LT(e_fine, e_coarse);
    // order p ≈ log2(e(h)/e(h/2)); trapezoidal is O(h^2) ⇒ p≈2.
    double order1 = std::log(e_coarse / e_fine) / std::log(2.0);
    double order2 = std::log(e_fine / e_finer) / std::log(2.0);
    EXPECT_GE(order1, 1.8) << "trapezoidal should be ~2nd order";
    EXPECT_GE(order2, 1.8) << "trapezoidal should be ~2nd order";
}
```

- [ ] **Step 2: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake --build build && ctest --test-dir build -R "Trapezoidal.ConvergesAtSecondOrder" --output-on-failure'`
Expected: PASS — observed order ≥ 1.8. If order looks like ~4 (too good) or ~1, the defect formula is likely wrong; ~2 confirms trapezoidal. Note: IPOPT tol must be tight enough (default 1e-8) that solver error doesn't dominate discretization error at 40 intervals — if the fine error plateaus, tighten IpoptSolver tolerance in this test via solver.set_tolerance(1e-10).

- [ ] **Step 3: Commit**

```bash
git add tests/transcription/test_trapezoidal.cpp
git commit -m "test: trapezoidal converges at 2nd order on exp-decay"
```

---

### Task 6: HermiteSimpson scheme — midpoint state + interpolation defect + Simpson quadrature

**Files:**
- Create: `include/goss/transcription/hermite_simpson.hpp`
- Modify: `src/transcription/hermite_simpson.cpp` (real TU including the header)
- Create: `tests/transcription/test_hermite_simpson.cpp`
- Modify: `CMakeLists.txt` (add test_hermite_simpson.cpp)

**Interfaces:**
- Consumes: same as Trapezoidal.
- Produces: `struct goss::transcription::HermiteSimpson` with:
  `template <typename DynamicsFn, typename CostFn> static CompiledOcp compile(const OcpProblem<...>& ocp, const std::string& model_name = "goss_hs");`

**Hermite-Simpson (separated/compressed form — use COMPRESSED, no extra midpoint variables):** For interval k with endpoints x_k, x_{k+1}, controls u_k, u_{k+1}, step h:
- Interpolated midpoint state: `x_mid = (x_k + x_{k+1})/2 + (h/8)*(f_k - f_{k+1})`.
- Midpoint control: `u_mid = (u_k + u_{k+1})/2`.
- Midpoint dynamics: `f_mid = f(x_mid, u_mid, t_k + h/2)`.
- **Defect (collocation) constraint per state i:** `x_{k+1}[i] - x_k[i] - (h/6)*(f_k[i] + 4*f_mid[i] + f_{k+1}[i]) = 0`.
This "compressed" form does NOT add midpoint states to z — it derives x_mid analytically from the endpoints. So the decision vector layout is IDENTICAL to trapezoidal (same VariableLayout), only the defect formula changes. This keeps z packing consistent across schemes.

**Cost quadrature (Simpson):** `sum over intervals of (h/6)*(L_k + 4*L_mid + L_{k+1})` where L_mid uses x_mid, u_mid at the interval midpoint. (Zero for feasibility problems.)

Bounds/boundary handling: IDENTICAL to Trapezoidal (states/controls per node, pinned initial/final via equal bounds, defects [0,0]). Number of defect constraints is the same: ni*ns.

- [ ] **Step 1: Write the failing solve test**

```cpp
// tests/transcription/test_hermite_simpson.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "transcription/ocp_fixtures.hpp"

TEST(HermiteSimpson, SolvesExponentialDecay) {
    const double x0 = 1.0, tf = 1.0;
    auto ocp = goss::transcription::test::make_exponential_decay(x0, tf, 20);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "hs_expdecay");
    goss::solver::IpoptSolver solver;
    std::vector<double> z0(compiled.problem->num_variables(), x0);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    std::size_t last = compiled.layout.num_nodes() - 1;
    double x_final = result.x[compiled.layout.state_index(last, 0)];
    EXPECT_NEAR(x_final, goss::transcription::test::exp_decay_solution(x0, tf), 1e-5);
}

TEST(HermiteSimpson, SolvesHarmonicOscillator) {
    // x0(0)=1, x1(0)=0 -> x0(t)=cos t. Check x0(tf).
    const double tf = 1.0;
    auto ocp = goss::transcription::test::make_harmonic(1.0, 0.0, tf, 20);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "hs_harmonic");
    goss::solver::IpoptSolver solver;
    std::vector<double> z0(compiled.problem->num_variables(), 0.5);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    std::size_t last = compiled.layout.num_nodes() - 1;
    double x0_final = result.x[compiled.layout.state_index(last, 0)];
    EXPECT_NEAR(x0_final, goss::transcription::test::harmonic_x0_solution(1.0, 0.0, tf), 1e-4);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `scripts/dev.sh 'cmake --build build 2>&1 | tail -20'` (after adding test to CMake)
Expected: FAIL — hermite_simpson.hpp not found.

- [ ] **Step 3: Write hermite_simpson.hpp**

Mirror trapezoidal.hpp's structure exactly (same VariableLayout, same bounds/pinning/defect-count code), changing ONLY the packed functor's cost and defect formulas:

```cpp
// include/goss/transcription/hermite_simpson.hpp
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "goss/ad/cppadcg_backend.hpp"
#include "goss/nlp/nlp_problem.hpp"
#include "goss/transcription/ocp_problem.hpp"
#include "goss/transcription/transcription.hpp"
#include "goss/transcription/variable_layout.hpp"

namespace goss::transcription {

struct HermiteSimpson {
    template <typename DynamicsFn, typename CostFn>
    static CompiledOcp compile(const OcpProblem<DynamicsFn, CostFn>& ocp,
                               const std::string& model_name = "goss_hs") {
        ocp.mesh.validate();
        const std::size_t ns = ocp.num_states;
        const std::size_t nc = ocp.num_controls;
        const std::size_t nn = ocp.mesh.num_nodes();
        const std::size_t ni = ocp.mesh.num_intervals;
        const double t0 = ocp.mesh.t_initial;
        const double h = ocp.mesh.interval_width();
        VariableLayout layout(ns, nc, nn);

        auto packed = [ocp, layout, ns, nc, nn, ni, t0, h](const auto& z) {
            using T = typename std::decay_t<decltype(z)>::value_type;
            std::vector<T> outputs;
            outputs.reserve(1 + ni * ns);

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
            auto midpoint_control = [&](const std::vector<T>& uk, const std::vector<T>& uk1) {
                std::vector<T> um(nc);
                for (std::size_t j = 0; j < nc; ++j) um[j] = T(0.5) * (uk[j] + uk1[j]);
                return um;
            };

            // Cost (Simpson) + defects (Hermite-Simpson compressed).
            T cost = T(0);
            std::vector<T> defects;
            defects.reserve(ni * ns);
            for (std::size_t k = 0; k < ni; ++k) {
                T tk = T(t0 + static_cast<double>(k) * h);
                T tmid = T(t0 + (static_cast<double>(k) + 0.5) * h);
                T tk1 = T(t0 + static_cast<double>(k + 1) * h);
                auto xk = state_at(k);
                auto xk1 = state_at(k + 1);
                auto uk = control_at(k);
                auto uk1 = control_at(k + 1);
                auto fk = ocp.dynamics(xk, uk, tk);
                auto fk1 = ocp.dynamics(xk1, uk1, tk1);
                // Interpolated midpoint state.
                std::vector<T> xmid(ns);
                for (std::size_t i = 0; i < ns; ++i)
                    xmid[i] = T(0.5) * (xk[i] + xk1[i]) + T(h / 8.0) * (fk[i] - fk1[i]);
                auto umid = midpoint_control(uk, uk1);
                auto fmid = ocp.dynamics(xmid, umid, tmid);
                // Defect per state (Simpson collocation).
                for (std::size_t i = 0; i < ns; ++i)
                    defects.push_back(xk1[i] - xk[i] - T(h / 6.0) * (fk[i] + T(4) * fmid[i] + fk1[i]));
                // Simpson cost contribution.
                T Lk = ocp.cost(xk, uk, tk);
                T Lmid = ocp.cost(xmid, umid, tmid);
                T Lk1 = ocp.cost(xk1, uk1, tk1);
                cost += T(h / 6.0) * (Lk + T(4) * Lmid + Lk1);
            }
            outputs.push_back(cost);
            for (auto& d : defects) outputs.push_back(d);
            return outputs;
        };

        auto backend = std::make_unique<goss::ad::CppADCGBackend>(
            packed, layout.total_variables(), model_name);

        const std::size_t nv = layout.total_variables();
        std::vector<double> zl(nv, -kInf), zu(nv, kInf);
        for (std::size_t k = 0; k < nn; ++k) {
            for (std::size_t i = 0; i < ns; ++i) {
                std::size_t idx = layout.state_index(k, i);
                zl[idx] = ocp.state_lower[i];
                zu[idx] = ocp.state_upper[i];
            }
            for (std::size_t j = 0; j < nc; ++j) {
                std::size_t idx = layout.control_index(k, j);
                zl[idx] = ocp.control_lower[j];
                zu[idx] = ocp.control_upper[j];
            }
        }
        for (std::size_t i = 0; i < ns; ++i) {
            if (!ocp.initial_state_fixed.empty() && ocp.initial_state_fixed[i] != 0.0) {
                std::size_t idx = layout.state_index(0, i);
                zl[idx] = zu[idx] = ocp.initial_state[i];
            }
            if (!ocp.final_state_fixed.empty() && ocp.final_state_fixed[i] != 0.0) {
                std::size_t idx = layout.state_index(nn - 1, i);
                zl[idx] = zu[idx] = ocp.final_state[i];
            }
        }
        const std::size_t num_defects = ni * ns;
        std::vector<double> gl(num_defects, 0.0), gu(num_defects, 0.0);

        auto problem = std::make_unique<nlp::NLPProblem>(
            std::move(backend), std::move(zl), std::move(zu), std::move(gl), std::move(gu));
        return CompiledOcp{std::move(problem), layout};
    }
};

}  // namespace goss::transcription
```

**Implementer note:** the bounds/pinning/defect-count block is IDENTICAL to Trapezoidal (same VariableLayout, same output packing order: cost then defects). Only the interval loop's math differs (x_mid interpolation, Simpson defect h/6·(fk+4fmid+fk1), Simpson cost). Do NOT add midpoint variables to z — the compressed form derives x_mid from endpoints, keeping the layout identical to trapezoidal.

- [ ] **Step 4: Make hermite_simpson.cpp a real TU + add test to CMake**

Set `src/transcription/hermite_simpson.cpp` to `#include "goss/transcription/hermite_simpson.hpp"` + a one-line comment. Add `tests/transcription/test_hermite_simpson.cpp` to the test executable.

- [ ] **Step 5: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake -S . -B build && cmake --build build && ctest --test-dir build -R "HermiteSimpson" --output-on-failure'`
Expected: both HS tests PASS — exp-decay final within 1e-5 (much tighter than trapezoidal's 1e-3 at fewer intervals, since HS is O(h⁴)), harmonic within 1e-4.

- [ ] **Step 6: Commit**

```bash
git add include/goss/transcription/hermite_simpson.hpp src/transcription/hermite_simpson.cpp tests/transcription/test_hermite_simpson.cpp CMakeLists.txt
git commit -m "feat: Hermite-Simpson collocation (compressed) compiles and solves OCPs"
```

---

### Task 7: HS convergence-order (O(h⁴)) + cross-scheme agreement

**Files:**
- Modify: `tests/transcription/test_hermite_simpson.cpp`
- Create: `tests/transcription/test_scheme_agreement.cpp`
- Modify: `CMakeLists.txt` (test_scheme_agreement.cpp already in the executable list from Task 1)

**Interfaces:**
- Consumes: both schemes' `compile`, `IpoptSolver`, fixtures.
- Produces: HS 4th-order convergence test + a test that trapezoidal and HS converge to the same solution as the mesh refines.

- [ ] **Step 1: Write the HS convergence test (O(h⁴))**

```cpp
// append to tests/transcription/test_hermite_simpson.cpp
#include <vector>

namespace {
double hs_max_error(std::size_t intervals) {
    const double x0 = 1.0, tf = 1.0;
    auto ocp = goss::transcription::test::make_exponential_decay(x0, tf, intervals);
    auto compiled = goss::transcription::HermiteSimpson::compile(
        ocp, "hs_conv_" + std::to_string(intervals));
    goss::solver::IpoptSolver solver;
    solver.set_tolerance(1e-11);  // discretization error must dominate, not solver tol
    std::vector<double> z0(compiled.problem->num_variables(), x0);
    auto result = solver.solve(*compiled.problem, z0);
    if (result.status != goss::solver::SolverStatus::Success) return 1e9;
    const auto& layout = compiled.layout;
    const double h = tf / static_cast<double>(intervals);
    double max_err = 0.0;
    for (std::size_t k = 0; k < layout.num_nodes(); ++k) {
        double xk = result.x[layout.state_index(k, 0)];
        double exact = goss::transcription::test::exp_decay_solution(x0, k * h);
        max_err = std::max(max_err, std::abs(xk - exact));
    }
    return max_err;
}
}  // namespace

TEST(HermiteSimpson, ConvergesAtFourthOrder) {
    // Use coarse meshes so O(h^4) error stays above solver tolerance floor.
    double e1 = hs_max_error(5);
    double e2 = hs_max_error(10);
    double e3 = hs_max_error(20);
    ASSERT_LT(e2, e1);
    ASSERT_LT(e3, e2);
    double order1 = std::log(e1 / e2) / std::log(2.0);
    double order2 = std::log(e2 / e3) / std::log(2.0);
    // HS is O(h^4). Allow slack (3.5) for solver-tolerance contamination at fine meshes.
    EXPECT_GE(order1, 3.5) << "Hermite-Simpson should be ~4th order";
    EXPECT_GE(order2, 3.5) << "Hermite-Simpson should be ~4th order";
}
```

- [ ] **Step 2: Write the scheme-agreement test**

```cpp
// tests/transcription/test_scheme_agreement.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "goss/transcription/trapezoidal.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "transcription/ocp_fixtures.hpp"

namespace {
double final_state(goss::transcription::CompiledOcp& compiled, double guess) {
    goss::solver::IpoptSolver solver;
    std::vector<double> z0(compiled.problem->num_variables(), guess);
    auto result = solver.solve(*compiled.problem, z0);
    EXPECT_EQ(result.status, goss::solver::SolverStatus::Success);
    std::size_t last = compiled.layout.num_nodes() - 1;
    return result.x[compiled.layout.state_index(last, 0)];
}
}  // namespace

TEST(SchemeAgreement, TrapezoidalAndHermiteSimpsonAgreeOnExpDecay) {
    const double x0 = 1.0, tf = 1.0;
    const std::size_t intervals = 80;  // fine enough that both are accurate
    auto ocp_t = goss::transcription::test::make_exponential_decay(x0, tf, intervals);
    auto ocp_h = goss::transcription::test::make_exponential_decay(x0, tf, intervals);
    auto ct = goss::transcription::Trapezoidal::compile(ocp_t, "agree_trap");
    auto ch = goss::transcription::HermiteSimpson::compile(ocp_h, "agree_hs");
    double xt = final_state(ct, x0);
    double xh = final_state(ch, x0);
    double exact = goss::transcription::test::exp_decay_solution(x0, tf);
    EXPECT_NEAR(xt, exact, 1e-3);
    EXPECT_NEAR(xh, exact, 1e-3);
    EXPECT_NEAR(xt, xh, 2e-3);  // both converge to the same analytic solution
}
```

- [ ] **Step 3: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake --build build && ctest --test-dir build -R "HermiteSimpson.ConvergesAtFourthOrder|SchemeAgreement" --output-on-failure'`
Expected: HS order ≥ 3.5 (distinctly steeper than trapezoidal's ~2); agreement test passes. If HS order looks like ~2 not ~4, the midpoint interpolation or Simpson defect is wrong. If the fine-mesh error plateaus (order drops), the solver tolerance floor is being hit — the test already uses coarse meshes (5/10/20) and tol 1e-11 to avoid this.

- [ ] **Step 4: Run the FULL suite**

Run: `scripts/dev.sh 'ctest --test-dir build --output-on-failure'`
Expected: ALL tests pass (44 prior + transcription: layout 4 + ocp 2 + trap 3 + hs 3 + agreement 1 ≈ 57).

- [ ] **Step 5: Commit**

```bash
git add tests/transcription/test_hermite_simpson.cpp tests/transcription/test_scheme_agreement.cpp
git commit -m "test: Hermite-Simpson 4th-order convergence + cross-scheme agreement"
```

---

## Self-Review

**Spec coverage (transcription/ layer portion of the design spec):**
- "Transcription interface + Trapezoidal + HermiteSimpson (multiple schemes behind an interface)" → Task 3 (interface/helpers + OcpProblem), Task 4 (Trapezoidal), Task 6 (HermiteSimpson). The "interface" is realized as templated static `compile` functions + shared `CompiledOcp`/`kInf`/`VariableLayout` (design note explains why runtime polymorphism is deferred — OcpProblem is templated for AD recording). ✓
- "Discretizes time → creates NLP variables (x,u at nodes) + defect constraints" → Task 2 (VariableLayout), Tasks 4/6 (defects). ✓
- "Defect-constraint correctness: known exact trajectory, defects≈0 at analytic solution" → Tasks 4/6 solve tests hit the analytic solution. ✓
- "Convergence order: refine mesh, confirm theoretical rate (trap O(h²), HS O(h⁴))" → Task 5 (trap order≥1.8), Task 7 (HS order≥3.5). THE definitive test. ✓
- "Cross-scheme agreement: both converge to the same solution" → Task 7. ✓
- "Local schemes, highly sparse/banded" → both are local (defects couple only adjacent nodes); compressed HS keeps the same layout. ✓
- "The queue example / states+constraints" → OcpProblem carries state/control bounds + boundary conditions; a bounded state (e.g. queue≥0) is expressible via state_lower. The full DSL is a later layer (correctly out of scope). ✓

**Placeholder scan:** Task 1 creates comment-only src placeholders + trims the test list to only what exists, with later tasks adding their real test files and turning the src files into real TUs — standard staged scaffolding, each resolved in a named task. No "TBD"/"add validation"/"similar to Task N". Every code step has literal content.

**Type consistency:**
- `VariableLayout(num_states, num_controls, num_nodes)` + `state_index/control_index/total_variables/num_nodes` used identically in Tasks 2, 4, 6, and every test.
- `OcpProblem<DynamicsFn,CostFn>` fields (num_states, num_controls, dynamics, cost, mesh, state_lower/upper, control_lower/upper, initial_state/initial_state_fixed, final_state/final_state_fixed) defined in Task 3, used verbatim by both schemes (Tasks 4, 6) and the fixtures.
- `Mesh{t_initial, t_final, num_intervals}` + `num_nodes()`/`interval_width()`/`validate()` consistent.
- `CompiledOcp{ std::unique_ptr<nlp::NLPProblem> problem; VariableLayout layout; }` returned by both schemes' `compile` and consumed identically in all tests (`compiled.problem`, `compiled.layout`, `layout.state_index(node,state)`).
- Both schemes: `compile(const OcpProblem<...>&, const std::string& model_name)` static signature, output packing (cost at 0, then ni*ns defects), constraint bounds [0,0], pinned states via equal var bounds — identical structure, only the interval math differs.
- NLPProblem construction matches the real API on main: `NLPProblem(unique_ptr<ADBackend>, var_lb, var_ub, con_lb, con_ub)` (verified against include/goss/nlp/nlp_problem.hpp).
- `CppADCGBackend(functor, input_size, model_name)` matches the real ctor; the generic-lambda functor satisfies its `template<typename F>` ctor; unique model_name per compile call (tests pass distinct names).

**Known risks flagged in-plan:** convergence tests can be contaminated by solver tolerance at fine meshes — Task 5/7 use appropriate mesh sizes + tightened tolerance and document the failure signature. The compressed-HS decision (no midpoint variables) is stated explicitly so the layout stays identical across schemes. The templated-compile-vs-virtual-interface decision is documented in transcription.hpp with a YAGNI rationale.
