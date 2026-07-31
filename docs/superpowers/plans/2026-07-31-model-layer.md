# Model DSL Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `model/` layer — a user-facing `Model` DSL that lets a user declare a continuous optimal-control problem (named states, controls, dynamics, box bounds, boundary conditions, running cost) and `build()` it into a `transcription::OcpProblem`, realizing the framework's queue example end-to-end.

**Architecture:** `Model` is a RUNTIME metadata builder: `add_state("name")`/`add_control("name")` return typed handles; `set_state_bounds`/`set_control_bounds`/`set_initial_state`/`set_final_state`/`set_mesh` record per-handle metadata. The user's dynamics and running cost are supplied as GENERIC lambdas `(const std::vector<T>& x, const std::vector<T>& u, T t)` — exactly `OcpProblem`'s `DynamicsFn`/`CostFn` contract. A templated `build(dynamics, cost)` validates the metadata and stamps out an `OcpProblem<Dyn,Cost>` (assembling the full-size `state_lower/upper`, `control_lower/upper`, `initial_state`, `initial_state_fixed`, `final_state`, `final_state_fixed` vectors that transcription expects), which the existing `Trapezoidal`/`HermiteSimpson` schemes compile and IPOPT solves. No type erasure — the lambdas pass straight through as `OcpProblem`'s template parameters. This is a v1 lambda DSL; an operator-overload expression AST (spec §7 `q >= 0.0`, `integral(...)`) is a documented future enhancement layered ON TOP without changing this core.

**Tech Stack:** C++17, the merged `goss::ad` + `goss::nlp` + `goss::solver` + `goss::transcription` layers, GoogleTest, CMake, containerized build via `scripts/dev.sh`.

## Global Constraints

- Language: **C++17**.
- `Model::build(dynamics, cost)` produces a `goss::transcription::OcpProblem<DynamicsFn,CostFn>` (fields: `num_states`, `num_controls`, `dynamics`, `cost`, `mesh`, `state_lower`, `state_upper`, `control_lower`, `control_upper`, `initial_state`, `initial_state_fixed`, `final_state`, `final_state_fixed`). The transcription layer's `compile()` requires the bound/state vectors to be exactly size `num_states`/`num_controls` — `build()` MUST produce full-size vectors.
- Dynamics/cost are generic lambdas with the exact signatures: dynamics `template<T> std::vector<T> operator()(const std::vector<T>& x, const std::vector<T>& u, T t)` returning the full `dx/dt` (size `num_states`); cost `template<T> T operator()(const std::vector<T>& x, const std::vector<T>& u, T t)` returning the running (Lagrange) cost. They must be AD-safe (wrap literals in `T(...)`, ADL-friendly `using std::sin;`) — same rule as the ad/transcription layers.
- Free bounds use `goss::transcription::kInf = 2e19` (the framework-wide sentinel).
- **v1 lambda DSL only.** Per-state `set_dynamics(handle, lambda)`, general nonlinear path constraints (beyond box bounds), and the operator-overload expression AST are OUT of scope — documented as extension points, not built. Box bounds on states/controls + boundary conditions + one full-vector dynamics lambda + one cost lambda fully express the queue example.
- The `Model` header MUST carry a documented "Extending to an expression DSL" note: where operator-overloaded handles (`q >= 0.0`, `q.initial() == 10.0`, `integral(...)`) would slot in and what they would lower to (bounds/boundary/cost calls). This is required, not optional.
- Error handling per org standard: a `ModelError` exception for build/usage errors (unset mesh, missing dynamics dimension mismatch, duplicate names, bounds inverted) with meaningful messages naming the offending state/control. No silent catch-all.
- Coding standards: verbose descriptive variable names; type annotations everywhere; comments explain WHY.
- Container-first build: all `cmake`/`ctest` run inside the container via `scripts/dev.sh '<command>'`. Never build on the host.
- Test framework: **GoogleTest**. The queue model + a min-energy control problem (analytic answer) are solved via `IpoptSolver` through the full DSL→transcription→solver chain.

---

## File Structure

- `include/goss/model/errors.hpp` — `ModelError : std::runtime_error`.
- `include/goss/model/handles.hpp` — `StateHandle` / `ControlHandle` (index + implicit `std::size_t` conversion for `x[q]` indexing).
- `include/goss/model/model.hpp` — the `Model` builder: declaration, bounds, boundary conditions, mesh, and the templated `build()`. Header-only (build() is templated on lambda types). Carries the "Extending to an expression DSL" doc note.
- `tests/model/test_model_build.hpp` or fixtures inline — small model fixtures.
- `tests/model/test_model_declaration.cpp` — handles, naming, dimension accounting, duplicate detection.
- `tests/model/test_model_build.cpp` — bounds/boundary/mesh → OcpProblem field assembly + validation errors.
- `tests/model/test_model_solve.cpp` — end-to-end: min-energy control problem (analytic) + queue model, via the DSL → HermiteSimpson → IpoptSolver.
- `CMakeLists.txt` — `goss_model` target (INTERFACE — header-only) + `goss_model_tests` executable.

---

### Task 1: Scaffold model target + ModelError + handles

**Files:**
- Create: `include/goss/model/errors.hpp`
- Create: `include/goss/model/handles.hpp`
- Modify: `CMakeLists.txt`
- Create: `tests/model/test_model_declaration.cpp` (smoke portion only in this task)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `class goss::model::ModelError : public std::runtime_error` (ctor from `const std::string&`).
  - `struct goss::model::StateHandle { std::size_t index; constexpr operator std::size_t() const noexcept { return index; } };`
  - `struct goss::model::ControlHandle { std::size_t index; constexpr operator std::size_t() const noexcept { return index; } };`
  - A `goss_model` INTERFACE library (header-only) that PUBLIC-links `goss_transcription goss_nlp goss_ad`, and a `goss_model_tests` executable wired into ctest.

**Design note:** handles are strong structs (so the future AST can hang operators off them) but implicitly convert to `std::size_t` so `x[q]`/`u[rate]` work inside the lambdas exactly as the spec shows. The v1 limitation (a `StateHandle` can index a control vector since both convert to `size_t`) is acceptable and documented; the future typed-expression AST removes it.

- [ ] **Step 1: Write the failing smoke test**

```cpp
// tests/model/test_model_declaration.cpp
#include <gtest/gtest.h>
#include <cstddef>
#include <vector>
#include "goss/model/errors.hpp"
#include "goss/model/handles.hpp"

TEST(ModelError, IsThrowable) {
    EXPECT_THROW(throw goss::model::ModelError("boom"), goss::model::ModelError);
}

TEST(Handles, ImplicitlyIndexVectors) {
    goss::model::StateHandle q{2};
    goss::model::ControlHandle r{0};
    std::vector<double> x{10.0, 20.0, 30.0};
    std::vector<double> u{5.0};
    EXPECT_DOUBLE_EQ(x[q], 30.0);   // x[2]
    EXPECT_DOUBLE_EQ(u[r], 5.0);    // u[0]
}
```

- [ ] **Step 2: Write the headers**

```cpp
// include/goss/model/errors.hpp
#pragma once
#include <stdexcept>
#include <string>
namespace goss::model {
class ModelError : public std::runtime_error {
 public:
    explicit ModelError(const std::string& message) : std::runtime_error(message) {}
};
}  // namespace goss::model
```

```cpp
// include/goss/model/handles.hpp
#pragma once
#include <cstddef>
namespace goss::model {

/// Opaque handle to a declared state. Implicitly converts to std::size_t so it
/// can index the x-vector inside dynamics/cost lambdas (`x[q]`). It is a struct
/// (not a bare size_t) so a future expression-DSL can attach operators to it
/// (e.g. q >= 0.0, q.initial() == 10.0) without changing call sites.
struct StateHandle {
    std::size_t index;
    constexpr operator std::size_t() const noexcept { return index; }
};

/// Opaque handle to a declared control. See StateHandle.
struct ControlHandle {
    std::size_t index;
    constexpr operator std::size_t() const noexcept { return index; }
};

}  // namespace goss::model
```

- [ ] **Step 3: Wire CMake**

Append to `CMakeLists.txt` (after the `goss_transcription_tests` block):

```cmake
# ---- Model DSL layer ----
add_library(goss_model INTERFACE)
target_include_directories(goss_model INTERFACE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(goss_model INTERFACE goss_transcription goss_nlp goss_ad)

add_executable(goss_model_tests
  tests/model/test_model_declaration.cpp)
target_include_directories(goss_model_tests PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(goss_model_tests PRIVATE
  goss_model goss_transcription goss_nlp goss_ad goss_ad_impl goss_solver
  goss_ipopt_iface goss_nlopt_iface cppadcg
  $<$<BOOL:${CPPAD_LIB}>:${CPPAD_LIB}> GTest::gtest_main)
gtest_discover_tests(goss_model_tests)
```

Note: `test_model_build.cpp` and `test_model_solve.cpp` are created in later tasks and added to the `add_executable` list then — for THIS task list only `tests/model/test_model_declaration.cpp`.

- [ ] **Step 4: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake -S . -B build && cmake --build build && ctest --test-dir build -R "ModelError|Handles" --output-on-failure'`
Expected: both tests PASS. Confirm no regression: `scripts/dev.sh 'ctest --test-dir build --output-on-failure'` (61 prior + 2 new = 63).

- [ ] **Step 5: Commit**

```bash
git add include/goss/model/ tests/model/ CMakeLists.txt
git commit -m "build: scaffold model DSL layer with ModelError and handles"
```

---

### Task 2: Model state/control declaration

**Files:**
- Create: `include/goss/model/model.hpp`
- Modify: `tests/model/test_model_declaration.cpp`

**Interfaces:**
- Consumes: `ModelError`, `StateHandle`, `ControlHandle`.
- Produces: `class goss::model::Model` (header-only) with:
  - `StateHandle add_state(const std::string& name);` — appends a state, returns a handle with the next index. Throws `ModelError` on duplicate name (across states AND controls).
  - `ControlHandle add_control(const std::string& name);` — analogous.
  - `std::size_t num_states() const;` `std::size_t num_controls() const;`
  - `const std::string& state_name(std::size_t index) const;` `const std::string& control_name(std::size_t index) const;` (throw on out-of-range).
  - Internally stores `std::vector<std::string> state_names_`, `control_names_`, and per-state/per-control bound + boundary metadata (added in Task 3 — declare the members now with sensible defaults so Task 3 only fills logic). Declaring a state/control also pushes default metadata: bounds [-kInf, +kInf], not-fixed boundary.

- [ ] **Step 1: Write the failing tests**

```cpp
// append to tests/model/test_model_declaration.cpp
#include "goss/model/model.hpp"

TEST(ModelDeclaration, AssignsSequentialIndices) {
    goss::model::Model model;
    auto q = model.add_state("queue_length");
    auto x2 = model.add_state("second");
    auto rate = model.add_control("service_rate");
    EXPECT_EQ(q.index, 0u);
    EXPECT_EQ(x2.index, 1u);
    EXPECT_EQ(rate.index, 0u);          // controls indexed independently
    EXPECT_EQ(model.num_states(), 2u);
    EXPECT_EQ(model.num_controls(), 1u);
    EXPECT_EQ(model.state_name(0), "queue_length");
    EXPECT_EQ(model.control_name(0), "service_rate");
}

TEST(ModelDeclaration, RejectsDuplicateNames) {
    goss::model::Model model;
    model.add_state("x");
    EXPECT_THROW(model.add_state("x"), goss::model::ModelError);
    EXPECT_THROW(model.add_control("x"), goss::model::ModelError);  // clash across kinds
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `scripts/dev.sh 'cmake --build build 2>&1 | tail -20'`
Expected: FAIL — model.hpp not found.

- [ ] **Step 3: Write model.hpp (declaration portion + members for later tasks)**

```cpp
// include/goss/model/model.hpp
#pragma once
#include <cstddef>
#include <string>
#include <vector>
#include "goss/model/errors.hpp"
#include "goss/model/handles.hpp"
#include "goss/transcription/ocp_problem.hpp"
#include "goss/transcription/transcription.hpp"  // kInf

namespace goss::model {

/// Runtime builder for a continuous optimal-control problem.
///
/// Usage (lambda DSL, v1):
///   Model model;
///   auto q    = model.add_state("queue_length");
///   auto rate = model.add_control("service_rate");
///   model.set_state_bounds(q, 0.0, kInf);       // q >= 0
///   model.set_control_bounds(rate, 0.0, MAX);   // 0 <= rate <= MAX
///   model.set_initial_state(q, 10.0);           // q(0) = 10
///   model.set_mesh(0.0, T, num_intervals);
///   auto ocp = model.build(dynamics_lambda, cost_lambda);
///   auto compiled = transcription::HermiteSimpson::compile(ocp, "queue");
///
/// EXTENDING TO AN EXPRESSION DSL (future, not v1):
///   The spec's operator-overloaded syntax (`q >= 0.0`, `q.initial() == 10.0`,
///   `set_cost(integral(q + w*rate*rate))`) can be layered ON TOP of this core
///   without changing it. Add operators to StateHandle/ControlHandle that build
///   small expression nodes; a path-constraint expression like `q >= 0.0`
///   lowers to set_state_bounds(q, 0.0, kInf); a boundary expression
///   `q.initial() == 10.0` lowers to set_initial_state(q, 10.0); an
///   `integral(expr)` cost lowers to a cost lambda that evaluates the expr AST
///   under the templated scalar T. The AST evaluator must be templated on T
///   (so it records under CppAD AD types), exactly like the lambdas here. The
///   Model's build() and all downstream layers stay unchanged — the AST is a
///   front-end that produces the same lambdas/bounds this API takes directly.
class Model {
 public:
    StateHandle add_state(const std::string& name) {
        ensure_unique_name(name);
        const std::size_t index = state_names_.size();
        state_names_.push_back(name);
        state_lower_.push_back(-transcription::kInf);
        state_upper_.push_back(transcription::kInf);
        initial_value_.push_back(0.0);
        initial_fixed_.push_back(false);
        final_value_.push_back(0.0);
        final_fixed_.push_back(false);
        return StateHandle{index};
    }

    ControlHandle add_control(const std::string& name) {
        ensure_unique_name(name);
        const std::size_t index = control_names_.size();
        control_names_.push_back(name);
        control_lower_.push_back(-transcription::kInf);
        control_upper_.push_back(transcription::kInf);
        return ControlHandle{index};
    }

    std::size_t num_states() const { return state_names_.size(); }
    std::size_t num_controls() const { return control_names_.size(); }

    const std::string& state_name(std::size_t index) const {
        if (index >= state_names_.size()) throw ModelError("Model::state_name: index out of range");
        return state_names_[index];
    }
    const std::string& control_name(std::size_t index) const {
        if (index >= control_names_.size()) throw ModelError("Model::control_name: index out of range");
        return control_names_[index];
    }

    // Bounds / boundary / mesh setters added in Task 3-4.
    // build() added in Task 4.

 private:
    void ensure_unique_name(const std::string& name) const {
        for (const auto& existing : state_names_)
            if (existing == name) throw ModelError("Model: duplicate name '" + name + "' (already a state)");
        for (const auto& existing : control_names_)
            if (existing == name) throw ModelError("Model: duplicate name '" + name + "' (already a control)");
    }

    std::vector<std::string> state_names_;
    std::vector<std::string> control_names_;
    std::vector<double> state_lower_;
    std::vector<double> state_upper_;
    std::vector<double> control_lower_;
    std::vector<double> control_upper_;
    std::vector<double> initial_value_;
    std::vector<bool> initial_fixed_;
    std::vector<double> final_value_;
    std::vector<bool> final_fixed_;

    bool mesh_set_ = false;
    transcription::Mesh mesh_{0.0, 1.0, 1};
};

}  // namespace goss::model
```

- [ ] **Step 4: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake --build build && ctest --test-dir build -R "ModelDeclaration" --output-on-failure'`
Expected: both ModelDeclaration tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/goss/model/model.hpp tests/model/test_model_declaration.cpp
git commit -m "feat: Model state/control declaration with named handles"
```

---
### Task 3: Bounds, boundary conditions, and mesh setters

**Files:**
- Modify: `include/goss/model/model.hpp`
- Modify: `tests/model/test_model_declaration.cpp` (or a new test file — use test_model_build.cpp; see note)

**Interfaces:**
- Consumes: `Model` (Task 2), `StateHandle`/`ControlHandle`, `transcription::Mesh`, `transcription::kInf`.
- Produces (all return `void`, all validate and throw `ModelError`):
  - `void set_state_bounds(StateHandle s, double lower, double upper);` — throws if `s.index` out of range or `lower > upper`.
  - `void set_control_bounds(ControlHandle c, double lower, double upper);` — analogous.
  - `void set_initial_state(StateHandle s, double value);` — pins state s at node 0 (sets initial_value_ + initial_fixed_=true).
  - `void set_final_state(StateHandle s, double value);` — pins state s at the last node.
  - `void set_mesh(double t_initial, double t_final, std::size_t num_intervals);` — stores a `transcription::Mesh`, sets mesh_set_=true. (Validation of the mesh itself happens in build via Mesh::validate.)
  - Read-back accessors used by tests: `double state_lower(std::size_t) const;` `double state_upper(std::size_t) const;` `double control_lower(std::size_t) const;` `double control_upper(std::size_t) const;` `bool initial_fixed(std::size_t) const;` `double initial_value(std::size_t) const;` `bool final_fixed(std::size_t) const;` `double final_value(std::size_t) const;` (each throws on out-of-range).

**Note:** create `tests/model/test_model_build.cpp` in this task for the setter/read-back tests, and ADD it to the `goss_model_tests` add_executable in CMakeLists.txt.

- [ ] **Step 1: Write the failing tests**

```cpp
// tests/model/test_model_build.cpp
#include <gtest/gtest.h>
#include "goss/model/model.hpp"
#include "goss/transcription/transcription.hpp"

TEST(ModelSetters, RecordsStateAndControlBounds) {
    goss::model::Model model;
    auto q = model.add_state("q");
    auto r = model.add_control("r");
    model.set_state_bounds(q, 0.0, goss::transcription::kInf);
    model.set_control_bounds(r, -2.0, 2.0);
    EXPECT_DOUBLE_EQ(model.state_lower(0), 0.0);
    EXPECT_DOUBLE_EQ(model.state_upper(0), goss::transcription::kInf);
    EXPECT_DOUBLE_EQ(model.control_lower(0), -2.0);
    EXPECT_DOUBLE_EQ(model.control_upper(0), 2.0);
}

TEST(ModelSetters, RecordsBoundaryConditions) {
    goss::model::Model model;
    auto q = model.add_state("q");
    model.set_initial_state(q, 10.0);
    EXPECT_TRUE(model.initial_fixed(0));
    EXPECT_DOUBLE_EQ(model.initial_value(0), 10.0);
    EXPECT_FALSE(model.final_fixed(0));   // not set → free
}

TEST(ModelSetters, RejectsInvertedBounds) {
    goss::model::Model model;
    auto q = model.add_state("q");
    EXPECT_THROW(model.set_state_bounds(q, 5.0, -5.0), goss::model::ModelError);
}

TEST(ModelSetters, RejectsOutOfRangeHandle) {
    goss::model::Model model;
    model.add_state("q");
    goss::model::StateHandle bogus{7};
    EXPECT_THROW(model.set_state_bounds(bogus, 0.0, 1.0), goss::model::ModelError);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `scripts/dev.sh 'cmake --build build 2>&1 | tail -20'` (after adding test_model_build.cpp to CMake)
Expected: FAIL — setters not declared.

- [ ] **Step 3: Add setters + accessors to model.hpp**

Insert into `Model`'s public section (replacing the `// Bounds / boundary / mesh setters added in Task 3-4.` comment):

```cpp
    void set_state_bounds(StateHandle s, double lower, double upper) {
        check_state_index(s.index, "set_state_bounds");
        if (lower > upper) throw ModelError("set_state_bounds: lower > upper for state '" + state_names_[s.index] + "'");
        state_lower_[s.index] = lower;
        state_upper_[s.index] = upper;
    }
    void set_control_bounds(ControlHandle c, double lower, double upper) {
        check_control_index(c.index, "set_control_bounds");
        if (lower > upper) throw ModelError("set_control_bounds: lower > upper for control '" + control_names_[c.index] + "'");
        control_lower_[c.index] = lower;
        control_upper_[c.index] = upper;
    }
    void set_initial_state(StateHandle s, double value) {
        check_state_index(s.index, "set_initial_state");
        initial_value_[s.index] = value;
        initial_fixed_[s.index] = true;
    }
    void set_final_state(StateHandle s, double value) {
        check_state_index(s.index, "set_final_state");
        final_value_[s.index] = value;
        final_fixed_[s.index] = true;
    }
    void set_mesh(double t_initial, double t_final, std::size_t num_intervals) {
        mesh_ = transcription::Mesh{t_initial, t_final, num_intervals};
        mesh_set_ = true;
    }

    double state_lower(std::size_t i) const { check_state_index(i, "state_lower"); return state_lower_[i]; }
    double state_upper(std::size_t i) const { check_state_index(i, "state_upper"); return state_upper_[i]; }
    double control_lower(std::size_t i) const { check_control_index(i, "control_lower"); return control_lower_[i]; }
    double control_upper(std::size_t i) const { check_control_index(i, "control_upper"); return control_upper_[i]; }
    bool initial_fixed(std::size_t i) const { check_state_index(i, "initial_fixed"); return initial_fixed_[i]; }
    double initial_value(std::size_t i) const { check_state_index(i, "initial_value"); return initial_value_[i]; }
    bool final_fixed(std::size_t i) const { check_state_index(i, "final_fixed"); return final_fixed_[i]; }
    double final_value(std::size_t i) const { check_state_index(i, "final_value"); return final_value_[i]; }
```

Add to the private section:

```cpp
    void check_state_index(std::size_t i, const char* who) const {
        if (i >= state_names_.size())
            throw ModelError(std::string(who) + ": state index out of range");
    }
    void check_control_index(std::size_t i, const char* who) const {
        if (i >= control_names_.size())
            throw ModelError(std::string(who) + ": control index out of range");
    }
```

- [ ] **Step 4: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake --build build && ctest --test-dir build -R "ModelSetters" --output-on-failure'`
Expected: all 4 ModelSetters tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/goss/model/model.hpp tests/model/test_model_build.cpp CMakeLists.txt
git commit -m "feat: Model bounds, boundary condition, and mesh setters"
```

---

### Task 4: Model::build() — assemble the OcpProblem

**Files:**
- Modify: `include/goss/model/model.hpp`
- Modify: `tests/model/test_model_build.cpp`

**Interfaces:**
- Consumes: all of `Model`'s metadata, `transcription::OcpProblem`, `transcription::Mesh`.
- Produces:
  - `template <typename DynamicsFn, typename CostFn> transcription::OcpProblem<DynamicsFn, CostFn> build(DynamicsFn dynamics, CostFn cost) const;`
  - Behavior: validates the model is buildable, then constructs and returns an `OcpProblem<DynamicsFn,CostFn>` with:
    - `num_states = num_states()`, `num_controls = num_controls()`.
    - `dynamics = std::move(dynamics)`, `cost = std::move(cost)`.
    - `mesh = mesh_`.
    - `state_lower = state_lower_`, `state_upper = state_upper_` (copies of the full-size vectors).
    - `control_lower = control_lower_`, `control_upper = control_upper_`.
    - `initial_state = initial_value_`, `final_state = final_value_`.
    - `initial_state_fixed`: a `std::vector<double>` size num_states where entry i = `initial_fixed_[i] ? 1.0 : 0.0` (transcription's `*_fixed` convention is "nonzero => pinned").
    - `final_state_fixed`: analogous from `final_fixed_`.
  - Validation (throw `ModelError`): `num_states() >= 1` ("a model needs at least one state"); `mesh_set_ == true` ("call set_mesh before build"). (The mesh's own validity — intervals>=1, tf>t0 — is checked by the transcription `compile()` via `Mesh::validate()`, but calling `mesh_.validate()` here too gives an earlier, model-level error; do call it.)

**Design note:** `build()` is `const` and templated on the lambda types — the lambdas are moved into the returned `OcpProblem`, whose type is `OcpProblem<DynamicsFn,CostFn>`. This is why there is no type erasure: the concrete lambda types flow straight into `OcpProblem`'s template parameters, and from there into `Trapezoidal::compile`/`HermiteSimpson::compile` (also templated). `std::vector<bool>` → `std::vector<double>` conversion for the `*_fixed` vectors must be an explicit loop (can't copy-assign different types).

- [ ] **Step 1: Write the failing tests**

```cpp
// append to tests/model/test_model_build.cpp
#include "goss/transcription/ocp_problem.hpp"

namespace {
struct DummyDyn {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x, const std::vector<T>& /*u*/, T /*t*/) const {
        return { -x[0] };
    }
};
struct DummyCost {
    template <typename T>
    T operator()(const std::vector<T>&, const std::vector<T>&, T) const { return T(0); }
};
}  // namespace

TEST(ModelBuild, AssemblesOcpProblemFields) {
    goss::model::Model model;
    auto q = model.add_state("q");
    auto r = model.add_control("r");
    model.set_state_bounds(q, 0.0, goss::transcription::kInf);
    model.set_control_bounds(r, -1.0, 1.0);
    model.set_initial_state(q, 10.0);
    model.set_mesh(0.0, 2.0, 5);

    auto ocp = model.build(DummyDyn{}, DummyCost{});
    EXPECT_EQ(ocp.num_states, 1u);
    EXPECT_EQ(ocp.num_controls, 1u);
    ASSERT_EQ(ocp.state_lower.size(), 1u);
    EXPECT_DOUBLE_EQ(ocp.state_lower[0], 0.0);
    ASSERT_EQ(ocp.control_upper.size(), 1u);
    EXPECT_DOUBLE_EQ(ocp.control_upper[0], 1.0);
    ASSERT_EQ(ocp.initial_state_fixed.size(), 1u);
    EXPECT_DOUBLE_EQ(ocp.initial_state_fixed[0], 1.0);   // pinned
    EXPECT_DOUBLE_EQ(ocp.initial_state[0], 10.0);
    ASSERT_EQ(ocp.final_state_fixed.size(), 1u);
    EXPECT_DOUBLE_EQ(ocp.final_state_fixed[0], 0.0);      // free
    EXPECT_EQ(ocp.mesh.num_intervals, 5u);
}

TEST(ModelBuild, RejectsBuildWithoutMesh) {
    goss::model::Model model;
    model.add_state("q");
    EXPECT_THROW(model.build(DummyDyn{}, DummyCost{}), goss::model::ModelError);
}

TEST(ModelBuild, RejectsBuildWithNoStates) {
    goss::model::Model model;
    model.set_mesh(0.0, 1.0, 4);
    EXPECT_THROW(model.build(DummyDyn{}, DummyCost{}), goss::model::ModelError);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `scripts/dev.sh 'cmake --build build 2>&1 | tail -20'`
Expected: FAIL — build() not declared.

- [ ] **Step 3: Implement build() in model.hpp**

Add to `Model`'s public section (replacing the `// build() added in Task 4.` comment):

```cpp
    template <typename DynamicsFn, typename CostFn>
    transcription::OcpProblem<DynamicsFn, CostFn> build(DynamicsFn dynamics, CostFn cost) const {
        if (state_names_.empty()) throw ModelError("Model::build: a model needs at least one state");
        if (!mesh_set_) throw ModelError("Model::build: call set_mesh() before build()");
        mesh_.validate();  // early model-level mesh check

        transcription::OcpProblem<DynamicsFn, CostFn> ocp;
        ocp.num_states = state_names_.size();
        ocp.num_controls = control_names_.size();
        ocp.dynamics = std::move(dynamics);
        ocp.cost = std::move(cost);
        ocp.mesh = mesh_;
        ocp.state_lower = state_lower_;
        ocp.state_upper = state_upper_;
        ocp.control_lower = control_lower_;
        ocp.control_upper = control_upper_;
        ocp.initial_state = initial_value_;
        ocp.final_state = final_value_;
        ocp.initial_state_fixed.assign(state_names_.size(), 0.0);
        ocp.final_state_fixed.assign(state_names_.size(), 0.0);
        for (std::size_t i = 0; i < state_names_.size(); ++i) {
            ocp.initial_state_fixed[i] = initial_fixed_[i] ? 1.0 : 0.0;
            ocp.final_state_fixed[i] = final_fixed_[i] ? 1.0 : 0.0;
        }
        return ocp;
    }
```

Add `#include <utility>` (for std::move) to model.hpp's includes.

- [ ] **Step 4: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake --build build && ctest --test-dir build -R "ModelBuild" --output-on-failure'`
Expected: all 3 ModelBuild tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/goss/model/model.hpp tests/model/test_model_build.cpp
git commit -m "feat: Model::build assembles OcpProblem from declared metadata"
```

---

### Task 5: End-to-end DSL solves — min-energy control + the queue model

**Files:**
- Create: `tests/model/test_model_solve.cpp`
- Modify: `CMakeLists.txt` (add test_model_solve.cpp)

**Interfaces:**
- Consumes: full `Model` API, `transcription::HermiteSimpson`, `solver::IpoptSolver`.
- Produces: two end-to-end tests through DSL → HermiteSimpson → IpoptSolver, including the framework's flagship queue example. This is also the layer's first `num_controls > 0` end-to-end solve (closing the transcription layer's deferred gap).

**Test 1 — min-energy double integrator (analytic-ish, sanity):** state x (position), control u (acceleration-like), dynamics dx/dt = u, cost integral of u². With x(0)=0, x(T)=1 fixed and u free, the min-energy control driving x from 0 to 1 is constant u = 1/T (since dx/dt=u, x(T)=∫u dt = u·T = 1 ⇒ u=1/T), cost = ∫u² dt = (1/T²)·T = 1/T. Check the final state hits 1 and the solve succeeds. (This exercises a control end-to-end.)

**Test 2 — the queue model (spec flagship):** state q (queue length), control rate (service rate), dynamics dq/dt = ARRIVAL - rate, q>=0, 0<=rate<=MAX, q(0)=10, cost integral of (q + w·rate²). Assert: solve succeeds, q stays >= 0 (approximately, within a small tolerance at nodes), q(0)==10, and the objective is finite/positive. This is a qualitative smoke test of the flagship example, not an analytic-optimum check.

- [ ] **Step 1: Write the min-energy test**

```cpp
// tests/model/test_model_solve.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "goss/model/model.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"

TEST(ModelSolve, MinEnergyDoubleIntegratorReachesTarget) {
    const double T = 2.0;
    const std::size_t intervals = 20;
    goss::model::Model model;
    auto x = model.add_state("position");
    auto u = model.add_control("accel");
    model.set_control_bounds(u, -10.0, 10.0);
    model.set_initial_state(x, 0.0);
    model.set_final_state(x, 1.0);
    model.set_mesh(0.0, T, intervals);

    // dynamics dx/dt = u ; running cost u^2.
    auto dynamics = [](const auto& xx, const auto& uu, auto /*t*/) {
        using T2 = typename std::decay_t<decltype(xx)>::value_type;
        return std::vector<T2>{ uu[0] };
    };
    auto cost = [](const auto& /*xx*/, const auto& uu, auto /*t*/) {
        return uu[0] * uu[0];
    };

    auto ocp = model.build(dynamics, cost);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "model_minenergy");
    goss::solver::IpoptSolver solver;
    std::vector<double> z0(compiled.problem->num_variables(), 0.5);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);

    // final position must be 1.0 (pinned); objective ≈ 1/T = 0.5 for the analytic min-energy control.
    std::size_t last = compiled.layout.num_nodes() - 1;
    double x_final = result.x[compiled.layout.state_index(last, 0)];
    EXPECT_NEAR(x_final, 1.0, 1e-6);
    EXPECT_NEAR(result.objective_value, 1.0 / T, 1e-3);
}
```

- [ ] **Step 2: Add test to CMake, build and run — verify pass**

Add `tests/model/test_model_solve.cpp` to the `goss_model_tests` add_executable.
Run: `scripts/dev.sh 'cmake -S . -B build && cmake --build build && ctest --test-dir build -R "ModelSolve.MinEnergyDoubleIntegratorReachesTarget" --output-on-failure'`
Expected: PASS — x_final ≈ 1.0, objective ≈ 0.5. If objective is off, the analytic min-energy value for dx=u, x:0→1 over [0,T] is 1/T; verify T. If the solve fails, check the control bounds are wide enough (±10) and the initial guess.

- [ ] **Step 3: Write the queue model test (flagship)**

```cpp
// append to tests/model/test_model_solve.cpp
TEST(ModelSolve, QueueModelKeepsQueueNonNegative) {
    const double ARRIVAL = 3.0, MAX_RATE = 5.0, WEIGHT = 0.1, T = 5.0;
    const std::size_t intervals = 30;
    goss::model::Model model;
    auto q = model.add_state("queue_length");
    auto rate = model.add_control("service_rate");
    model.set_state_bounds(q, 0.0, goss::transcription::kInf);   // q >= 0
    model.set_control_bounds(rate, 0.0, MAX_RATE);               // 0 <= rate <= MAX
    model.set_initial_state(q, 10.0);                            // q(0) = 10
    model.set_mesh(0.0, T, intervals);

    // dq/dt = ARRIVAL - rate ; running cost q + WEIGHT*rate^2.
    auto dynamics = [ARRIVAL](const auto& xx, const auto& uu, auto /*t*/) {
        using T2 = typename std::decay_t<decltype(xx)>::value_type;
        return std::vector<T2>{ T2(ARRIVAL) - uu[0] };
    };
    auto cost = [WEIGHT](const auto& xx, const auto& uu, auto /*t*/) {
        using T2 = typename std::decay_t<decltype(xx)>::value_type;
        return xx[0] + T2(WEIGHT) * uu[0] * uu[0];
    };

    auto ocp = model.build(dynamics, cost);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "model_queue");
    goss::solver::IpoptSolver solver;
    std::vector<double> z0(compiled.problem->num_variables(), 5.0);
    auto result = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);

    const auto& layout = compiled.layout;
    // q(0) pinned to 10.
    EXPECT_NEAR(result.x[layout.state_index(0, 0)], 10.0, 1e-6);
    // q stays >= 0 at every node (allow tiny solver slack).
    for (std::size_t k = 0; k < layout.num_nodes(); ++k) {
        EXPECT_GE(result.x[layout.state_index(k, 0)], -1e-4) << "queue negative at node " << k;
    }
    // rate respected its box [0, MAX].
    for (std::size_t k = 0; k < layout.num_nodes(); ++k) {
        double r = result.x[layout.control_index(k, 0)];
        EXPECT_GE(r, -1e-4);
        EXPECT_LE(r, MAX_RATE + 1e-4);
    }
    EXPECT_GT(result.objective_value, 0.0);
}
```

- [ ] **Step 4: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake --build build && ctest --test-dir build -R "ModelSolve" --output-on-failure'`
Expected: both ModelSolve tests PASS. The queue solve should keep q>=0 (the state bound) and rate in [0,MAX]. If IPOPT reports infeasible, the horizon/arrival/rate combo may be over-constrained — MAX_RATE=5 > ARRIVAL=3 so the queue is drainable; q(0)=10 with T=5 is feasible.

- [ ] **Step 5: Run the FULL suite**

Run: `scripts/dev.sh 'ctest --test-dir build --output-on-failure'`
Expected: ALL tests pass (61 prior + model: declaration 4 + build/setters 7 + solve 2 = ~74).

- [ ] **Step 6: Commit**

```bash
git add tests/model/test_model_solve.cpp CMakeLists.txt
git commit -m "test: DSL solves min-energy control and the flagship queue model"
```

---

## Self-Review

**Spec coverage (model/ layer + spec §7 Option A queue example):**
- "Modeling DSL: states, controls, dynamics, path/boundary constraints, cost → Problem" → Task 2 (states/controls), Task 3 (bounds = box path constraints + boundary conditions + mesh), Task 4 (dynamics + cost via build). ✓ (General nonlinear path constraints beyond box bounds are documented out-of-scope for v1.)
- "The queue example as a permanent integration test" → Task 5 Test 2 (flagship queue: named state/control, dynamics, q>=0 bound, rate box, q(0)=10, integral cost). ✓
- "declare states/controls; pick a scheme + solver; solve" (spec §7 Option A) → Task 5 runs Model → HermiteSimpson → IpoptSolver end-to-end. ✓
- "operator-overload syntax (q>=0.0, integral(...))" → documented as a future extension point in model.hpp with where/what-it-lowers-to (per user decision: lambda now, AST later with docs). ✓
- "closes the nc>0 end-to-end gap deferred from transcription" → Task 5 both tests have controls. ✓
- Assembly/validation (dimensions, duplicate names, inverted bounds, unset mesh) → Tasks 2-4. ✓

**Placeholder scan:** Task 1 lists only test_model_declaration.cpp in CMake; later tasks add test_model_build.cpp (Task 3) and test_model_solve.cpp (Task 5) — standard staged test wiring, each in a named task. Model member fields for later tasks are declared in Task 2 with defaults and filled with logic in Task 3/4 (comments mark the seams). No "TBD"/"add validation"/"similar to Task N"; every code step has literal content.

**Type consistency:**
- `StateHandle{index}`/`ControlHandle{index}` with `operator std::size_t` used in handles, all setters, and the lambda indexing (`x[q]`) consistently.
- `Model` members (state_names_, control_names_, state_lower_/upper_, control_lower_/upper_, initial_value_/initial_fixed_, final_value_/final_fixed_, mesh_/mesh_set_) declared in Task 2, filled by Task 3 setters, read by Task 4 build.
- `build<DynamicsFn,CostFn>(dynamics, cost) const → OcpProblem<DynamicsFn,CostFn>` signature consistent between Task 4 decl/impl and Task 5 call sites.
- OcpProblem field names (num_states, num_controls, dynamics, cost, mesh, state_lower, state_upper, control_lower, control_upper, initial_state, initial_state_fixed, final_state, final_state_fixed) match the real transcription/ocp_problem.hpp on main (verified). The `*_fixed` "nonzero => pinned" convention (double 1.0/0.0) matches transcription's compile() guard usage.
- Mesh{t_initial, t_final, num_intervals} + validate() matches transcription's Mesh.
- HermiteSimpson::compile(ocp, name) → CompiledOcp{problem, layout}; layout.state_index/control_index/num_nodes and problem->num_variables()/solve via IpoptSolver — all match the merged transcription + solver APIs.
- The generic-lambda dynamics/cost use `using T2 = typename std::decay_t<decltype(xx)>::value_type;` and wrap literals in T2(...) for AD safety, matching the transcription fixtures' proven pattern.

**Known risks flagged in-plan:** the min-energy analytic objective (1/T) assumes the exact continuous optimum; HS at 20 intervals should hit it within 1e-3, but if discretization/solver slack causes a miss, the tolerance note in Task 5 Step 2 explains the expected value derivation. The queue test is deliberately qualitative (feasibility + bounds + pinned initial), not an analytic-optimum assertion, since its closed-form optimum is not trivial. std::vector<bool>→double needs an explicit loop (flagged in Task 4).
