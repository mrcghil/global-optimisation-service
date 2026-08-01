# Composition Quick-Wins: Multi-State-Owner + Multi-Derived Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

---

## Goal

Lift two documented v1 restrictions in `include/goss/model/composed_model.hpp` (model layer only — no transcription, NLP, solver, or AD layer changes):

- **(A) Multi-state-owning components:** Remove the `prepare_build()` guard at ~line 497–501 that throws when more than one component owns states. Allow N components each owning states. Assemble their per-component dynamics into ONE combined generic-lambda dynamics functor over the global state vector, writing each component's `dx` into its contiguous global slot via `component_state_offsets`.

- **(B) Multiple derived quantities (2+):** Replace the fixed-arity 0-derived and 1-derived `build()` overloads with a single variadic `build()` accepting any number of derived-expression lambdas + any number of component dynamics lambdas + one cost lambda. Derived quantities are evaluated in topological order using the `topo_ordered_deriveds_` structure that `resolve_names()` already populates (including `dependency_global_derived_indices`). Both restrictions are lifted together in a single new `build()` overload that supersedes the two existing overloads.

---

## Architecture

### Variadic-Assembly Approach: (a) — variadic-template `build()` accumulating a `std::tuple`

The new `build()` is:

```cpp
template <typename... DerivedExprFns, typename... DynFns, typename CostFn>
auto build(
    std::tuple<DerivedExprFns...> derived_expr_tuple,
    std::tuple<DynFns...>         component_dyn_tuple,
    CostFn                        combined_cost);
```

called via two helper free functions:

```cpp
template <typename... DerivedExprFns>
auto make_derived_exprs(DerivedExprFns... fns) {
    return std::make_tuple(std::forward<DerivedExprFns>(fns)...);
}

template <typename... DynFns>
auto make_component_dyns(DynFns... fns) {
    return std::make_tuple(std::forward<DynFns>(fns)...);
}
```

**Justification:** this mirrors the `ExprModel::DynamicsFunctor<DynTuple>` pattern already merged in `include/goss/model/expr/expr_model.hpp`, which solves an identical tuple-accumulation AD-safety problem. It avoids type erasure entirely: the full tuple type is fixed at the `build()` call site, so the combined functor's `operator()` is a function template instantiated concretely at CppAD recording time. Alternative (b) — a heterogeneous `build(Dyn0, ..., DynN, Deriveds...)` with parameter-pack deduction — requires complex tag-based disambiguation between derived-expr and dynamics packs that is fragile in C++17 without concepts; the tuple helper approach is cleaner and already proven in this codebase.

### Combined Functor Design

A `ComposedDynamicsFunctor<DerivedTuple, DynTuple>` struct (analogous to `DynamicsFunctor<DynTuple>` in `expr_model.hpp`) is defined inside `composed_model.hpp`:

```cpp
template <typename DerivedTuple, typename DynTuple>
struct ComposedDynamicsFunctor {
    DerivedTuple                derived_expr_tuple;
    DynTuple                    component_dyn_tuple;
    std::size_t                 num_states;
    std::vector<std::size_t>    component_state_offsets;
    std::vector<std::size_t>    component_num_owned_states;
    // dependency_indices[i] = list of topo-global derived indices that derived i depends on
    std::vector<std::vector<std::size_t>> derived_dependency_indices;

    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x,
                               const std::vector<T>& u,
                               T                     t) const {
        const std::size_t num_deriveds = std::tuple_size_v<DerivedTuple>;
        std::vector<T> deriveds(num_deriveds);
        // Step 1: evaluate derived quantities in topological order.
        // derived_dependency_indices[i] contains the global topo indices of all
        // quantities this entry depends on — they are guaranteed to already be
        // filled in deriveds[] because topo order ensures deps come first.
        evaluate_deriveds(deriveds, x, u, t,
                          std::make_index_sequence<std::tuple_size_v<DerivedTuple>>{});
        // Step 2: evaluate per-component dynamics and write into global dx slots.
        std::vector<T> dx(num_states, T(0));
        evaluate_component_dyns(dx, x, u, deriveds, t,
                                std::make_index_sequence<std::tuple_size_v<DynTuple>>{});
        return dx;
    }

 private:
    template <typename T, std::size_t... Idxs>
    void evaluate_deriveds(std::vector<T>&       deriveds,
                           const std::vector<T>& x,
                           const std::vector<T>& u,
                           T                     t,
                           std::index_sequence<Idxs...>) const {
        // Build deriveds_so_far slice per entry using dependency_indices[I].
        // The comma-fold evaluates entries in ascending index order, which is
        // exactly topological order because topo_ordered_deriveds_ preserves it.
        ((deriveds[Idxs] = [&]() {
            std::vector<T> deps_so_far;
            deps_so_far.reserve(derived_dependency_indices[Idxs].size());
            for (const std::size_t dep_idx : derived_dependency_indices[Idxs]) {
                deps_so_far.push_back(deriveds[dep_idx]);
            }
            return std::get<Idxs>(derived_expr_tuple)(x, u, deps_so_far, t);
        }()), ...);
    }

    template <typename T, std::size_t... Idxs>
    void evaluate_component_dyns(std::vector<T>&       dx,
                                  const std::vector<T>& x,
                                  const std::vector<T>& u,
                                  const std::vector<T>& deriveds,
                                  T                     t,
                                  std::index_sequence<Idxs...>) const {
        ((([&]() {
            auto comp_dx = std::get<Idxs>(component_dyn_tuple)(x, u, deriveds, t);
            const std::size_t offset   = component_state_offsets[Idxs];
            const std::size_t n_states = component_num_owned_states[Idxs];
            for (std::size_t k = 0; k < n_states; ++k) {
                dx[offset + k] = comp_dx[k];
            }
        })()), ...);
    }
};
```

A `ComposedCostFunctor<DerivedTuple, CostFn>` mirrors the same `evaluate_deriveds` logic and wraps `combined_cost(x, u, deriveds, t)`.

### Backward Compatibility

The existing 0-derived and 1-derived `build()` overloads are **replaced** by the new variadic `build()`. Existing call sites migrate by wrapping arguments in `make_derived_exprs(...)` / `make_component_dyns(...)` helpers:

| Old call | New call |
|---|---|
| `composed.build(dyn0, cost)` | `composed.build(make_derived_exprs(), make_component_dyns(dyn0), cost)` |
| `composed.build(derived0, dyn0, cost)` | `composed.build(make_derived_exprs(derived0), make_component_dyns(dyn0), cost)` |

All existing tests in `test_composed_model.cpp` and `test_composition_solve.cpp` are migrated to the new call syntax as part of Task 2. The I2 guard (derived-count mismatch) changes: instead of checking `total_num_derived() != 1`, the new guard checks `std::tuple_size_v<DerivedTuple> != topo_ordered_deriveds_.size()` (the number of derived-expr lambdas provided must equal the number of derived quantities declared across all components). The I1 guard (zero state owners) is kept; the upper-bound guard (`> 1`) is removed. A new guard checks `std::tuple_size_v<DynTuple>` equals the number of state-owning components (components with `num_owned_states() > 0`).

---

## Tech Stack

- C++17 (`CMAKE_CXX_STANDARD 17`, header-only `goss_model INTERFACE` library)
- GoogleTest v1.14.0 via FetchContent
- IPOPT via `goss::solver::IpoptSolver` + `goss::solver::SolverStatus::Success`
- CppADCG JIT via `goss_ad_impl` (AD path uses concrete generic lambdas; no `std::function` in recording path)
- Build/test: `scripts/dev.sh '<cmake/ctest command>'` — never invoke cmake/ctest directly on the host
- Accuracy integration tests reuse `tests/accuracy/accuracy_helpers.hpp` from the companion accuracy-validation-suite plan (must be merged before Task 5 executes)

---

## Global Constraints

1. **C++17 only** — no C++20 features (no concepts, no `std::span`, no designated initialisers on non-aggregate types).
2. **Header-only** — all changes live in `include/goss/model/composed_model.hpp`. No new `.cpp` files.
3. **AD-safety invariant** — `std::function` MUST NOT appear in any path that reaches `Model::build()`. The `ComposedDynamicsFunctor` and `ComposedCostFunctor` structs are concrete template types holding tuples by value; `std::function` is used only in the double-typed validation members of `Component` (which are never called after `build()` returns).
4. **Do NOT modify** transcription, NLP, solver, or AD layers (`include/goss/transcription/`, `include/goss/nlp/`, `include/goss/solver/`, `include/goss/ad/`, `src/`).
5. **ComponentError for misuse** — wrong derived-expr count, wrong dynamics-lambda count, zero state owners, cycle in deriveds, unresolved input state.
6. **Container-first** — all cmake/ctest commands run inside the dev container via `scripts/dev.sh`.
7. **Verbose names** — variable names like `component_num_owned_states`, `derived_dependency_indices`, `num_state_owning_components`; no one-letter local variables except conventional loop indices.
8. **Backward compatibility** — existing tests must pass after migration to new call syntax (Task 2 migrates them).

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `include/goss/model/composed_model.hpp` | Modify | Add `ComposedDynamicsFunctor`, `ComposedCostFunctor`, `make_derived_exprs`, `make_component_dyns`; replace 0/1-derived `build()` overloads with variadic `build()`; remove I1 upper-bound guard; add new I_dyn guard; update I2 guard |
| `tests/model/test_composed_model.cpp` | Modify | Migrate all existing `build(...)` call sites to new syntax; add unit tests for multi-state-owner functor assembly, multi-derived topo evaluation, new guard errors |
| `tests/model/test_composition_solve.cpp` | Modify | Migrate `QueueModelWithDerivedServiceRate` to new syntax |
| `tests/accuracy/test_composition_accuracy.cpp` | Create | End-to-end accuracy integration test: 2-state coupled system with 2 derived quantities, solved and asserted against known closed-form |
| `CMakeLists.txt` | Modify | Add `tests/accuracy/test_composition_accuracy.cpp` to `goss_accuracy_tests` sources (requires accuracy suite to be merged first) |

---

## Task 1: Define `ComposedDynamicsFunctor`, `ComposedCostFunctor`, and call-site helpers (failing unit tests first)

**Files:**
- Read (no modify): `include/goss/model/composed_model.hpp`, `include/goss/model/expr/expr_model.hpp`
- Modify: `tests/model/test_composed_model.cpp`

**Interfaces consumed:**
- `goss::model::ComposedModel::total_num_derived()` → `std::size_t`
- `goss::model::ComposedModel::topo_ordered_deriveds_` (private, accessed via helper in `prepare_build()`)
- `goss::model::ResolvedDerivedEntry::dependency_global_derived_indices` → `std::vector<std::size_t>`
- `goss::model::Component::num_owned_states()` → `std::size_t`
- `goss::model::Component::has_dynamics()` → `bool`
- `goss::model::ComponentError(const std::string&)` constructor

**Interfaces produced:**
- `goss::model::make_derived_exprs(DerivedExprFns... fns)` → `std::tuple<DerivedExprFns...>`
- `goss::model::make_component_dyns(DynFns... fns)` → `std::tuple<DynFns...>`
- `goss::model::ComposedDynamicsFunctor<DerivedTuple, DynTuple>` struct with `template<T> std::vector<T> operator()(const std::vector<T>& x, const std::vector<T>& u, T t) const`
- `goss::model::ComposedCostFunctor<DerivedTuple, CostFn>` struct with `template<T> T operator()(const std::vector<T>& x, const std::vector<T>& u, T t) const`

### Steps

- [ ] **Step 1.1: Write failing tests for `make_derived_exprs` / `make_component_dyns` helpers**

  Add to `tests/model/test_composed_model.cpp`:

  ```cpp
  // Task 1 — helpers compile and produce tuples of correct size.
  TEST(ComposedModelVariadic, MakeDerivedExprsProducesCorrectTupleSize) {
      auto derived_tuple = goss::model::make_derived_exprs(
          [](const auto& x, const auto& /*u*/, const auto& /*d*/, auto /*t*/) { return x[0]; },
          [](const auto& x, const auto& /*u*/, const auto& d, auto /*t*/) { return d[0] + x[0]; });
      constexpr std::size_t expected_size = 2u;
      EXPECT_EQ(std::tuple_size_v<decltype(derived_tuple)>, expected_size);
  }

  TEST(ComposedModelVariadic, MakeComponentDynsProducesCorrectTupleSize) {
      auto dyn_tuple = goss::model::make_component_dyns(
          [](const auto& /*x*/, const auto& /*u*/, const auto& /*d*/, auto /*t*/) {
              using T = double;
              return std::vector<T>{ T(0.0) };
          },
          [](const auto& /*x*/, const auto& /*u*/, const auto& /*d*/, auto /*t*/) {
              using T = double;
              return std::vector<T>{ T(0.0) };
          });
      constexpr std::size_t expected_size = 2u;
      EXPECT_EQ(std::tuple_size_v<decltype(dyn_tuple)>, expected_size);
  }
  ```

- [ ] **Step 1.2: Run-to-fail**

  ```bash
  scripts/dev.sh 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target goss_model_tests -- -j4 2>&1 | tail -30'
  ```

  Expected: compile error — `goss::model::make_derived_exprs` not found.

- [ ] **Step 1.3: Implement helpers and struct definitions in `composed_model.hpp`**

  Add after the `ResolvedDerivedEntry` struct (before the `ComposedModel` class declaration):

  ```cpp
  // ---- Variadic call-site helpers ----

  /// Wrap any number of derived-expression generic lambdas into a std::tuple.
  /// Called at the ComposedModel::build() call site:
  ///   composed.build(make_derived_exprs(expr0, expr1), make_component_dyns(dyn0, dyn1), cost);
  template <typename... DerivedExprFns>
  auto make_derived_exprs(DerivedExprFns... fns) {
      return std::make_tuple(std::forward<DerivedExprFns>(fns)...);
  }

  /// Wrap any number of per-component dynamics generic lambdas into a std::tuple.
  /// Components must be listed in the same order as their add_component() registration order.
  template <typename... DynFns>
  auto make_component_dyns(DynFns... fns) {
      return std::make_tuple(std::forward<DynFns>(fns)...);
  }

  // ---- ComposedDynamicsFunctor ----

  /// Assembled dynamics functor for a composed model.
  /// Holds a tuple of derived-expression generic lambdas (in topo order) and a tuple
  /// of per-component dynamics generic lambdas (in component registration order).
  ///
  /// AD-safety: no std::function anywhere in this struct. Both tuples hold concrete
  /// generic lambda closures captured by value. operator() is a function template
  /// instantiated by the caller (double at validation time, CppAD AD type at recording time).
  template <typename DerivedTuple, typename DynTuple>
  struct ComposedDynamicsFunctor {
      DerivedTuple                          derived_expr_tuple;
      DynTuple                              component_dyn_tuple;
      std::size_t                           num_states;
      /// component_state_offsets[i] = global state index of the first state owned by
      /// state-owning component i (in the order state-owning components appear in
      /// components_ registration order, which matches the DynTuple position).
      std::vector<std::size_t>              component_state_offsets;
      /// component_num_owned_states[i] = number of states owned by state-owning component i.
      std::vector<std::size_t>              component_num_owned_states;
      /// derived_dependency_indices[i] = topo-global indices of derived quantities that
      /// derived entry i depends on (from ResolvedDerivedEntry::dependency_global_derived_indices).
      /// Used to build the deriveds_so_far slice passed to each derived-expression lambda.
      std::vector<std::vector<std::size_t>> derived_dependency_indices;

      template <typename T>
      std::vector<T> operator()(const std::vector<T>& x,
                                 const std::vector<T>& u,
                                 T                     t) const {
          const std::size_t num_deriveds = std::tuple_size_v<DerivedTuple>;
          std::vector<T> deriveds(num_deriveds);
          // Step 1: evaluate all derived quantities in topological order.
          // The comma-fold in evaluate_deriveds_impl processes indices 0, 1, ..., N-1
          // which is exactly topological order because topo_ordered_deriveds_ preserves it.
          evaluate_deriveds_impl(deriveds, x, u, t,
                                 std::make_index_sequence<std::tuple_size_v<DerivedTuple>>{});
          // Step 2: evaluate per-component dynamics into global dx slots.
          std::vector<T> dx(num_states, T(0));
          evaluate_dyns_impl(dx, x, u, deriveds, t,
                             std::make_index_sequence<std::tuple_size_v<DynTuple>>{});
          return dx;
      }

   private:
      template <typename T, std::size_t... Idxs>
      void evaluate_deriveds_impl(std::vector<T>&       deriveds,
                                   const std::vector<T>& x,
                                   const std::vector<T>& u,
                                   T                     t,
                                   std::index_sequence<Idxs...>) const {
          // For each derived entry at topo index Idxs: build a deriveds_so_far slice
          // containing only the entries this entry explicitly declared as dependencies
          // (via input_derived() before add_derived()). Those entries have smaller topo
          // indices (guaranteed by Kahn's algorithm) so deriveds[dep_idx] is already
          // filled when we arrive here.
          ((deriveds[Idxs] = [&, this]() {
              std::vector<T> deriveds_so_far;
              deriveds_so_far.reserve(derived_dependency_indices[Idxs].size());
              for (const std::size_t dependency_index : derived_dependency_indices[Idxs]) {
                  deriveds_so_far.push_back(deriveds[dependency_index]);
              }
              return std::get<Idxs>(derived_expr_tuple)(x, u, deriveds_so_far, t);
          }()), ...);
      }

      template <typename T, std::size_t... Idxs>
      void evaluate_dyns_impl(std::vector<T>&       dx,
                               const std::vector<T>& x,
                               const std::vector<T>& u,
                               const std::vector<T>& deriveds,
                               T                     t,
                               std::index_sequence<Idxs...>) const {
          // For each state-owning component at position Idxs in DynTuple:
          // evaluate its dynamics lambda (returns a local vector of size num_owned_states[Idxs])
          // and copy each element into the global dx slot at component_state_offsets[Idxs] + k.
          ((([&, this]() {
              auto component_dx = std::get<Idxs>(component_dyn_tuple)(x, u, deriveds, t);
              const std::size_t global_offset = component_state_offsets[Idxs];
              const std::size_t n_owned       = component_num_owned_states[Idxs];
              for (std::size_t k = 0; k < n_owned; ++k) {
                  dx[global_offset + k] = component_dx[k];
              }
          })()), ...);
      }
  };

  // ---- ComposedCostFunctor ----

  /// Assembled cost functor for a composed model.
  /// Re-evaluates derived quantities in topological order then delegates to the combined cost lambda.
  /// AD-safety: same invariant as ComposedDynamicsFunctor — no std::function in this struct.
  template <typename DerivedTuple, typename CostFn>
  struct ComposedCostFunctor {
      DerivedTuple                          derived_expr_tuple;
      CostFn                                combined_cost_fn;
      std::vector<std::vector<std::size_t>> derived_dependency_indices;

      template <typename T>
      T operator()(const std::vector<T>& x,
                   const std::vector<T>& u,
                   T                     t) const {
          const std::size_t num_deriveds = std::tuple_size_v<DerivedTuple>;
          std::vector<T> deriveds(num_deriveds);
          evaluate_deriveds_impl(deriveds, x, u, t,
                                 std::make_index_sequence<std::tuple_size_v<DerivedTuple>>{});
          return combined_cost_fn(x, u, deriveds, t);
      }

   private:
      template <typename T, std::size_t... Idxs>
      void evaluate_deriveds_impl(std::vector<T>&       deriveds,
                                   const std::vector<T>& x,
                                   const std::vector<T>& u,
                                   T                     t,
                                   std::index_sequence<Idxs...>) const {
          ((deriveds[Idxs] = [&, this]() {
              std::vector<T> deriveds_so_far;
              deriveds_so_far.reserve(derived_dependency_indices[Idxs].size());
              for (const std::size_t dependency_index : derived_dependency_indices[Idxs]) {
                  deriveds_so_far.push_back(deriveds[dependency_index]);
              }
              return std::get<Idxs>(derived_expr_tuple)(x, u, deriveds_so_far, t);
          }()), ...);
      }
  };
  ```

- [ ] **Step 1.4: Run-to-pass**

  ```bash
  scripts/dev.sh 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target goss_model_tests -- -j4 && ctest --test-dir build -R "ComposedModelVariadic" -V 2>&1 | tail -30'
  ```

  Expected: both new tests pass; all pre-existing tests still pass.

- [ ] **Step 1.5: Commit**

  ```
  feat(model): add ComposedDynamicsFunctor, ComposedCostFunctor, and variadic helpers

  Adds make_derived_exprs/make_component_dyns helpers and the two assembled
  functor structs that will replace the fixed-arity build() overloads.
  No behaviour change yet — new types are unused.
  ```

---

## Task 2: Replace fixed-arity `build()` overloads with variadic `build()` and migrate existing tests

**Files:**
- Modify: `include/goss/model/composed_model.hpp`
- Modify: `tests/model/test_composed_model.cpp`
- Modify: `tests/model/test_composition_solve.cpp`

**Interfaces consumed:**
- `ComposedDynamicsFunctor<DerivedTuple, DynTuple>` (from Task 1)
- `ComposedCostFunctor<DerivedTuple, CostFn>` (from Task 1)
- `ComposedModel::prepare_build()` (calls `resolve_names()`, validates mesh, runs `validate_dynamics_dimensions()`)
- `ComposedModel::topo_ordered_deriveds_` → `std::vector<ResolvedDerivedEntry>` (populated by `resolve_names()`)
- `ResolvedDerivedEntry::dependency_global_derived_indices` → `std::vector<std::size_t>`
- `ComposedModel::compute_component_state_offsets()` → `std::vector<std::size_t>`
- `ComposedModel::build_internal_model(DynamicsFn, CostFn)` → `OcpProblem<DynamicsFn, CostFn>`

**Interfaces produced:**
- New `ComposedModel::build(std::tuple<DerivedExprFns...>, std::tuple<DynFns...>, CostFn)` overload
- Guard errors: `ComponentError` when `sizeof...(DerivedExprFns) != topo_ordered_deriveds_.size()`, when `sizeof...(DynFns) != num_state_owning_components`, when zero state owners

### Steps

- [ ] **Step 2.1: Write failing tests that use new call syntax**

  Add to `tests/model/test_composed_model.cpp` (BEFORE migrating old tests):

  ```cpp
  // Task 2 — variadic build() with 0 derived and 1 state-owning component.
  TEST(ComposedModelVariadic, ZeroDerivedOneDynBuildSucceeds) {
      goss::model::ComposedModel composed;
      composed.add_control("u", 0.0, 5.0);

      goss::model::Component comp("c");
      auto q_handle = comp.add_state("q");
      comp.set_initial_state(q_handle, 1.0);
      comp.set_dynamics(
          [](const std::vector<double>& /*x*/,
             const std::vector<double>& u,
             const std::vector<double>& /*d*/,
             double /*t*/) {
              return std::vector<double>{ u[0] };
          });
      composed.add_component(std::move(comp));
      composed.set_mesh(0.0, 1.0, 4);

      auto dyn_lambda = [](const auto& /*x*/, const auto& u,
                            const auto& /*d*/, auto /*t*/) {
          using T = typename std::decay_t<decltype(u)>::value_type;
          return std::vector<T>{ u[0] };
      };
      auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                             const auto& /*d*/, auto /*t*/) {
          return double(0.0);
      };
      auto ocp = composed.build(
          goss::model::make_derived_exprs(),
          goss::model::make_component_dyns(dyn_lambda),
          cost_lambda);
      EXPECT_EQ(ocp.num_states, 1u);
      EXPECT_EQ(ocp.num_controls, 1u);
  }

  // Task 2 — variadic build() with 1 derived and 1 state-owning component.
  TEST(ComposedModelVariadic, OneDerivedOneDynBuildSucceeds) {
      goss::model::ComposedModel composed;
      composed.add_control("u", 0.0, 5.0);

      goss::model::Component comp("c");
      comp.add_state("q");
      comp.add_derived(
          "d_val",
          [](const std::vector<double>& x, const std::vector<double>& /*u*/,
             const std::vector<double>& /*d*/, double /*t*/) { return x[0] * 2.0; });
      comp.set_dynamics(
          [](const std::vector<double>& /*x*/, const std::vector<double>& /*u*/,
             const std::vector<double>& d, double /*t*/) {
              return std::vector<double>{ -d[0] };
          });
      composed.add_component(std::move(comp));
      composed.set_mesh(0.0, 1.0, 3);

      auto derived_lambda = [](const auto& x, const auto& /*u*/,
                                const auto& /*d*/, auto /*t*/) {
          return x[0] * decltype(x[0])(2.0);
      };
      auto dyn_lambda = [](const auto& /*x*/, const auto& /*u*/,
                            const auto& d, auto /*t*/) {
          using T = typename std::decay_t<decltype(d)>::value_type;
          return std::vector<T>{ -d[0] };
      };
      auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                             const auto& /*d*/, auto /*t*/) {
          return double(0.0);
      };
      auto ocp = composed.build(
          goss::model::make_derived_exprs(derived_lambda),
          goss::model::make_component_dyns(dyn_lambda),
          cost_lambda);
      EXPECT_EQ(ocp.num_states, 1u);

      std::vector<double> x_test{ 3.0 }, u_test{ 0.0 };
      auto dx = ocp.dynamics(x_test, u_test, 0.0);
      ASSERT_EQ(dx.size(), 1u);
      EXPECT_DOUBLE_EQ(dx[0], -6.0);  // -2 * 3.0
  }

  // Task 2 — guard: wrong number of derived-expr lambdas throws ComponentError.
  TEST(ComposedModelVariadic, WrongDerivedExprCountThrows) {
      goss::model::ComposedModel composed;
      goss::model::Component comp("c");
      comp.add_state("q");
      comp.add_derived(
          "d_val",
          [](const std::vector<double>& x, const std::vector<double>& /*u*/,
             const std::vector<double>& /*d*/, double /*t*/) { return x[0]; });
      comp.set_dynamics(
          [](const std::vector<double>&, const std::vector<double>&,
             const std::vector<double>&, double) { return std::vector<double>{ 0.0 }; });
      composed.add_component(std::move(comp));
      composed.set_mesh(0.0, 1.0, 3);

      // Provide 0 derived-expr lambdas but model has 1 derived → guard must throw.
      auto dyn_lambda = [](const auto& /*x*/, const auto& /*u*/,
                            const auto& /*d*/, auto /*t*/) {
          using T = double;
          return std::vector<T>{ T(0.0) };
      };
      auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                             const auto& /*d*/, auto /*t*/) { return 0.0; };
      EXPECT_THROW(
          composed.build(
              goss::model::make_derived_exprs(),      // wrong: 0 provided, 1 needed
              goss::model::make_component_dyns(dyn_lambda),
              cost_lambda),
          goss::model::ComponentError);
  }

  // Task 2 — guard: wrong number of dynamics lambdas throws ComponentError.
  TEST(ComposedModelVariadic, WrongDynLambdaCountThrows) {
      goss::model::ComposedModel composed;
      goss::model::Component comp("c");
      comp.add_state("q");
      comp.set_dynamics(
          [](const std::vector<double>&, const std::vector<double>&,
             const std::vector<double>&, double) { return std::vector<double>{ 0.0 }; });
      composed.add_component(std::move(comp));
      composed.set_mesh(0.0, 1.0, 3);

      // Provide 0 dyn lambdas but model has 1 state-owning component → guard must throw.
      auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                             const auto& /*d*/, auto /*t*/) { return 0.0; };
      EXPECT_THROW(
          composed.build(
              goss::model::make_derived_exprs(),
              goss::model::make_component_dyns(),     // wrong: 0 provided, 1 needed
              cost_lambda),
          goss::model::ComponentError);
  }
  ```

- [ ] **Step 2.2: Run-to-fail**

  ```bash
  scripts/dev.sh 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target goss_model_tests -- -j4 2>&1 | tail -30'
  ```

  Expected: compile error — `ComposedModel::build(tuple, tuple, cost)` not found.

- [ ] **Step 2.3: Implement variadic `build()` in `composed_model.hpp`**

  **Remove** the two existing `build()` overloads (0-derived and 1-derived) and **replace** with:

  ```cpp
  /// Variadic build(): supports any number of derived-expression lambdas and any number
  /// of per-component dynamics lambdas. This is the single unified build() overload
  /// that supersedes the v1 0-derived and 1-derived overloads.
  ///
  /// Argument ordering:
  ///   1. derived_expr_tuple — wrap with make_derived_exprs(expr0, expr1, ...) in topo order
  ///      (signature per expr: (const auto& x, const auto& u, const auto& deps_so_far, auto t) -> T)
  ///   2. component_dyn_tuple — wrap with make_component_dyns(dyn0, dyn1, ...) in
  ///      component registration order, ONE lambda per state-owning component
  ///      (signature per dyn: (const auto& x, const auto& u, const auto& deriveds, auto t) -> vector<T>)
  ///   3. combined_cost — ONE lambda: (const auto& x, const auto& u, const auto& deriveds, auto t) -> T
  ///
  /// Guards (all throw ComponentError):
  ///   I1: zero state-owning components
  ///   I_dyn: number of lambdas in component_dyn_tuple != number of state-owning components
  ///   I2: number of lambdas in derived_expr_tuple != total_num_derived()
  ///   I3 (from prepare_build): dangling input_derived() names not consumed by add_derived()
  ///   Dimension mismatch: from validate_dynamics_dimensions()
  template <typename DerivedTuple, typename DynTuple, typename CostFn>
  auto build(DerivedTuple derived_expr_tuple,
             DynTuple     component_dyn_tuple,
             CostFn       combined_cost) {
      prepare_build();

      // I2 guard: number of derived-expr lambdas must equal total derived quantities declared.
      {
          const std::size_t num_provided_derived_exprs = std::tuple_size_v<DerivedTuple>;
          const std::size_t num_declared_deriveds      = topo_ordered_deriveds_.size();
          if (num_provided_derived_exprs != num_declared_deriveds) {
              throw ComponentError(
                  "ComposedModel::build: " +
                  std::to_string(num_declared_deriveds) +
                  " derived quantity/ies declared across all components, but " +
                  std::to_string(num_provided_derived_exprs) +
                  " derived-expression lambda(s) provided to build(). "
                  "Wrap derived lambdas with make_derived_exprs(...) in topological order.");
          }
      }

      // I_dyn guard: number of dynamics lambdas must equal number of state-owning components.
      {
          std::size_t num_state_owning_components = 0;
          for (const auto& component : components_) {
              if (component.num_owned_states() > 0) {
                  ++num_state_owning_components;
              }
          }
          const std::size_t num_provided_dyn_lambdas = std::tuple_size_v<DynTuple>;
          if (num_provided_dyn_lambdas != num_state_owning_components) {
              throw ComponentError(
                  "ComposedModel::build: " +
                  std::to_string(num_state_owning_components) +
                  " state-owning component(s), but " +
                  std::to_string(num_provided_dyn_lambdas) +
                  " dynamics lambda(s) provided to build(). "
                  "Wrap dynamics lambdas with make_component_dyns(...) in component registration order.");
          }
      }

      // Build the vectors of offsets and sizes for state-owning components in
      // component registration order — these are stored in ComposedDynamicsFunctor
      // and used inside its evaluate_dyns_impl to place each component's dx into
      // the correct global slots.
      const std::vector<std::size_t> all_component_state_offsets =
          compute_component_state_offsets();
      std::vector<std::size_t> state_owning_component_offsets;
      std::vector<std::size_t> state_owning_component_num_states;
      for (std::size_t component_idx = 0; component_idx < components_.size(); ++component_idx) {
          const std::size_t n_owned = components_[component_idx].num_owned_states();
          if (n_owned > 0) {
              state_owning_component_offsets.push_back(all_component_state_offsets[component_idx]);
              state_owning_component_num_states.push_back(n_owned);
          }
      }

      // Build the dependency-indices vector from topo_ordered_deriveds_ — one inner
      // vector per derived entry, containing the topo-global indices of its dependencies.
      std::vector<std::vector<std::size_t>> derived_dependency_indices;
      derived_dependency_indices.reserve(topo_ordered_deriveds_.size());
      for (const auto& resolved_entry : topo_ordered_deriveds_) {
          derived_dependency_indices.push_back(
              resolved_entry.dependency_global_derived_indices);
      }

      const std::size_t num_global_states = total_num_states();

      // Assemble combined dynamics functor — no std::function in this path.
      // ComposedDynamicsFunctor captures all lambdas by value inside the tuple.
      auto combined_dynamics = ComposedDynamicsFunctor<DerivedTuple, DynTuple>{
          derived_expr_tuple,
          component_dyn_tuple,
          num_global_states,
          state_owning_component_offsets,
          state_owning_component_num_states,
          derived_dependency_indices
      };

      // Assemble combined cost functor — also no std::function in the AD path.
      auto combined_cost_functor = ComposedCostFunctor<DerivedTuple, CostFn>{
          derived_expr_tuple,
          std::move(combined_cost),
          derived_dependency_indices
      };

      return build_internal_model(std::move(combined_dynamics),
                                   std::move(combined_cost_functor));
  }
  ```

  **Update `prepare_build()`:** Remove the upper-bound guard for `num_state_owners > 1` (lines ~497–501). Keep the lower-bound guard for `num_state_owners == 0`. The new function body for the state-owner section:

  ```cpp
  // I1 guard: at least one component must own states.
  {
      std::size_t num_state_owning_components = 0;
      for (const auto& component : components_) {
          if (component.num_owned_states() > 0) {
              ++num_state_owning_components;
          }
      }
      if (num_state_owning_components == 0) {
          throw ComponentError(
              "ComposedModel::build: no component owns any state; "
              "a composed model needs at least one state.");
      }
      // NOTE: num_state_owning_components > 1 is now supported (multi-state-owner feature).
      // The I_dyn guard in build() checks that the caller provides one dynamics lambda
      // per state-owning component.
  }
  ```

- [ ] **Step 2.4: Migrate all existing `build()` call sites in test files**

  In `tests/model/test_composed_model.cpp`, replace all existing `composed.build(...)` calls:

  - `composed.build(dyn_lambda, cost_lambda)` → `composed.build(make_derived_exprs(), make_component_dyns(dyn_lambda), cost_lambda)`
  - `composed.build(derived_lambda, dyn_lambda, cost_lambda)` → `composed.build(make_derived_exprs(derived_lambda), make_component_dyns(dyn_lambda), cost_lambda)`

  In `tests/model/test_composition_solve.cpp`:
  - `composed.build(service_rate_expr, queue_dynamics, combined_cost)` → `composed.build(make_derived_exprs(service_rate_expr), make_component_dyns(queue_dynamics), combined_cost)`

  Remove the now-obsolete `MultipleStateOwnersThrows` test (or convert it to verify the new behaviour — that 2 state owners is now VALID — see Task 3).

  Update the `WrongDerivedCountForOverloadThrows` test to use the new guard message pattern (the test assertion `EXPECT_THROW(..., ComponentError)` is still valid; just ensure the call uses `make_derived_exprs() / make_component_dyns(dyn_lambda)`).

- [ ] **Step 2.5: Run-to-pass**

  ```bash
  scripts/dev.sh 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target goss_model_tests -- -j4 && ctest --test-dir build -R "Composed" -V 2>&1 | tail -50'
  ```

  Expected: all `Composed*` tests pass (new variadic tests + migrated existing tests).

- [ ] **Step 2.6: Commit**

  ```
  feat(model): replace fixed-arity build() overloads with variadic build() (A+B stub)

  Removes the 0-derived and 1-derived build() overloads and replaces them with
  a single build(derived_expr_tuple, component_dyn_tuple, cost) overload.
  Removes the I1 upper-bound guard so multiple state-owning components are now
  accepted. Migrates all existing test call sites to the new syntax.
  ```

---

## Task 3: Enable multi-state-owning components (feature A) — unit tests and validation

**Files:**
- Modify: `tests/model/test_composed_model.cpp`

**Interfaces consumed:**
- `ComposedDynamicsFunctor` (Task 1/2 — already assembles per-component dx from `component_state_offsets`)
- `ComposedModel::build(DerivedTuple, DynTuple, CostFn)` with 2-element `DynTuple`

### Steps

- [ ] **Step 3.1: Write failing tests for multi-state-owner model**

  Add to `tests/model/test_composed_model.cpp`:

  ```cpp
  // Task 3 — two state-owning components: dimensions and assembled dynamics correct.
  TEST(ComposedModelMultiStateOwner, TwoStateOwnersProduceCorrectGlobalDimensions) {
      goss::model::ComposedModel composed;
      composed.add_control("u", 0.0, 1.0);

      goss::model::Component comp_a("a");
      comp_a.add_state("x_a");
      comp_a.set_dynamics(
          [](const std::vector<double>&, const std::vector<double>&,
             const std::vector<double>&, double) {
              return std::vector<double>{ 1.0 };
          });

      goss::model::Component comp_b("b");
      comp_b.add_state("x_b");
      comp_b.set_dynamics(
          [](const std::vector<double>&, const std::vector<double>&,
             const std::vector<double>&, double) {
              return std::vector<double>{ 2.0 };
          });

      composed.add_component(std::move(comp_a));
      composed.add_component(std::move(comp_b));
      composed.set_mesh(0.0, 1.0, 4);

      auto dyn_a = [](const auto& /*x*/, const auto& /*u*/,
                       const auto& /*d*/, auto /*t*/) {
          using T = double;
          return std::vector<T>{ T(1.0) };
      };
      auto dyn_b = [](const auto& /*x*/, const auto& /*u*/,
                       const auto& /*d*/, auto /*t*/) {
          using T = double;
          return std::vector<T>{ T(2.0) };
      };
      auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                             const auto& /*d*/, auto /*t*/) { return double(0.0); };

      auto ocp = composed.build(
          goss::model::make_derived_exprs(),
          goss::model::make_component_dyns(dyn_a, dyn_b),
          cost_lambda);

      EXPECT_EQ(ocp.num_states, 2u);
      EXPECT_EQ(ocp.num_controls, 1u);
  }

  TEST(ComposedModelMultiStateOwner, TwoStateOwnersAssembledDynamicsCorrect) {
      // Two components: a owns x_a (dx_a/dt = 3.0), b owns x_b (dx_b/dt = -2.0).
      // Global state vector: [x_a, x_b] (component registration order).
      // Expected assembled dx: [3.0, -2.0].
      goss::model::ComposedModel composed;

      goss::model::Component comp_a("a");
      comp_a.add_state("x_a");
      comp_a.set_dynamics(
          [](const std::vector<double>&, const std::vector<double>&,
             const std::vector<double>&, double) {
              return std::vector<double>{ 3.0 };
          });

      goss::model::Component comp_b("b");
      comp_b.add_state("x_b");
      comp_b.set_dynamics(
          [](const std::vector<double>&, const std::vector<double>&,
             const std::vector<double>&, double) {
              return std::vector<double>{ -2.0 };
          });

      composed.add_component(std::move(comp_a));
      composed.add_component(std::move(comp_b));
      composed.set_mesh(0.0, 1.0, 4);

      auto dyn_a = [](const auto& /*x*/, const auto& /*u*/,
                       const auto& /*d*/, auto /*t*/) {
          using T = double;
          return std::vector<T>{ T(3.0) };
      };
      auto dyn_b = [](const auto& /*x*/, const auto& /*u*/,
                       const auto& /*d*/, auto /*t*/) {
          using T = double;
          return std::vector<T>{ T(-2.0) };
      };
      auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                             const auto& /*d*/, auto /*t*/) { return double(0.0); };

      auto ocp = composed.build(
          goss::model::make_derived_exprs(),
          goss::model::make_component_dyns(dyn_a, dyn_b),
          cost_lambda);

      std::vector<double> x_test{ 0.0, 0.0 }, u_test{};
      auto dx = ocp.dynamics(x_test, u_test, 0.0);
      ASSERT_EQ(dx.size(), 2u);
      // x_a is global slot 0 (comp_a registered first), x_b is global slot 1.
      EXPECT_DOUBLE_EQ(dx[0], 3.0);
      EXPECT_DOUBLE_EQ(dx[1], -2.0);
  }

  TEST(ComposedModelMultiStateOwner, TwoStateOwnersCrossStateRead) {
      // Component b reads x_a (owned by component a) via input_state.
      // dx_a/dt = 1.0; dx_b/dt = x[0] * 2.0 = x_a * 2.0.
      goss::model::ComposedModel composed;

      goss::model::Component comp_a("a");
      comp_a.add_state("x_a");
      comp_a.set_dynamics(
          [](const std::vector<double>&, const std::vector<double>&,
             const std::vector<double>&, double) {
              return std::vector<double>{ 1.0 };
          });

      goss::model::Component comp_b("b");
      comp_b.input_state("x_a");  // declares dependency; global index resolved to 0
      comp_b.add_state("x_b");
      comp_b.set_dynamics(
          [](const std::vector<double>& x, const std::vector<double>&,
             const std::vector<double>&, double) {
              // x[0] is x_a (global state index 0 after resolve_names)
              return std::vector<double>{ x[0] * 2.0 };
          });

      composed.add_component(std::move(comp_a));
      composed.add_component(std::move(comp_b));
      composed.set_mesh(0.0, 1.0, 4);

      auto dyn_a = [](const auto& /*x*/, const auto& /*u*/,
                       const auto& /*d*/, auto /*t*/) {
          using T = double;
          return std::vector<T>{ T(1.0) };
      };
      auto dyn_b = [](const auto& x, const auto& /*u*/,
                       const auto& /*d*/, auto /*t*/) {
          using T = typename std::decay_t<decltype(x)>::value_type;
          return std::vector<T>{ x[0] * T(2.0) };
      };
      auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                             const auto& /*d*/, auto /*t*/) { return double(0.0); };

      auto ocp = composed.build(
          goss::model::make_derived_exprs(),
          goss::model::make_component_dyns(dyn_a, dyn_b),
          cost_lambda);

      // At x = [3.0, 5.0]: dx_a = 1.0, dx_b = 3.0 * 2.0 = 6.0.
      std::vector<double> x_test{ 3.0, 5.0 }, u_test{};
      auto dx = ocp.dynamics(x_test, u_test, 0.0);
      ASSERT_EQ(dx.size(), 2u);
      EXPECT_DOUBLE_EQ(dx[0], 1.0);
      EXPECT_DOUBLE_EQ(dx[1], 6.0);
  }

  // Guard: zero state-owning components still throws (I1 lower bound preserved).
  TEST(ComposedModelMultiStateOwner, ZeroStateOwnersStillThrows) {
      goss::model::ComposedModel composed;
      goss::model::Component comp("derived_only");
      comp.add_derived(
          "rate",
          [](const auto& /*x*/, const auto& /*u*/,
             const auto& /*d*/, double /*t*/) { return 1.0; });
      composed.add_component(std::move(comp));
      composed.set_mesh(0.0, 1.0, 4);

      auto derived_lambda = [](const auto& /*x*/, const auto& /*u*/,
                                const auto& /*d*/, auto /*t*/) { return 1.0; };
      auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                             const auto& /*d*/, auto /*t*/) { return double(0.0); };
      EXPECT_THROW(
          composed.build(
              goss::model::make_derived_exprs(derived_lambda),
              goss::model::make_component_dyns(),
              cost_lambda),
          goss::model::ComponentError);
  }
  ```

- [ ] **Step 3.2: Run-to-fail**

  ```bash
  scripts/dev.sh 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target goss_model_tests -- -j4 && ctest --test-dir build -R "ComposedModelMultiStateOwner" -V 2>&1 | tail -30'
  ```

  Expected: tests compile but `TwoStateOwnersProduceCorrectGlobalDimensions` and siblings fail (the `prepare_build()` I1 upper-bound guard still present from old code was removed in Task 2 — but if Task 2 is done correctly, these should already pass structurally; the test failures are due to the tests not existing yet, so they fail at compile until this step).

- [ ] **Step 3.3: Run-to-pass**

  After Task 2 correctly removed the I1 upper-bound guard and `ComposedDynamicsFunctor` correctly loops over `DynTuple`, the new tests should pass without further code changes.

  ```bash
  scripts/dev.sh 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target goss_model_tests -- -j4 && ctest --test-dir build -R "ComposedModelMultiStateOwner" -V 2>&1 | tail -30'
  ```

  Expected: all 4 new tests pass.

- [ ] **Step 3.4: Commit**

  ```
  test(model): add multi-state-owner unit tests for feature A

  Verifies that two state-owning components assemble correct global dx,
  cross-state reads work, and the I1 zero-owner guard is preserved.
  ```

---

## Task 4: Enable multi-derived quantities (feature B) — unit tests and validation

**Files:**
- Modify: `tests/model/test_composed_model.cpp`

**Interfaces consumed:**
- `ComposedDynamicsFunctor::evaluate_deriveds_impl` — topo-ordered fold with dependency slices
- `ResolvedDerivedEntry::dependency_global_derived_indices`
- `ComposedModel::topo_ordered_deriveds_` (populated by `resolve_names()` / `compute_topo_ordered_deriveds()`)

### Steps

- [ ] **Step 4.1: Write failing tests for multi-derived model**

  Add to `tests/model/test_composed_model.cpp`:

  ```cpp
  // Task 4 — two derived quantities with a dependency chain: b depends on a.
  TEST(ComposedModelMultiDerived, TwoDerivedWithDependencyChainEvaluatesCorrectly) {
      // Derived a = x[0] * 3.0; derived b depends on a: b = a + 1.0.
      // Component dynamics: dx/dt = -b = -(a + 1.0) = -(3 * x[0] + 1).
      // At x = [2.0]: a = 6.0, b = 7.0, dx/dt = -7.0.
      goss::model::ComposedModel composed;

      goss::model::Component comp("c");
      comp.add_state("x");
      comp.add_derived(
          "a",
          [](const std::vector<double>& x, const std::vector<double>& /*u*/,
             const std::vector<double>& /*d*/, double /*t*/) { return x[0] * 3.0; });
      comp.input_derived("a");  // declares: b depends on a
      comp.add_derived(
          "b",
          [](const std::vector<double>& /*x*/, const std::vector<double>& /*u*/,
             const std::vector<double>& d, double /*t*/) {
              return d[0] + 1.0;  // d[0] is a (topo index 0)
          });
      comp.set_dynamics(
          [](const std::vector<double>& /*x*/, const std::vector<double>& /*u*/,
             const std::vector<double>& d, double /*t*/) {
              return std::vector<double>{ -d[1] };  // d[1] is b (topo index 1)
          });
      composed.add_component(std::move(comp));
      composed.set_mesh(0.0, 1.0, 4);

      auto derived_a_lambda = [](const auto& x, const auto& /*u*/,
                                  const auto& /*d*/, auto /*t*/) {
          return x[0] * decltype(x[0])(3.0);
      };
      // derived_b_lambda: d is the deps_so_far slice — contains only d[0] = a.
      auto derived_b_lambda = [](const auto& /*x*/, const auto& /*u*/,
                                  const auto& d, auto /*t*/) {
          using T = typename std::decay_t<decltype(d)>::value_type;
          return d[0] + T(1.0);
      };
      auto dyn_lambda = [](const auto& /*x*/, const auto& /*u*/,
                            const auto& d, auto /*t*/) {
          using T = typename std::decay_t<decltype(d)>::value_type;
          return std::vector<T>{ -d[1] };  // d[1] is b in the full deriveds vector
      };
      auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                             const auto& /*d*/, auto /*t*/) { return double(0.0); };

      auto ocp = composed.build(
          goss::model::make_derived_exprs(derived_a_lambda, derived_b_lambda),
          goss::model::make_component_dyns(dyn_lambda),
          cost_lambda);

      EXPECT_EQ(ocp.num_states, 1u);
      std::vector<double> x_test{ 2.0 }, u_test{};
      auto dx = ocp.dynamics(x_test, u_test, 0.0);
      ASSERT_EQ(dx.size(), 1u);
      // a = 2.0 * 3.0 = 6.0; b = 6.0 + 1.0 = 7.0; dx/dt = -7.0
      EXPECT_DOUBLE_EQ(dx[0], -7.0);
  }

  TEST(ComposedModelMultiDerived, TwoDerivedNoDependencyBothEvaluated) {
      // Two independent derived quantities (no dependency between them).
      // a = x[0]; b = x[0] * 2.0.
      // dx/dt = a + b = x[0] + 2 * x[0] = 3 * x[0].
      // At x = [4.0]: dx/dt = 12.0.
      goss::model::ComposedModel composed;

      goss::model::Component comp("c");
      comp.add_state("x");
      comp.add_derived(
          "a",
          [](const std::vector<double>& x, const std::vector<double>& /*u*/,
             const std::vector<double>& /*d*/, double /*t*/) { return x[0]; });
      comp.add_derived(
          "b",
          [](const std::vector<double>& x, const std::vector<double>& /*u*/,
             const std::vector<double>& /*d*/, double /*t*/) { return x[0] * 2.0; });
      comp.set_dynamics(
          [](const std::vector<double>& /*x*/, const std::vector<double>& /*u*/,
             const std::vector<double>& d, double /*t*/) {
              return std::vector<double>{ d[0] + d[1] };  // a + b
          });
      composed.add_component(std::move(comp));
      composed.set_mesh(0.0, 1.0, 4);

      auto derived_a = [](const auto& x, const auto& /*u*/,
                           const auto& /*d*/, auto /*t*/) { return x[0]; };
      auto derived_b = [](const auto& x, const auto& /*u*/,
                           const auto& /*d*/, auto /*t*/) {
          return x[0] * decltype(x[0])(2.0);
      };
      auto dyn_lambda = [](const auto& /*x*/, const auto& /*u*/,
                            const auto& d, auto /*t*/) {
          using T = typename std::decay_t<decltype(d)>::value_type;
          return std::vector<T>{ d[0] + d[1] };
      };
      auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                             const auto& /*d*/, auto /*t*/) { return double(0.0); };

      auto ocp = composed.build(
          goss::model::make_derived_exprs(derived_a, derived_b),
          goss::model::make_component_dyns(dyn_lambda),
          cost_lambda);

      std::vector<double> x_test{ 4.0 }, u_test{};
      auto dx = ocp.dynamics(x_test, u_test, 0.0);
      ASSERT_EQ(dx.size(), 1u);
      EXPECT_DOUBLE_EQ(dx[0], 12.0);  // a=4, b=8, dx=12
  }

  TEST(ComposedModelMultiDerived, ThreeDerivedLinearChainEvaluatesCorrectly) {
      // Chain: c depends on b which depends on a.
      // a = 1.0 (constant); b = a * 2.0 = 2.0; c = b + a = 3.0.
      // dx/dt = c = 3.0.
      goss::model::ComposedModel composed;

      goss::model::Component comp("c");
      comp.add_state("x");
      comp.add_derived(
          "a",
          [](const std::vector<double>& /*x*/, const std::vector<double>& /*u*/,
             const std::vector<double>& /*d*/, double /*t*/) { return 1.0; });
      comp.input_derived("a");
      comp.add_derived(
          "b",
          [](const std::vector<double>& /*x*/, const std::vector<double>& /*u*/,
             const std::vector<double>& d, double /*t*/) { return d[0] * 2.0; });
      comp.input_derived("a");
      comp.input_derived("b");
      comp.add_derived(
          "c",
          [](const std::vector<double>& /*x*/, const std::vector<double>& /*u*/,
             const std::vector<double>& d, double /*t*/) { return d[0] + d[1]; });
      comp.set_dynamics(
          [](const std::vector<double>& /*x*/, const std::vector<double>& /*u*/,
             const std::vector<double>& d, double /*t*/) {
              return std::vector<double>{ d[2] };  // dx/dt = c (topo index 2)
          });
      composed.add_component(std::move(comp));
      composed.set_mesh(0.0, 1.0, 3);

      auto derived_a = [](const auto& /*x*/, const auto& /*u*/,
                           const auto& /*d*/, auto /*t*/) { return double(1.0); };
      auto derived_b = [](const auto& /*x*/, const auto& /*u*/,
                           const auto& d, auto /*t*/) {
          return d[0] * decltype(d[0])(2.0);  // d[0]=a in deps_so_far
      };
      // c depends on a (topo 0) and b (topo 1); deps_so_far = [a, b]
      auto derived_c = [](const auto& /*x*/, const auto& /*u*/,
                           const auto& d, auto /*t*/) {
          using T = typename std::decay_t<decltype(d)>::value_type;
          return d[0] + d[1];  // d[0]=a, d[1]=b in deps_so_far
      };
      auto dyn_lambda = [](const auto& /*x*/, const auto& /*u*/,
                            const auto& d, auto /*t*/) {
          using T = typename std::decay_t<decltype(d)>::value_type;
          return std::vector<T>{ d[2] };  // d[2]=c in the full deriveds vector
      };
      auto cost_lambda = [](const auto& /*x*/, const auto& /*u*/,
                             const auto& /*d*/, auto /*t*/) { return double(0.0); };

      auto ocp = composed.build(
          goss::model::make_derived_exprs(derived_a, derived_b, derived_c),
          goss::model::make_component_dyns(dyn_lambda),
          cost_lambda);

      std::vector<double> x_test{ 0.0 }, u_test{};
      auto dx = ocp.dynamics(x_test, u_test, 0.0);
      ASSERT_EQ(dx.size(), 1u);
      EXPECT_DOUBLE_EQ(dx[0], 3.0);  // a=1, b=2, c=3
  }
  ```

- [ ] **Step 4.2: Run-to-fail**

  ```bash
  scripts/dev.sh 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target goss_model_tests -- -j4 && ctest --test-dir build -R "ComposedModelMultiDerived" -V 2>&1 | tail -30'
  ```

  Expected: tests compile; correctness failures on chained derived quantity values (the `evaluate_deriveds_impl` dependency-slice logic needs verification).

- [ ] **Step 4.3: Run-to-pass** (no code changes expected if Tasks 1–2 implemented correctly)

  ```bash
  scripts/dev.sh 'ctest --test-dir build -R "ComposedModelMultiDerived" -V 2>&1 | tail -30'
  ```

  If any test fails, inspect the `derived_dependency_indices` population loop in `build()` (Task 2 Step 2.3). The most common bug is confusing the `deps_so_far` slice (which uses the dependency's position within the dependency list) with the full `deriveds` vector index. The `deps_so_far` in the lambda signature is a SLICE — `d[0]` is the first declared dependency, not global topo index 0.

  **Key note:** The dynamics lambdas in the tests above use `d[1]` and `d[2]` to refer to the FULL `deriveds` vector (as produced by `ComposedDynamicsFunctor::operator()`), while the derived-expression lambdas receive `deps_so_far` (a slice of only their declared dependencies). These are distinct vectors. The derived-expression lambdas for b and c use `d[0]` and `d[1]` inside the slice, meaning "my first declared dependency" and "my second declared dependency" respectively — these are guaranteed populated because topo order ensures they are evaluated before the current entry.

- [ ] **Step 4.4: Commit**

  ```
  test(model): add multi-derived unit tests for feature B (2 and 3 derived quantities)

  Tests cover independent multi-derived, linear dependency chain (b→a),
  and 3-level chain (c→b→a) with exact numerical assertions.
  ```

---

## Task 5: End-to-end integration test — 2-state coupled system with 2 derived quantities (accuracy suite dependency)

**Prerequisite:** `tests/accuracy/accuracy_helpers.hpp` (from `2026-08-01-accuracy-validation-suite.md`) MUST be merged before this task executes. The accuracy suite introduces:
- `goss::accuracy::SolutionTrajectory` struct
- `goss::accuracy::solve_and_extract_trajectory(compiled_ocp, initial_guess_value)` → `SolutionTrajectory`

**Files:**
- Create: `tests/accuracy/test_composition_accuracy.cpp`
- Modify: `CMakeLists.txt` (add new file to `goss_accuracy_tests` sources)

**Problem design — 2-state Lotka-Volterra-like coupled oscillator (closed-form minimum-energy):**

The test uses a LINEAR coupled system (not Lotka-Volterra, which has no closed-form) with two state-owning components and two derived quantities:

```
States:    x (component "alpha"), y (component "beta")
Controls:  u (single control, min integral u^2)
Derived:   d0 = x + y  (sum),  d1 = x - y  (difference) — no mutual dependency, so parallel topo

Dynamics:
  dx/dt  = -a * x + u        (alpha component, a = 1.0)
  dy/dt  = -b * y            (beta component,  b = 2.0)

Boundary:  x(0) = 1.0, x(T) = 0.0 (pinned);  y(0) = 1.0 (pinned, y(T) free)
Time horizon: T = 2.0, N = 40 intervals

Cost:  integral_0^T u^2 dt    (minimum-energy control of x; y evolves independently)

Closed-form answer:
  Since y is uncontrolled and y(T) is free, the y subsystem is decoupled from the optimisation:
    y(t) = y0 * exp(-b * t) = exp(-2t)
  The x-subsystem is a standard linear minimum-energy transfer with a = 1:
    dx/dt = -x + u,  x(0)=1, x(T)=0, min integral u^2.
  The closed-form optimal cost for this problem (Pontryagin) is:
    J* = (1 - exp(-2aT)) / (2a * (1 - exp(-2aT)) / (2a))
  More precisely, for the linear problem dx/dt = -a*x + u, x(0)=x0, x(T)=0, min integral u^2:
    J* = x0^2 * a / (1 - exp(-2aT))
  With a=1, x0=1, T=2:
    J* = 1.0 / (1.0 - exp(-4.0))
       ≈ 1.0 / (1.0 - 0.018316) ≈ 1.0 / 0.981684 ≈ 1.01864

  The derived quantities d0 and d1 are purely observational (cost does not depend on them);
  their evaluation is verified indirectly by confirming the dynamics are assembled correctly.
  As a bonus assertion: d0(0) = x(0) + y(0) = 2.0 — verified by evaluating ocp.dynamics
  with the initial state and checking the assembled dx values.
```

**Tolerance:** `EXPECT_NEAR(result.objective_value, 1.01864, 1e-2)` (1% tolerance for 40-interval Hermite-Simpson).

**Interfaces consumed:**
- `goss::model::ComposedModel::add_control`, `add_component`, `set_mesh`
- `goss::model::Component::add_state`, `set_initial_state`, `set_final_state`, `set_dynamics`, `add_derived`
- `goss::model::make_derived_exprs`, `goss::model::make_component_dyns`
- `goss::model::ComposedModel::build(DerivedTuple, DynTuple, CostFn)` with 2-element tuples
- `goss::transcription::HermiteSimpson::compile(ocp, model_name)` → `CompiledOcp`
- `goss::solver::IpoptSolver::solve(nlp, initial_guess)` → `SolverResult`
- `goss::solver::SolverStatus::Success`
- `goss::accuracy::solve_and_extract_trajectory` (from accuracy suite)
- `goss::accuracy::SolutionTrajectory`

### Steps

- [ ] **Step 5.1: Write the accuracy integration test**

  Create `tests/accuracy/test_composition_accuracy.cpp`:

  ```cpp
  // tests/accuracy/test_composition_accuracy.cpp
  // End-to-end accuracy test for the composition quick-wins feature:
  //   - 2 state-owning components (feature A)
  //   - 2 derived quantities (feature B)
  //   - solved result asserted against closed-form optimal cost
  //
  // Dependency: requires goss_accuracy_tests target (accuracy-validation-suite plan
  // must be merged before this task executes).
  #include <gtest/gtest.h>
  #include <cmath>
  #include <vector>
  #include "goss/model/component.hpp"
  #include "goss/model/composed_model.hpp"
  #include "goss/transcription/hermite_simpson.hpp"
  #include "goss/solver/ipopt_solver.hpp"
  #include "accuracy/accuracy_helpers.hpp"

  TEST(CompositionAccuracy, TwoStateOwnerTwoDerivedClosedFormOptimalCost) {
      // Problem: 2-state linear OCP with 2 state-owning components and 2 observational
      // derived quantities. The two states are decoupled: x is controlled, y is free.
      //
      // System:
      //   dx/dt = -a * x + u,  x(0)=1.0, x(T)=0.0,  min integral u^2
      //   dy/dt = -b * y,      y(0)=1.0, y(T) free
      //   d0    = x + y        (observational derived, topo index 0)
      //   d1    = x - y        (observational derived, topo index 1)
      //
      // Closed-form optimal cost (Pontryagin minimum principle for linear-quadratic):
      //   J* = x0^2 * a / (1 - exp(-2*a*T))
      // With a=1.0, x0=1.0, T=2.0:
      //   J* = 1.0 / (1.0 - exp(-4.0)) ≈ 1.01864

      constexpr double decay_rate_x  = 1.0;   // a
      constexpr double decay_rate_y  = 2.0;   // b
      constexpr double initial_x     = 1.0;
      constexpr double initial_y     = 1.0;
      constexpr double final_x       = 0.0;
      constexpr double time_horizon  = 2.0;
      constexpr std::size_t num_intervals = 40;

      // Closed-form optimal cost for linear minimum-energy transfer dx/dt = -a*x + u,
      // x(0)=x0, x(T)=0, min integral u^2:  J* = x0^2 * a / (1 - exp(-2*a*T))
      const double closed_form_optimal_cost =
          (initial_x * initial_x * decay_rate_x) /
          (1.0 - std::exp(-2.0 * decay_rate_x * time_horizon));

      goss::model::ComposedModel composed;
      const auto control_handle = composed.add_control("u", -10.0, 10.0);

      // Component "alpha" owns state x; publishes derived d0 = x + y.
      goss::model::Component comp_alpha("alpha");
      const auto x_handle = comp_alpha.add_state("x");
      comp_alpha.set_initial_state(x_handle, initial_x);
      comp_alpha.set_final_state(x_handle, final_x);
      comp_alpha.add_derived(
          "d0",
          // Validation lambda: d0 = x + y = global_x[0] + global_x[1]
          [](const std::vector<double>& global_x, const std::vector<double>& /*u*/,
             const std::vector<double>& /*d*/, double /*t*/) {
              return global_x[0] + global_x[1];  // x + y
          });
      comp_alpha.set_dynamics(
          // Validation lambda: dx/dt = -a*x + u
          [decay_rate_x](const std::vector<double>& global_x,
                          const std::vector<double>& global_u,
                          const std::vector<double>& /*d*/,
                          double /*t*/) {
              return std::vector<double>{ -decay_rate_x * global_x[0] + global_u[0] };
          });

      // Component "beta" owns state y; publishes derived d1 = x - y.
      goss::model::Component comp_beta("beta");
      comp_beta.input_state("x");  // reads x from component alpha
      const auto y_handle = comp_beta.add_state("y");
      comp_beta.set_initial_state(y_handle, initial_y);
      comp_beta.add_derived(
          "d1",
          // Validation lambda: d1 = x - y = global_x[0] - global_x[1]
          [](const std::vector<double>& global_x, const std::vector<double>& /*u*/,
             const std::vector<double>& /*d*/, double /*t*/) {
              return global_x[0] - global_x[1];  // x - y
          });
      comp_beta.set_dynamics(
          // Validation lambda: dy/dt = -b*y
          [decay_rate_y](const std::vector<double>& global_x,
                          const std::vector<double>& /*u*/,
                          const std::vector<double>& /*d*/,
                          double /*t*/) {
              // global_x[0]=x, global_x[1]=y (registration order)
              return std::vector<double>{ -decay_rate_y * global_x[1] };
          });

      composed.add_component(std::move(comp_alpha));
      composed.add_component(std::move(comp_beta));
      composed.set_mesh(0.0, time_horizon, num_intervals);

      // Generic AD-safe lambdas for the build() call.
      // d0 = global_x[0] + global_x[1]; no declared dependencies (no input_derived calls).
      auto derived_d0 = [](const auto& global_x, const auto& /*u*/,
                            const auto& /*deps_so_far*/, auto /*t*/) {
          using T = typename std::decay_t<decltype(global_x)>::value_type;
          return global_x[0] + global_x[1];
      };
      // d1 = global_x[0] - global_x[1]; also no declared dependencies.
      auto derived_d1 = [](const auto& global_x, const auto& /*u*/,
                            const auto& /*deps_so_far*/, auto /*t*/) {
          using T = typename std::decay_t<decltype(global_x)>::value_type;
          return global_x[0] - global_x[1];
      };

      // alpha dynamics: dx/dt = -a * global_x[0] + global_u[0]
      auto alpha_dyn = [decay_rate_x](const auto& global_x, const auto& global_u,
                                       const auto& /*deriveds*/, auto /*t*/) {
          using T = typename std::decay_t<decltype(global_x)>::value_type;
          return std::vector<T>{ -T(decay_rate_x) * global_x[0] + global_u[0] };
      };
      // beta dynamics: dy/dt = -b * global_x[1]
      auto beta_dyn = [decay_rate_y](const auto& global_x, const auto& /*global_u*/,
                                      const auto& /*deriveds*/, auto /*t*/) {
          using T = typename std::decay_t<decltype(global_x)>::value_type;
          return std::vector<T>{ -T(decay_rate_y) * global_x[1] };
      };

      // Cost: integral u^2 (x subsystem only; y is free)
      auto running_cost = [](const auto& /*global_x*/, const auto& global_u,
                              const auto& /*deriveds*/, auto /*t*/) {
          return global_u[0] * global_u[0];
      };

      auto ocp = composed.build(
          goss::model::make_derived_exprs(derived_d0, derived_d1),
          goss::model::make_component_dyns(alpha_dyn, beta_dyn),
          running_cost);

      ASSERT_EQ(ocp.num_states, 2u);
      ASSERT_EQ(ocp.num_controls, 1u);

      // Structural assertion: evaluate assembled dynamics at initial state.
      // At x=[1.0, 1.0], u=[0.0]: dx/dt = [-1.0, -2.0].
      {
          std::vector<double> x_initial{ initial_x, initial_y };
          std::vector<double> u_zero{ 0.0 };
          auto dx_initial = ocp.dynamics(x_initial, u_zero, 0.0);
          ASSERT_EQ(dx_initial.size(), 2u);
          EXPECT_NEAR(dx_initial[0], -decay_rate_x * initial_x, 1e-12);  // dx/dt = -1*1 + 0
          EXPECT_NEAR(dx_initial[1], -decay_rate_y * initial_y, 1e-12);  // dy/dt = -2*1
      }

      // Solve the composed OCP.
      auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "composition_accuracy_two_state");
      const goss::accuracy::SolutionTrajectory trajectory =
          goss::accuracy::solve_and_extract_trajectory(compiled, /*initial_guess_value=*/0.1);

      // Primary accuracy assertion: objective must match closed-form to 1% tolerance.
      // With 40 Hermite-Simpson intervals on a smooth linear-quadratic problem,
      // the transcription error is O(h^4) ≈ O((2/40)^4) ≈ 1.56e-5; 1% is very conservative.
      EXPECT_NEAR(trajectory.objective_value, closed_form_optimal_cost, 1e-2);

      // Secondary: initial state pinned correctly.
      EXPECT_NEAR(trajectory.states[0][0], initial_x, 1e-6);  // x(0) = 1.0
      EXPECT_NEAR(trajectory.states[0][1], initial_y, 1e-6);  // y(0) = 1.0

      // Final state of x must be pinned to 0.0.
      EXPECT_NEAR(trajectory.states.back()[0], final_x, 1e-4);

      // y evolves freely: y(T) should match exp(-b*T) = exp(-4.0) within 1%.
      const double expected_y_final = initial_y * std::exp(-decay_rate_y * time_horizon);
      EXPECT_NEAR(trajectory.states.back()[1], expected_y_final, 1e-2);
  }
  ```

- [ ] **Step 5.2: Wire into CMakeLists.txt**

  In `CMakeLists.txt`, find the `goss_accuracy_tests` executable sources block and append `tests/accuracy/test_composition_accuracy.cpp`. The block to modify (after accuracy suite is merged):

  ```cmake
  add_executable(goss_accuracy_tests
    tests/accuracy/test_closed_form.cpp
    tests/accuracy/test_benchmarks.cpp
    tests/accuracy/test_convergence_order.cpp
    tests/accuracy/test_invariants.cpp
    tests/accuracy/test_composition_accuracy.cpp)   # <-- add this line
  ```

- [ ] **Step 5.3: Run-to-fail**

  ```bash
  scripts/dev.sh 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target goss_accuracy_tests -- -j4 2>&1 | tail -30'
  ```

  Expected: compile error if accuracy suite not merged; if merged, compile success and test fails because composition feature not yet in `build()`.

- [ ] **Step 5.4: Run-to-pass**

  ```bash
  scripts/dev.sh 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target goss_accuracy_tests -- -j4 && ctest --test-dir build -R "CompositionAccuracy" -V 2>&1 | tail -40'
  ```

  Expected: `CompositionAccuracy.TwoStateOwnerTwoDerivedClosedFormOptimalCost` passes with `objective_value ≈ 1.01864 ± 0.01`.

- [ ] **Step 5.5: Full regression pass — all test targets**

  ```bash
  scripts/dev.sh 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -- -j4 && ctest --test-dir build --output-on-failure 2>&1 | tail -60'
  ```

  Expected: all existing tests pass (goss_model_tests, goss_transcription_tests, goss_solver_tests, goss_ad_tests, goss_nlp_tests, goss_sim_tests, goss_bench_tests, goss_accuracy_tests).

- [ ] **Step 5.6: Commit**

  ```
  feat(model): add composition accuracy integration test (A+B end-to-end)

  Adds tests/accuracy/test_composition_accuracy.cpp: 2-state coupled OCP
  (alpha+beta components, 2 derived quantities) solved with HermiteSimpson and
  asserted against closed-form optimal cost J*=1/(1-exp(-4))≈1.01864 at 1e-2 tol.
  Wires the file into goss_accuracy_tests.
  ```

---

## Self-Review

### Spec Coverage of A + B

- **A (Multi-state-owner):** The `prepare_build()` I1 upper-bound guard is removed in Task 2. `ComposedDynamicsFunctor` in Task 1 assembles dx by iterating over `DynTuple` with `std::index_sequence`, writing each component's output to its `component_state_offsets[Idxs]` global slot. Task 3 adds explicit unit tests for the 2-state case. Task 5 integration test uses 2 state-owning components in a solved OCP.

- **B (Multiple derived):** The existing `topo_ordered_deriveds_` populated by `resolve_names()` with `dependency_global_derived_indices` is used directly in Task 2's `build()` to populate `derived_dependency_indices` in `ComposedDynamicsFunctor`. `evaluate_deriveds_impl` processes them in tuple index order (which equals topo order because `make_derived_exprs` receives arguments in topo order, as enforced by the I2 guard). Task 4 adds unit tests for 2 and 3 derived quantities with dependency chains.

### Type/Signature Consistency vs Real Headers

- `ComposedDynamicsFunctor::operator()` signature: `template<T> std::vector<T> operator()(const std::vector<T>& x, const std::vector<T>& u, T t) const` — matches `OcpProblem<DynamicsFn>` contract at `ocp_problem.hpp` line 25.
- `ComposedCostFunctor::operator()` signature: `template<T> T operator()(const std::vector<T>& x, const std::vector<T>& u, T t) const` — matches `OcpProblem<CostFn>` contract at `ocp_problem.hpp` line 26.
- `build()` delegates to `build_internal_model(combined_dynamics, combined_cost_functor)` which calls `internal_model.build(std::move(dynamics), std::move(cost))` — the `Model::build` signature at `model.hpp` line 124 is `template<DynamicsFn, CostFn> OcpProblem<DynamicsFn, CostFn> build(DynamicsFn, CostFn) const`. Types match.
- All lambda signatures at call sites: `(const auto& x, const auto& u, const auto& deriveds_or_deps, auto t)` — consistent with existing patterns in `test_composed_model.cpp` and `test_composition_solve.cpp`.

### AD-Safety Audit

- `ComposedDynamicsFunctor` and `ComposedCostFunctor` are plain structs with concrete template `operator()`. No `std::function` anywhere in the structs.
- `DerivedTuple` and `DynTuple` hold concrete generic lambda types captured by value in `std::tuple`; no type erasure.
- `evaluate_deriveds_impl` and `evaluate_dyns_impl` are private templates expanded via `std::index_sequence` — the same pattern used by `DynamicsFunctor::fill_result` in `expr_model.hpp`.
- The `std::vector<std::vector<std::size_t>> derived_dependency_indices` member stores only runtime index data (plain `std::size_t`), not lambdas — it does not introduce any type erasure.
- The validation path (double-typed `Component::dynamics_fn_` / `Component::cost_fn_` / `DerivedEntry::validation_fn` std::functions) is invoked only in `validate_dynamics_dimensions()` inside `prepare_build()`, which runs before `build_internal_model()` and before any CppAD recording. The AD recording happens inside `transcription::HermiteSimpson::compile(ocp, ...)` which calls `ocp.dynamics(CppAD_AD_vector, ...)` — by that point `ocp.dynamics` is the concrete `ComposedDynamicsFunctor` with no `std::function`.

### Backward-Compatibility Check

- Old `build(dyn_lambda, cost_lambda)` (0-derived) → new `build(make_derived_exprs(), make_component_dyns(dyn_lambda), cost_lambda)`. All existing 0-derived tests in `test_composed_model.cpp` are migrated in Task 2 Step 2.4.
- Old `build(derived_lambda, dyn_lambda, cost_lambda)` (1-derived) → new `build(make_derived_exprs(derived_lambda), make_component_dyns(dyn_lambda), cost_lambda)`. All existing 1-derived tests migrated in Task 2 Step 2.4.
- Old `MultipleStateOwnersThrows` test (asserting the I1 upper-bound guard throws for 2 state owners) is converted in Task 2 Step 2.4 to verify the NEW behaviour (2 state owners succeeds, or removed and replaced by `TwoStateOwnersProduceCorrectGlobalDimensions` in Task 3).
- `test_composition_solve.cpp`'s `QueueModelWithDerivedServiceRate` test is migrated in Task 2 Step 2.4 and must continue passing end-to-end.
- The I3 guard (dangling `input_derived`) is unchanged — `DanglingInputDerivedThrows` test still passes.
- The I1 lower-bound guard (zero state owners) is unchanged — `BuildWithNoStateOwnerThrows` test still passes (migrated call syntax only).
