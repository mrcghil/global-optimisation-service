// include/goss/model/composed_model.hpp
// Task 3 — name resolution + global state/derived index mapping.
// Task 4 — topological sort and cycle detection for inline derived quantities.
// Task 5 — build() assembles combined generic dynamics functor + cost, returns OcpProblem.
#pragma once
#include <algorithm>
#include <numeric>
#include <queue>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include "goss/model/component.hpp"
#include "goss/model/errors.hpp"
#include "goss/model/handles.hpp"
#include "goss/model/model.hpp"
#include "goss/transcription/ocp_problem.hpp"  // Mesh

namespace goss::model {

/// Resolution result for one inline derived quantity after name resolution.
struct ResolvedDerivedEntry {
    std::string name;
    std::size_t global_component_index;
    std::size_t local_derived_index;
    /// Global derived indices (topo order) this entry depends on.
    /// Populated by Kahn's algorithm in compute_topo_ordered_deriveds() and consumed by
    /// ComposedDynamicsFunctor/ComposedCostFunctor::evaluate_deriveds_impl to build the
    /// deps_so_far slice passed to each derived-expression lambda during evaluation.
    std::vector<std::size_t> dependency_global_derived_indices;
};

// ---- Tuple-type detection helper (used by build() static_assert) ----

/// Primary template: not a std::tuple specialization.
template <typename T>
struct is_std_tuple : std::false_type {};

/// Partial specialisation that matches any std::tuple<Ts...>.
template <typename... Ts>
struct is_std_tuple<std::tuple<Ts...>> : std::true_type {};

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

class ComposedModel {
 public:
    /// Register a control variable shared across all components.
    ControlHandle add_control(const std::string& name, double lower, double upper) {
        const std::size_t index = control_names_.size();
        control_names_.push_back(name);
        control_lower_.push_back(lower);
        control_upper_.push_back(upper);
        return ControlHandle{index};
    }

    /// Add a component. Components are stored by value; all lambdas captured inside
    /// Component's std::function fields are copied.
    void add_component(Component component) {
        components_.push_back(std::move(component));
        // Reset resolution state so resolve_names() must be called again.
        names_resolved_ = false;
    }

    /// Set the mesh for the composed problem.
    void set_mesh(double t_initial, double t_final, std::size_t num_intervals) {
        mesh_ = transcription::Mesh{t_initial, t_final, num_intervals};
        mesh_set_ = true;
    }

    /// Resolve all names, validate uniqueness, build global index maps.
    /// Throws ComponentError on unresolved input names or duplicate state/derived names.
    void resolve_names() {
        // Reset maps so re-entrant calls are safe.
        global_state_name_to_index_.clear();
        global_derived_name_to_index_.clear();
        topo_ordered_deriveds_.clear();

        validate_no_duplicate_state_names();
        validate_no_duplicate_derived_names();

        // Build global state index map: components are enumerated in insertion order;
        // each component's owned states are laid out contiguously in the global x-vector.
        std::size_t state_offset = 0;
        for (const auto& component : components_) {
            for (const auto& owned_state : component.owned_states()) {
                global_state_name_to_index_[owned_state.name] = state_offset++;
            }
        }

        // Build a preliminary name-to-index map (insertion order) so that
        // compute_topo_ordered_deriveds() can look up dependency indices.
        // The map is rebuilt after topo-sort to reflect the final topo order.
        {
            std::size_t derived_offset = 0;
            for (const auto& component : components_) {
                for (const auto& entry : component.derived_entries()) {
                    global_derived_name_to_index_[entry.name] = derived_offset++;
                }
            }
        }

        // Run Kahn's algorithm; this populates topo_ordered_deriveds_ and
        // rebuilds global_derived_name_to_index_ in topo order.
        compute_topo_ordered_deriveds();

        validate_all_inputs_resolvable();

        names_resolved_ = true;
    }

    // ---- Accessors for test assertions ----

    /// Return the derived-quantity names in topological (dependency-first) order.
    ///
    /// This is the order in which build() expects the make_derived_exprs(...) lambdas to be
    /// supplied: lambda at position i must compute the derived quantity named at position i.
    /// Passing lambdas in a different order cannot be caught at runtime (lambdas are type-opaque)
    /// and will produce silent incorrect results.
    ///
    /// If resolve_names() has not yet been called this method calls it first (which may throw
    /// ComponentError if the component graph is invalid). After a successful build() call the
    /// names are already populated and this method is cheap (no re-resolution needed).
    std::vector<std::string> topo_ordered_derived_names() {
        // resolve_names() is idempotent once names_resolved_==true; calling it here when
        // not yet resolved ensures this accessor works standalone (before build()).
        if (!names_resolved_) {
            resolve_names();
        }
        std::vector<std::string> names;
        names.reserve(topo_ordered_deriveds_.size());
        for (const auto& resolved_entry : topo_ordered_deriveds_) {
            names.push_back(resolved_entry.name);
        }
        return names;
    }

    std::size_t total_num_states() const {
        std::size_t total = 0;
        for (const auto& component : components_) {
            total += component.num_owned_states();
        }
        return total;
    }

    std::size_t total_num_derived() const {
        std::size_t total = 0;
        for (const auto& component : components_) {
            total += component.num_derived();
        }
        return total;
    }

    /// Count of algebraic variables across all registered components.
    std::size_t total_num_algebraic() const {
        std::size_t total = 0;
        for (const auto& component : components_) {
            total += component.num_algebraic();
        }
        return total;
    }

    std::size_t num_controls() const { return control_names_.size(); }

    /// Returns the global state index for a named state, or throws ComponentError.
    std::size_t global_state_index(const std::string& name) const {
        auto it = global_state_name_to_index_.find(name);
        if (it == global_state_name_to_index_.end()) {
            throw ComponentError("ComposedModel: unknown state name '" + name + "'");
        }
        return it->second;
    }

    /// Returns the global derived index (topo order) for a named derived, or throws.
    std::size_t global_derived_index(const std::string& name) const {
        auto it = global_derived_name_to_index_.find(name);
        if (it == global_derived_name_to_index_.end()) {
            throw ComponentError("ComposedModel: unknown derived name '" + name + "'");
        }
        return it->second;
    }

    // ---- Task 5 / Task 2 (composition-quickwins): single variadic build() ----
    //
    // NOTE — Algebraic-variable (DAE) flavor is NOT implemented in v1.
    // See component.hpp "FOLLOW-ON: Algebraic-variable flavor" for the full design note.
    // When Flavor 2 is implemented, build() will additionally collect algebraic entries
    // from all components, extend the OcpProblem with algebraic_residuals/num_algebraic,
    // and route them through the transcription defect machinery.

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
        // Enforce that callers use the make_derived_exprs/make_component_dyns wrappers.
        // Without this, a caller that accidentally passes a raw lambda (or a mismatched
        // container) gets an obscure tuple_size_v error deep inside the functor; this
        // static_assert surfaces the problem immediately with a clear message.
        static_assert(is_std_tuple<DerivedTuple>::value,
                      "build() expects make_derived_exprs(...) as the first argument "
                      "— wrap your derived-expression lambdas with make_derived_exprs(...)");
        static_assert(is_std_tuple<DynTuple>::value,
                      "build() expects make_component_dyns(...) as the second argument "
                      "— wrap your dynamics lambdas with make_component_dyns(...)");

        prepare_build();

        // I2 guard: number of derived-expr lambdas must equal total derived quantities declared.
        //
        // WHY ORDER CANNOT BE CHECKED AT RUNTIME: the lambdas in DerivedTuple are type-opaque
        // (each is a distinct, unnamed closure type). There is no portable way to ask a lambda
        // "which derived quantity do you compute?" at runtime or compile time. The contract is
        // therefore purely positional: lambda at tuple index i must compute the derived quantity
        // at topo position i in topo_ordered_deriveds_. Kahn's algorithm guarantees that
        // dependencies always precede dependents in that list, but it CANNOT verify that the
        // caller's lambdas are in the same order. The caller's responsibility is to pass lambdas
        // in the order returned by topo_ordered_derived_names(). Violating this contract produces
        // silent wrong results (each lambda is wired to the wrong dependency set) — not a crash.
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
                    "Wrap derived lambdas with make_derived_exprs(...) in the topological order "
                    "returned by topo_ordered_derived_names(). Passing lambdas in the wrong order "
                    "cannot be detected at runtime (lambdas are type-opaque) and will produce "
                    "silent incorrect results. Call topo_ordered_derived_names() to inspect the "
                    "required lambda ordering before calling build().");
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

    /// Build overload for 0 inline derived quantities + 1 algebraic residual functor.
    ///
    /// AD-safety: AlgResFn must be a concrete generic functor (not std::function).
    /// It is captured by value into the packed functor in build_internal_model_alg()
    /// and called during CppADCG recording under AD types.
    ///
    /// @note v1 LIMITATION — substituted dynamics only: the component dynamics lambda
    /// receives (x, u, deriveds, t) and does NOT receive the algebraic variable vector.
    /// For a semi-explicit DAE dx/dt = f(x, z_alg), the caller must analytically
    /// substitute the algebraic constraint g(x, z_alg) = 0 into f to eliminate z_alg,
    /// yielding the substituted dynamics. The NLP enforces g = 0 at every node
    /// alongside the collocation defects, so both are satisfied simultaneously by the
    /// solver. DAEs where f cannot be written without z_alg explicitly require a future
    /// build overload that threads alg_vars into the dynamics signature.
    template <typename AlgResFn, typename Dyn0Fn, typename CostFn>
    auto build_with_algebraic(AlgResFn algebraic_residuals,
                               Dyn0Fn component_0_dyn,
                               CostFn combined_cost) {
        prepare_build();

        // Guard: this overload is for 0 inline derived quantities.
        {
            const std::size_t num_derived_quantities = total_num_derived();
            if (num_derived_quantities != 0) {
                throw ComponentError(
                    "ComposedModel::build_with_algebraic (0-derived overload): "
                    "0 inline derived quantities expected, found " +
                    std::to_string(num_derived_quantities) +
                    "; use the 1-derived algebraic overload.");
            }
        }

        const std::size_t num_states      = total_num_states();
        const std::size_t num_alg         = total_num_algebraic();

        std::vector<std::size_t> component_state_offsets = compute_component_state_offsets();
        std::size_t comp0_offset  = 0;
        std::size_t comp0_nstates = 0;
        for (std::size_t ci = 0; ci < components_.size(); ++ci) {
            if (components_[ci].num_owned_states() > 0) {
                comp0_offset  = component_state_offsets[ci];
                comp0_nstates = components_[ci].num_owned_states();
                break;
            }
        }

        // Combined dynamics: the ODE right-hand side is written in the substituted form.
        // For the canonical semi-explicit index-1 DAE:
        //   dx/dt = f(x, z_alg),  g(x, z_alg) = 0
        // The caller writes the dynamics lambda as f_substituted(x, t) where z_alg is
        // replaced by its expression from g=0 (e.g. z_alg = c*x => dx/dt = (c-1)*x).
        // The algebraic_residuals functor independently enforces g=0 at every node via
        // equality constraints in the NLP. The solver satisfies both simultaneously.
        //
        // IMPORTANT: component_0_dyn takes (x, u, deriveds, t). deriveds is empty here
        // (0 inline derived quantities). The algebraic variable vector is separate from
        // x and is not passed into dynamics — the ODE sees only x and u.
        auto combined_dynamics = [
            component_0_dyn,
            num_states,
            comp0_offset,
            comp0_nstates
        ](const auto& x, const auto& u, auto t) {
            using ScalarT = typename std::decay_t<decltype(x)>::value_type;
            std::vector<ScalarT> deriveds;
            std::vector<ScalarT> dx(num_states);
            auto comp0_dx = component_0_dyn(x, u, deriveds, t);
            for (std::size_t state_idx = 0; state_idx < comp0_nstates; ++state_idx) {
                dx[comp0_offset + state_idx] = comp0_dx[state_idx];
            }
            return dx;
        };

        auto ocp_cost = [combined_cost](const auto& x, const auto& u, auto t) {
            using ScalarT = typename std::decay_t<decltype(x)>::value_type;
            std::vector<ScalarT> deriveds;
            return combined_cost(x, u, deriveds, t);
        };

        return build_internal_model_alg(
            std::move(combined_dynamics), std::move(ocp_cost),
            std::move(algebraic_residuals), num_alg);
    }

 private:
    /// Throw ComponentError if the same state name appears in more than one component.
    void validate_no_duplicate_state_names() const {
        std::unordered_map<std::string, std::string> seen;  // name → component name
        for (const auto& component : components_) {
            for (const auto& owned_state : component.owned_states()) {
                auto [it, inserted] = seen.emplace(owned_state.name, component.component_name());
                if (!inserted) {
                    throw ComponentError(
                        "ComposedModel: duplicate state name '" + owned_state.name +
                        "' in components '" + it->second + "' and '" +
                        component.component_name() + "'");
                }
            }
        }
    }

    /// Throw ComponentError if the same derived name appears in more than one component.
    void validate_no_duplicate_derived_names() const {
        std::unordered_map<std::string, std::string> seen;  // name → component name
        for (const auto& component : components_) {
            for (const auto& entry : component.derived_entries()) {
                auto [it, inserted] = seen.emplace(entry.name, component.component_name());
                if (!inserted) {
                    throw ComponentError(
                        "ComposedModel: duplicate derived name '" + entry.name +
                        "' in components '" + it->second + "' and '" +
                        component.component_name() + "'");
                }
            }
        }
    }

    /// Run Kahn's algorithm over the derived-quantity dependency graph.
    /// Populates topo_ordered_deriveds_ in dependency-first order and rebuilds
    /// global_derived_name_to_index_ to match the topo order.
    /// Throws ComponentError if a cycle is detected.
    void compute_topo_ordered_deriveds() {
        // Step 1: Collect all derived entries in insertion order into a flat list.
        // Build a preliminary insertion-order index for each entry so dependency names
        // can be resolved to integer indices before the topo sort.
        struct FlatEntry {
            std::string name;
            std::size_t global_component_index;
            std::size_t local_derived_index;
            // Dependency names declared via input_derived() before this add_derived() call.
            std::vector<std::string> dependency_names;
        };
        std::vector<FlatEntry> flat;
        for (std::size_t comp_idx = 0; comp_idx < components_.size(); ++comp_idx) {
            const auto& component = components_[comp_idx];
            for (std::size_t local_idx = 0; local_idx < component.num_derived(); ++local_idx) {
                const auto& entry = component.derived_entries()[local_idx];
                flat.push_back(FlatEntry{
                    entry.name,
                    comp_idx,
                    local_idx,
                    entry.dependency_names
                });
            }
        }

        const std::size_t n = flat.size();
        if (n == 0) {
            topo_ordered_deriveds_.clear();
            global_derived_name_to_index_.clear();
            return;
        }

        // Step 2: Build adjacency list (edges point FROM dependency TO dependent).
        // dep_of[j] = list of indices i such that flat[i] depends on flat[j].
        std::vector<std::vector<std::size_t>> dep_of(n);   // j → dependents
        std::vector<std::size_t> in_degree(n, 0);

        for (std::size_t i = 0; i < n; ++i) {
            for (const auto& dep_name : flat[i].dependency_names) {
                auto it = global_derived_name_to_index_.find(dep_name);
                if (it == global_derived_name_to_index_.end()) {
                    throw ComponentError(
                        "ComposedModel: derived quantity '" + flat[i].name +
                        "' declares input_derived(\"" + dep_name +
                        "\") which is not a known derived name");
                }
                const std::size_t j = it->second;  // insertion-order index of dependency
                dep_of[j].push_back(i);
                ++in_degree[i];
            }
        }

        // Step 3: Kahn's algorithm — initialise queue with zero-in-degree nodes.
        std::queue<std::size_t> ready;
        for (std::size_t i = 0; i < n; ++i) {
            if (in_degree[i] == 0) {
                ready.push(i);
            }
        }

        // Step 4: Process queue; build topo order.
        topo_ordered_deriveds_.clear();
        topo_ordered_deriveds_.reserve(n);
        global_derived_name_to_index_.clear();

        while (!ready.empty()) {
            const std::size_t i = ready.front();
            ready.pop();

            // Record the topo position for this entry.
            const std::size_t topo_index = topo_ordered_deriveds_.size();
            global_derived_name_to_index_[flat[i].name] = topo_index;

            // Collect dependency indices in TOPO order (already assigned above for
            // entries that precede this one in topo order).
            std::vector<std::size_t> dep_indices;
            dep_indices.reserve(flat[i].dependency_names.size());
            for (const auto& dep_name : flat[i].dependency_names) {
                dep_indices.push_back(global_derived_name_to_index_.at(dep_name));
            }

            topo_ordered_deriveds_.push_back(ResolvedDerivedEntry{
                flat[i].name,
                flat[i].global_component_index,
                flat[i].local_derived_index,
                std::move(dep_indices)
            });

            // Decrement in-degrees of dependents; enqueue those reaching zero.
            for (const std::size_t dependent : dep_of[i]) {
                if (--in_degree[dependent] == 0) {
                    ready.push(dependent);
                }
            }
        }

        // Step 5: If not all entries were emitted, there is a cycle.
        if (topo_ordered_deriveds_.size() < n) {
            // Collect names of entries still in a cycle for the error message.
            std::vector<std::string> cyclic_names;
            for (std::size_t i = 0; i < n; ++i) {
                if (in_degree[i] > 0) {
                    cyclic_names.push_back(flat[i].name);
                }
            }
            std::string names_str;
            for (std::size_t k = 0; k < cyclic_names.size(); ++k) {
                if (k > 0) { names_str += ", "; }
                names_str += cyclic_names[k];
            }
            throw ComponentError(
                "ComposedModel: cyclic dependency among inline derived quantities: " +
                names_str);
        }
    }

    /// Throw ComponentError if any component's input_state name cannot be resolved
    /// to an owned state published by any component in this ComposedModel.
    void validate_all_inputs_resolvable() const {
        for (const auto& component : components_) {
            for (const auto& input_name : component.input_state_names()) {
                if (global_state_name_to_index_.find(input_name) ==
                    global_state_name_to_index_.end()) {
                    throw ComponentError(
                        "ComposedModel: component '" + component.component_name() +
                        "' has unresolved input state '" + input_name + "'");
                }
            }
        }
    }

    /// Validate preconditions shared by all build() overloads; resolve names if needed.
    void prepare_build() {
        if (!names_resolved_) {
            resolve_names();
        }
        if (!mesh_set_) {
            throw ModelError("ComposedModel::build: call set_mesh() before build()");
        }
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

        // I3 guard: detect dangling input_derived() calls (input_derived() was called
        // without a following add_derived(), leaving pending dependency names unflushed).
        for (const auto& c : components_) {
            if (!c.pending_derived_input_names().empty()) {
                throw ComponentError(
                    "ComposedModel::build: component '" + c.component_name() +
                    "' has " + std::to_string(c.pending_derived_input_names().size()) +
                    " pending input_derived() name(s) that were never consumed by add_derived(). "
                    "Call add_derived() after each input_derived() sequence.");
            }
        }
        // Use the stored double-typed validation lambdas to verify that each component's
        // dynamics returns the correct number of derivatives (== num_owned_states).
        // This runs BEFORE any AD codegen in build_internal_model(), using zero-filled
        // probe inputs of the correct global sizes to catch dimension mismatches early.
        validate_dynamics_dimensions();
        validate_algebraic_dimensions();
    }

    /// For each component that has algebraic entries, verify that
    /// evaluate_algebraic_residual returns the expected scalar value without throwing.
    /// Uses zero-filled probe vectors of correct global sizes.
    void validate_algebraic_dimensions() const {
        const std::size_t num_states_probe     = total_num_states();
        const std::size_t num_algebraics_probe = total_num_algebraic();
        const std::vector<double> probe_x(num_states_probe, 0.0);
        const std::vector<double> probe_u(control_names_.size(), 0.0);
        const std::vector<double> probe_alg(num_algebraics_probe, 0.0);
        constexpr double probe_t = 0.0;
        for (const auto& component : components_) {
            for (std::size_t alg_idx = 0; alg_idx < component.num_algebraic(); ++alg_idx) {
                // evaluate_algebraic_residual must not throw with zero-filled inputs.
                // It returns a scalar; we only check it doesn't throw (dimension mismatch
                // inside the lambda would be a runtime error caught here).
                component.evaluate_algebraic_residual(alg_idx, probe_x, probe_u, probe_alg, probe_t);
            }
        }
    }

    /// For each component that has a dynamics lambda registered, call evaluate_dynamics()
    /// with zero-filled double vectors of the correct global sizes, and verify the returned
    /// vector's size equals num_owned_states() for that component.
    /// Throws ComponentError if there is a dimension mismatch.
    void validate_dynamics_dimensions() const {
        const std::size_t num_states   = total_num_states();
        const std::size_t num_controls = control_names_.size();
        // Probe with zero-filled global x, u, and deriveds vectors.
        const std::vector<double> probe_x(num_states, 0.0);
        const std::vector<double> probe_u(num_controls, 0.0);
        const std::size_t num_deriveds = total_num_derived();
        const std::vector<double> probe_deriveds(num_deriveds, 0.0);
        constexpr double probe_t = 0.0;

        for (const auto& component : components_) {
            if (!component.has_dynamics()) {
                // Components without dynamics do not contribute to dx; skip.
                continue;
            }
            const std::size_t expected = component.num_owned_states();
            const auto result = component.evaluate_dynamics(
                probe_x, probe_u, probe_deriveds, probe_t);
            const std::size_t got = result.size();
            if (got != expected) {
                throw ComponentError(
                    "Component '" + component.component_name() +
                    "': dynamics lambda returns " + std::to_string(got) +
                    " values but component owns " + std::to_string(expected) + " states");
            }
        }
    }

    /// Compute the global state offset for each component (components laid out contiguously
    /// in add_component() registration order).
    std::vector<std::size_t> compute_component_state_offsets() const {
        std::vector<std::size_t> offsets;
        offsets.reserve(components_.size());
        std::size_t offset = 0;
        for (const auto& comp : components_) {
            offsets.push_back(offset);
            offset += comp.num_owned_states();
        }
        return offsets;
    }

    /// Extend build_internal_model to populate algebraic fields of OcpProblem.
    /// Collects algebraic bounds from all components and forwards them through
    /// the transcription pipeline alongside the dynamics and cost functors.
    template <typename DynamicsFn, typename CostFn, typename AlgResFn>
    auto build_internal_model_alg(DynamicsFn dynamics, CostFn cost,
                                   AlgResFn algebraic_residuals,
                                   std::size_t num_alg) const {
        Model internal_model;

        for (const auto& comp : components_) {
            for (const auto& owned_state : comp.owned_states()) {
                auto sh = internal_model.add_state(owned_state.name);
                internal_model.set_state_bounds(sh, owned_state.lower_bound, owned_state.upper_bound);
                if (owned_state.initial_fixed) {
                    internal_model.set_initial_state(sh, owned_state.initial_value);
                }
                if (owned_state.final_fixed) {
                    internal_model.set_final_state(sh, owned_state.final_value);
                }
            }
        }

        for (std::size_t ci = 0; ci < control_names_.size(); ++ci) {
            auto ch = internal_model.add_control(control_names_[ci]);
            internal_model.set_control_bounds(ch, control_lower_[ci], control_upper_[ci]);
        }

        internal_model.set_mesh(mesh_.t_initial, mesh_.t_final, mesh_.num_intervals);

        // Build the base OcpProblem (two-template-param form).
        auto base_ocp = internal_model.build(std::move(dynamics), std::move(cost));

        // Collect algebraic bounds from all components in registration order.
        // Components lay out their algebraic variables contiguously in the same order.
        std::vector<double> alg_lower_bounds;
        std::vector<double> alg_upper_bounds;
        alg_lower_bounds.reserve(num_alg);
        alg_upper_bounds.reserve(num_alg);
        for (const auto& comp : components_) {
            for (const auto& alg_entry : comp.algebraic_entries()) {
                alg_lower_bounds.push_back(alg_entry.lower_bound);
                alg_upper_bounds.push_back(alg_entry.upper_bound);
            }
        }

        // Construct the three-template-param OcpProblem with algebraic fields.
        // Use make_algebraic_ocp to deduce Dyn/Cost types without naming them explicitly
        // (OcpProblem does not expose typedef aliases for its template parameters).
        return make_algebraic_ocp(std::move(base_ocp), std::move(algebraic_residuals),
                                  num_alg,
                                  std::move(alg_lower_bounds),
                                  std::move(alg_upper_bounds));
    }

    /// Helper to construct OcpProblem<Dyn, Cost, AlgRes> from a base OcpProblem<Dyn, Cost>
    /// and algebraic metadata, without naming the Dyn/Cost types explicitly.
    /// Template argument deduction infers DynamicsFn and CostFn from base_ocp.
    template <typename DynamicsFn, typename CostFn, typename AlgResFn>
    static auto make_algebraic_ocp(
            transcription::OcpProblem<DynamicsFn, CostFn> base_ocp,
            AlgResFn algebraic_residuals,
            std::size_t num_alg,
            std::vector<double> alg_lower_bounds,
            std::vector<double> alg_upper_bounds) {
        // C++17 aggregate initialization: members listed in exact OcpProblem struct order:
        // num_states, num_controls, dynamics, cost, mesh,
        // state_lower, state_upper, control_lower, control_upper,
        // initial_state, initial_state_fixed, final_state, final_state_fixed,
        // num_algebraic, algebraic_residuals_functor,
        // algebraic_lower_bounds, algebraic_upper_bounds.
        transcription::OcpProblem<DynamicsFn, CostFn, AlgResFn> ocp{
            base_ocp.num_states,
            base_ocp.num_controls,
            std::move(base_ocp.dynamics),
            std::move(base_ocp.cost),
            base_ocp.mesh,
            std::move(base_ocp.state_lower),
            std::move(base_ocp.state_upper),
            std::move(base_ocp.control_lower),
            std::move(base_ocp.control_upper),
            std::move(base_ocp.initial_state),
            std::move(base_ocp.initial_state_fixed),
            std::move(base_ocp.final_state),
            std::move(base_ocp.final_state_fixed),
            num_alg,
            std::move(algebraic_residuals),
            std::move(alg_lower_bounds),
            std::move(alg_upper_bounds)
        };
        return ocp;
    }

    /// Construct an internal Model populated with metadata from all components and controls,
    /// then call internal_model.build(dynamics, cost) and return the resulting OcpProblem.
    ///
    /// This keeps all bound-forwarding logic in one place, shared by all build() overloads.
    template <typename DynamicsFn, typename CostFn>
    auto build_internal_model(DynamicsFn dynamics, CostFn cost) const {
        Model internal_model;

        // Register states in global order (component 0 states first, then component 1, …).
        for (const auto& comp : components_) {
            for (const auto& owned_state : comp.owned_states()) {
                auto sh = internal_model.add_state(owned_state.name);
                internal_model.set_state_bounds(sh, owned_state.lower_bound, owned_state.upper_bound);
                if (owned_state.initial_fixed) {
                    internal_model.set_initial_state(sh, owned_state.initial_value);
                }
                if (owned_state.final_fixed) {
                    internal_model.set_final_state(sh, owned_state.final_value);
                }
            }
        }

        // Register controls in registration order.
        for (std::size_t ci = 0; ci < control_names_.size(); ++ci) {
            auto ch = internal_model.add_control(control_names_[ci]);
            internal_model.set_control_bounds(ch, control_lower_[ci], control_upper_[ci]);
        }

        internal_model.set_mesh(mesh_.t_initial, mesh_.t_final, mesh_.num_intervals);

        return internal_model.build(std::move(dynamics), std::move(cost));
    }

    std::vector<Component> components_;
    std::vector<std::string> control_names_;
    std::vector<double> control_lower_;
    std::vector<double> control_upper_;
    std::unordered_map<std::string, std::size_t> global_state_name_to_index_;
    std::unordered_map<std::string, std::size_t> global_derived_name_to_index_;
    std::vector<ResolvedDerivedEntry> topo_ordered_deriveds_;
    bool names_resolved_ = false;
    bool mesh_set_ = false;
    transcription::Mesh mesh_{0.0, 1.0, 1};
};

}  // namespace goss::model
