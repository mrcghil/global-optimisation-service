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
    /// Global derived indices this entry depends on (for topo-sort in Task 4).
    std::vector<std::size_t> dependency_global_derived_indices;
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

    // ---- Task 5: build() overloads ----
    //
    // Convention for argument ordering:
    //   1. All derived-expression generic lambdas in topological order
    //      (signature: (const auto& x, const auto& u, const auto& deriveds_so_far, auto t) -> T)
    //   2. One dynamics generic lambda per component in add_component() registration order
    //      (signature: (const auto& x, const auto& u, const auto& deriveds, auto t) -> vector<T>)
    //   3. One combined cost generic lambda
    //      (signature: (const auto& x, const auto& u, const auto& deriveds, auto t) -> T)
    //
    // All lambdas are captured by VALUE into a std::tuple inside the assembled combined closure;
    // no std::function is in the AD path.

    /// Overload for 0 derived expressions and exactly 1 component dynamics lambda.
    ///
    /// Combined dynamics functor: evaluates component_0_dyn, writes into global dx slots.
    /// Combined cost functor: forwards (x, u, {}, t) to the cost lambda.
    template <typename Dyn0Fn, typename CostFn>
    auto build(Dyn0Fn component_0_dyn, CostFn combined_cost) {
        prepare_build();

        // Snapshot per-component state layout (global offsets) before moving into lambdas.
        // Components are laid out contiguously in add_component() order.
        const std::size_t num_states  = total_num_states();

        // Per-component state offsets (component i starts at component_state_offsets[i]).
        std::vector<std::size_t> component_state_offsets = compute_component_state_offsets();
        // For 1 component, grab offset directly.
        const std::size_t comp0_offset = component_state_offsets[0];
        const std::size_t comp0_nstates = components_[0].num_owned_states();

        // Assemble combined dynamics — NO std::function in this lambda.
        // Both component_0_dyn and combined_cost are generic lambdas (template operator());
        // capturing them by value preserves full template instantiation at CppAD recording time.
        auto combined_dynamics = [
            component_0_dyn,
            num_states,
            comp0_offset,
            comp0_nstates
        ](const auto& x, const auto& u, auto t) {
            using T = typename std::decay_t<decltype(x)>::value_type;
            // No derived quantities in this overload: pass empty deriveds vector.
            std::vector<T> deriveds;
            std::vector<T> dx(num_states);
            auto comp0_dx = component_0_dyn(x, u, deriveds, t);
            for (std::size_t i = 0; i < comp0_nstates; ++i) {
                dx[comp0_offset + i] = comp0_dx[i];
            }
            return dx;
        };

        // Wrap cost so the stored signature matches what Model::build expects:
        // OcpProblem cost signature is (x, u, t) but our combined cost takes (x, u, deriveds, t).
        auto ocp_cost = [combined_cost](const auto& x, const auto& u, auto t) {
            using T = typename std::decay_t<decltype(x)>::value_type;
            std::vector<T> deriveds;
            return combined_cost(x, u, deriveds, t);
        };

        return build_internal_model(std::move(combined_dynamics), std::move(ocp_cost));
    }

    /// Overload for exactly 1 derived-expression lambda and 1 component dynamics lambda.
    ///
    /// Combined dynamics functor:
    ///   1. Evaluates derived_expr_0(x, u, {}, t) → deriveds[0]   (topo index 0, no deps)
    ///   2. Calls component_0_dyn(x, u, deriveds, t) → comp0_dx
    ///   3. Writes comp0_dx into global dx at component 0's state offset.
    template <typename DerivedExpr0Fn, typename Dyn0Fn, typename CostFn>
    auto build(DerivedExpr0Fn derived_expr_0, Dyn0Fn component_0_dyn, CostFn combined_cost) {
        prepare_build();

        const std::size_t num_states   = total_num_states();
        const std::size_t num_deriveds = 1;  // exactly one derived expression

        std::vector<std::size_t> component_state_offsets = compute_component_state_offsets();

        // Determine which component owns state slots — find the first component with owned states.
        // For the 1-component-with-states layout, walk components to find it.
        std::size_t comp0_offset  = 0;
        std::size_t comp0_nstates = 0;
        for (std::size_t ci = 0; ci < components_.size(); ++ci) {
            if (components_[ci].num_owned_states() > 0) {
                comp0_offset  = component_state_offsets[ci];
                comp0_nstates = components_[ci].num_owned_states();
                break;
            }
        }

        // Assemble combined dynamics lambda — generic, captures all lambdas by value.
        // No std::function in this path; both derived_expr_0 and component_0_dyn are
        // generic lambdas whose operator() is a function template.
        auto combined_dynamics = [
            derived_expr_0,
            component_0_dyn,
            num_states,
            num_deriveds,
            comp0_offset,
            comp0_nstates
        ](const auto& x, const auto& u, auto t) {
            using T = typename std::decay_t<decltype(x)>::value_type;
            // Step 1: evaluate all derived quantities in topological order.
            // For topo index 0 with no declared deps, deriveds_so_far is empty.
            std::vector<T> deriveds(num_deriveds);
            {
                std::vector<T> deriveds_so_far;  // empty: topo index 0 has no dependencies
                deriveds[0] = derived_expr_0(x, u, deriveds_so_far, t);
            }
            // Step 2: assemble global dx vector.
            std::vector<T> dx(num_states);
            auto comp0_dx = component_0_dyn(x, u, deriveds, t);
            for (std::size_t i = 0; i < comp0_nstates; ++i) {
                dx[comp0_offset + i] = comp0_dx[i];
            }
            return dx;
        };

        // Wrap cost to conform to OcpProblem's (x, u, t) signature.
        // Re-evaluate derived quantities in topo order (mirroring the dynamics path) so
        // the cost lambda receives the real derived values, not a zero-filled placeholder.
        auto ocp_cost = [combined_cost, derived_expr_0, num_deriveds](
                const auto& x, const auto& u, auto t) {
            using T = typename std::decay_t<decltype(x)>::value_type;
            std::vector<T> deriveds(num_deriveds);
            {
                std::vector<T> deriveds_so_far;  // topo index 0 has no dependencies
                deriveds[0] = derived_expr_0(x, u, deriveds_so_far, t);
            }
            return combined_cost(x, u, deriveds, t);
        };

        return build_internal_model(std::move(combined_dynamics), std::move(ocp_cost));
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
        // A composed OCP needs at least one component that owns a state; without one the
        // combined dynamics would write nothing and produce a silently-empty problem.
        bool any_state_owner = false;
        for (const auto& c : components_) {
            if (c.num_owned_states() > 0) {
                any_state_owner = true;
                break;
            }
        }
        if (!any_state_owner) {
            throw ComponentError(
                "ComposedModel::build: no component owns any state; "
                "a composed model needs at least one state.");
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
