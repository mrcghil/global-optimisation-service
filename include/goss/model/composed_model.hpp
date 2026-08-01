// include/goss/model/composed_model.hpp
// Stub: Task 3 — name resolution + global state/derived index mapping.
// build() is added in Task 5.
#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include "goss/model/component.hpp"
#include "goss/model/errors.hpp"
#include "goss/model/handles.hpp"
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

        // Build global derived index map (flat, topo-sort deferred to Task 4).
        std::size_t derived_offset = 0;
        for (std::size_t comp_idx = 0; comp_idx < components_.size(); ++comp_idx) {
            const auto& component = components_[comp_idx];
            for (std::size_t local_idx = 0; local_idx < component.num_derived(); ++local_idx) {
                const auto& entry = component.derived_entries()[local_idx];
                global_derived_name_to_index_[entry.name] = derived_offset;
                topo_ordered_deriveds_.push_back(ResolvedDerivedEntry{
                    entry.name,
                    comp_idx,
                    local_idx,
                    {}  // dependency resolution deferred to Task 4
                });
                ++derived_offset;
            }
        }

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

    // build() is added in Task 5.

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
