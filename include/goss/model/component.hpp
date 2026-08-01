// include/goss/model/component.hpp
#pragma once
#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "goss/model/errors.hpp"
#include "goss/model/handles.hpp"
#include "goss/transcription/transcription.hpp"  // kInf

namespace goss::model {

// Sentinel for unresolved input-state handles (replaced at ComposedModel::build()).
constexpr std::size_t kUnresolvedIndex = std::numeric_limits<std::size_t>::max();

/// Opaque handle to an inline derived quantity, analogous to StateHandle.
struct DerivedHandle {
    std::size_t index;
    constexpr operator std::size_t() const noexcept { return index; }
};

/// Metadata for one owned state within a Component.
struct OwnedStateEntry {
    std::string name;
    double lower_bound;
    double upper_bound;
    double initial_value;
    bool initial_fixed;
    double final_value;
    bool final_fixed;
};

/// Metadata for an inline derived quantity.
struct DerivedEntry {
    std::string name;
    // Validation lambda: (x, u, deriveds_so_far, t) → scalar double.
    // Sized: x uses GLOBAL state indices (post-resolution); deriveds_so_far uses
    // local derived indices in topological order.
    std::function<double(
        const std::vector<double>&,  // global x
        const std::vector<double>&,  // global u
        const std::vector<double>&,  // already-evaluated deriveds (topo order)
        double)> validation_fn;
    /// Names of other derived quantities this entry explicitly depends on (declared via
    /// input_derived() before the add_derived() call that produced this entry).
    std::vector<std::string> dependency_names;
};

class Component {
 public:
    /// Construct a named component.
    explicit Component(std::string component_name)
        : component_name_(std::move(component_name)) {}

    /// Declare that this component reads a state owned by another component/parent.
    /// Returns a handle with kUnresolvedIndex (resolved at ComposedModel::build()).
    StateHandle input_state(const std::string& state_name) {
        input_state_names_.push_back(state_name);
        return StateHandle{kUnresolvedIndex};
    }

    /// Declare an owned state with default bounds [-kInf, kInf].
    StateHandle add_state(const std::string& state_name) {
        ensure_name_unique_in_component(state_name);
        const std::size_t local_index = owned_states_.size();
        owned_states_.push_back(OwnedStateEntry{
            state_name,
            -transcription::kInf,  // lower_bound
            transcription::kInf,   // upper_bound
            0.0,                   // initial_value
            false,                 // initial_fixed
            0.0,                   // final_value
            false                  // final_fixed
        });
        return StateHandle{local_index};
    }

    /// Set bounds on an owned state.
    void set_state_bounds(StateHandle owned_state_handle, double lower, double upper) {
        check_owned_state_handle(owned_state_handle.index, "set_state_bounds");
        owned_states_[owned_state_handle.index].lower_bound = lower;
        owned_states_[owned_state_handle.index].upper_bound = upper;
    }

    /// Pin the initial value of an owned state.
    void set_initial_state(StateHandle owned_state_handle, double value) {
        check_owned_state_handle(owned_state_handle.index, "set_initial_state");
        owned_states_[owned_state_handle.index].initial_value = value;
        owned_states_[owned_state_handle.index].initial_fixed = true;
    }

    /// Pin the final value of an owned state.
    void set_final_state(StateHandle owned_state_handle, double value) {
        check_owned_state_handle(owned_state_handle.index, "set_final_state");
        owned_states_[owned_state_handle.index].final_value = value;
        owned_states_[owned_state_handle.index].final_fixed = true;
    }

    /// Declare that the NEXT derived quantity added to this component depends on a named
    /// derived quantity (by name). Must be called BEFORE the add_derived() call for the
    /// dependent entry. The accumulated names are flushed into the entry on add_derived().
    /// Returns a DerivedHandle with kUnresolvedIndex; the name is staged in a pending list.
    DerivedHandle input_derived(const std::string& derived_name) {
        pending_derived_input_names_.push_back(derived_name);
        return DerivedHandle{kUnresolvedIndex};
    }

    /// Register an inline derived quantity for validation (double-typed lambda).
    /// Any names staged via input_derived() since the last add_derived() are attached to
    /// this entry as declared dependencies, then the pending list is cleared.
    /// Returns a DerivedHandle for use inside this component's dynamics lambda.
    DerivedHandle add_derived(
        const std::string& derived_name,
        std::function<double(const std::vector<double>&,
                             const std::vector<double>&,
                             const std::vector<double>&,
                             double)> validation_lambda) {
        ensure_name_unique_in_component(derived_name);
        const std::size_t local_index = derived_entries_.size();
        derived_entries_.push_back(DerivedEntry{
            derived_name,
            std::move(validation_lambda),
            std::move(pending_derived_input_names_)  // flush dependency names into the entry
        });
        pending_derived_input_names_.clear();
        return DerivedHandle{local_index};
    }

    /// Register this component's dynamics (double-typed, for validation only).
    /// The lambda must return a vector of size == num_owned_states().
    void set_dynamics(
        std::function<std::vector<double>(
            const std::vector<double>&,
            const std::vector<double>&,
            const std::vector<double>&,
            double)> validation_lambda) {
        dynamics_fn_ = std::move(validation_lambda);
        has_dynamics_ = true;
    }

    /// Optional: register a running cost contribution (double-typed, for validation only).
    void set_cost(
        std::function<double(const std::vector<double>&,
                             const std::vector<double>&,
                             const std::vector<double>&,
                             double)> validation_lambda) {
        cost_fn_ = std::move(validation_lambda);
        has_cost_ = true;
    }

    // ---- Accessors for ComposedModel ----

    const std::string& component_name() const { return component_name_; }

    std::size_t num_owned_states() const { return owned_states_.size(); }

    std::size_t num_derived() const { return derived_entries_.size(); }

    const std::vector<OwnedStateEntry>& owned_states() const { return owned_states_; }

    const std::vector<DerivedEntry>& derived_entries() const { return derived_entries_; }

    const std::vector<std::string>& input_state_names() const { return input_state_names_; }

    /// Returns names of derived quantities declared as inputs to any derived entry via
    /// input_derived() calls that are still pending (not yet flushed into an entry).
    const std::vector<std::string>& pending_derived_input_names() const {
        return pending_derived_input_names_;
    }

    bool has_dynamics() const { return has_dynamics_; }

    bool has_cost() const { return has_cost_; }

    /// Invoke the double-path dynamics lambda (validation use only).
    std::vector<double> evaluate_dynamics(
        const std::vector<double>& global_x,
        const std::vector<double>& global_u,
        const std::vector<double>& deriveds,
        double t) const {
        if (!has_dynamics_) {
            throw ComponentError(
                "Component '" + component_name_ + "': evaluate_dynamics called but no dynamics set");
        }
        return dynamics_fn_(global_x, global_u, deriveds, t);
    }

    /// Invoke the double-path cost lambda (validation use only).
    double evaluate_cost(
        const std::vector<double>& global_x,
        const std::vector<double>& global_u,
        const std::vector<double>& deriveds,
        double t) const {
        if (!has_cost_) {
            throw ComponentError(
                "Component '" + component_name_ + "': evaluate_cost called but no cost set");
        }
        return cost_fn_(global_x, global_u, deriveds, t);
    }

 private:
    /// Throw ComponentError if local_index is out of range for owned_states_.
    void check_owned_state_handle(std::size_t local_index, const char* caller) const {
        if (local_index >= owned_states_.size()) {
            throw ComponentError(
                std::string(caller) + ": owned state handle index " +
                std::to_string(local_index) + " out of range (component '" +
                component_name_ + "' has " + std::to_string(owned_states_.size()) +
                " owned states)");
        }
    }

    /// Throw ComponentError if the given name is already used by any owned state or derived entry.
    void ensure_name_unique_in_component(const std::string& name) const {
        for (const auto& entry : owned_states_) {
            if (entry.name == name) {
                throw ComponentError(
                    "Component '" + component_name_ + "': duplicate name '" + name + "'");
            }
        }
        for (const auto& entry : derived_entries_) {
            if (entry.name == name) {
                throw ComponentError(
                    "Component '" + component_name_ + "': duplicate name '" + name + "'");
            }
        }
    }

    std::string component_name_;
    std::vector<OwnedStateEntry> owned_states_;
    std::vector<DerivedEntry> derived_entries_;
    /// Names declared as inputs (states owned by other components).
    std::vector<std::string> input_state_names_;
    /// Staging area: derived names declared via input_derived() since the last add_derived().
    /// Flushed into the next DerivedEntry's dependency_names on add_derived().
    std::vector<std::string> pending_derived_input_names_;

    // Double-typed validation lambdas (AD path uses generic lambdas at build() time).
    std::function<std::vector<double>(const std::vector<double>&,
                                      const std::vector<double>&,
                                      const std::vector<double>&,
                                      double)> dynamics_fn_;
    std::function<double(const std::vector<double>&,
                          const std::vector<double>&,
                          const std::vector<double>&,
                          double)> cost_fn_;
    bool has_dynamics_ = false;
    bool has_cost_ = false;
};

}  // namespace goss::model
