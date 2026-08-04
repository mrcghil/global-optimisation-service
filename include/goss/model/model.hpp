// include/goss/model/model.hpp
#pragma once
#include <cstddef>
#include <string>
#include <utility>
#include <vector>
#include "goss/model/errors.hpp"
#include "goss/model/handles.hpp"
#include "goss/model/parameter.hpp"
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

    /// Declares a model parameter with a name, default value, and optional
    /// bounds. Returns a ParameterHandle whose index can be used to index into
    /// a parameter vector at solve time.
    ///
    /// Throws ModelError if:
    ///   - name collides with an existing state, control, or parameter name,
    ///   - lower_bound > upper_bound,
    ///   - default_value is outside [lower_bound, upper_bound].
    ParameterHandle add_parameter(const std::string& name,
                                  double default_value,
                                  double lower_bound = -transcription::kInf,
                                  double upper_bound =  transcription::kInf) {
        ensure_unique_name(name);
        if (lower_bound > upper_bound)
            throw ModelError("add_parameter: lower > upper for parameter '" + name + "'");
        if (default_value < lower_bound || default_value > upper_bound)
            throw ModelError("add_parameter: default for parameter '" + name +
                             "' is outside its bounds");
        const std::size_t index = parameter_specs_.size();
        parameter_specs_.push_back(ParameterSpec{name, default_value, lower_bound, upper_bound});
        return ParameterHandle{index};
    }

    std::size_t num_parameters() const { return parameter_specs_.size(); }

    const std::string& parameter_name(std::size_t index) const {
        if (index >= parameter_specs_.size())
            throw ModelError("Model::parameter_name: index out of range");
        return parameter_specs_[index].name;
    }

    /// Returns a ParameterValidator built from the currently declared parameters.
    /// The validator checks size, NaN, and per-parameter bounds, throwing ModelError
    /// with a message that names the offending parameter and its bound.
    ParameterValidator parameter_validator() const { return ParameterValidator(parameter_specs_); }

    const std::string& state_name(std::size_t index) const {
        if (index >= state_names_.size()) {
            throw ModelError("Model::state_name: index out of range");
        }
        return state_names_[index];
    }

    const std::string& control_name(std::size_t index) const {
        if (index >= control_names_.size()) {
            throw ModelError("Model::control_name: index out of range");
        }
        return control_names_[index];
    }

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

    /// Assembles a transcription::OcpProblem from the declared metadata.
    ///
    /// Throws ModelError if:
    ///   - no states declared,
    ///   - set_mesh() was not called,
    ///   - mesh parameters are invalid (e.g. t_final <= t_initial, num_intervals == 0),
    ///   - a pinned boundary value violates the state's declared bounds.
    template <typename DynamicsFn, typename CostFn>
    transcription::OcpProblem<DynamicsFn, CostFn> build(DynamicsFn dynamics, CostFn cost) const {
        if (state_names_.empty()) throw ModelError("Model::build: a model needs at least one state");
        if (!mesh_set_) throw ModelError("Model::build: call set_mesh() before build()");
        // Wrap mesh validation so any TranscriptionError surfaces as ModelError,
        // ensuring build()'s contract is uniform (only ModelError escapes).
        try {
            mesh_.validate();
        } catch (const transcription::TranscriptionError& mesh_error) {
            throw ModelError(std::string("Model::build: invalid mesh: ") + mesh_error.what());
        }

        // Validate that pinned boundary values respect state bounds; an
        // inconsistency here causes a silently infeasible problem in the solver.
        for (std::size_t i = 0; i < state_names_.size(); ++i) {
            if (initial_fixed_[i] && (initial_value_[i] < state_lower_[i] || initial_value_[i] > state_upper_[i]))
                throw ModelError("Model::build: initial value for state '" + state_names_[i] + "' violates its bounds");
            if (final_fixed_[i] && (final_value_[i] < state_lower_[i] || final_value_[i] > state_upper_[i]))
                throw ModelError("Model::build: final value for state '" + state_names_[i] + "' violates its bounds");
        }

        // Convert std::vector<bool> to std::vector<double> first;
        // nonzero => pinned (transcription convention). std::vector<bool> cannot
        // be copy-assigned to std::vector<double> directly (specialisation).
        const std::size_t ns = state_names_.size();
        std::vector<double> init_fixed(ns);
        std::vector<double> final_fixed(ns);
        for (std::size_t i = 0; i < ns; ++i) {
            init_fixed[i]  = initial_fixed_[i] ? 1.0 : 0.0;
            final_fixed[i] = final_fixed_[i]   ? 1.0 : 0.0;
        }

        // Build per-parameter vectors from parameter_specs_ before the aggregate.
        // We do this outside the initializer list for readability and to avoid
        // relying on evaluation order of function arguments.
        std::vector<std::string> param_names;
        std::vector<double> param_defaults;
        std::vector<double> param_lower;
        std::vector<double> param_upper;
        param_names.reserve(parameter_specs_.size());
        param_defaults.reserve(parameter_specs_.size());
        param_lower.reserve(parameter_specs_.size());
        param_upper.reserve(parameter_specs_.size());
        for (const ParameterSpec& spec : parameter_specs_) {
            param_names.push_back(spec.name);
            param_defaults.push_back(spec.default_value);
            param_lower.push_back(spec.lower_bound);
            param_upper.push_back(spec.upper_bound);
        }

        // Use aggregate initialization so the lambdas are move-constructed rather
        // than default-constructed. Capturing lambdas have a deleted default
        // constructor, so we must not default-construct OcpProblem and then assign.
        //
        // Field order matches OcpProblem struct declaration EXACTLY (26 fields):
        //   Fields  1-13 : base (num_states..final_state_fixed)
        //   Fields 14-17 : algebraic defaults (0, NoAlgebraicResiduals{}, {}, {})
        //   Fields 18-21 : path-constraint defaults (0, {}, {}, NoPathConstraints{})
        //   Fields 22-26 : parameter metadata (num_parameters, names, defaults, lower, upper)
        return transcription::OcpProblem<DynamicsFn, CostFn>{
            // --- 13 base fields ---
            state_names_.size(),
            control_names_.size(),
            std::move(dynamics),
            std::move(cost),
            mesh_,
            state_lower_,
            state_upper_,
            control_lower_,
            control_upper_,
            initial_value_,
            std::move(init_fixed),
            final_value_,
            std::move(final_fixed),
            // --- 4 algebraic defaults (fields 14-17) ---
            std::size_t{0},
            transcription::NoAlgebraicResiduals{},
            std::vector<double>{},  // algebraic_lower_bounds (empty — no algebraic vars)
            std::vector<double>{},  // algebraic_upper_bounds (empty — no algebraic vars)
            // --- 4 path-constraint defaults (fields 18-21) ---
            std::size_t{0},
            std::vector<double>{},  // path_constraint_lower (empty — no path constraints)
            std::vector<double>{},  // path_constraint_upper (empty — no path constraints)
            transcription::NoPathConstraints{},
            // --- 5 parameter fields (fields 22-26) ---
            parameter_specs_.size(),
            std::move(param_names),
            std::move(param_defaults),
            std::move(param_lower),
            std::move(param_upper)
        };
    }

    /// Build an OcpProblem with path constraints. Called by ExprModel::build()
    /// when one or more path constraints have been registered via with_path_constraint().
    ///
    /// Reuses build()'s validation logic by first constructing a 2-param OcpProblem,
    /// then extending it to 4-param by inserting algebraic-field defaults (fields 14-17)
    /// and path-constraint fields (fields 18-21) via aggregate initialization.
    ///
    /// AD-safety: PathConstraintFn is passed by value and stored in OcpProblem::path_constraints
    /// as a concrete template parameter — no std::function, no virtual dispatch.
    template <typename DynamicsFn, typename CostFn, typename PathConstraintFn>
    transcription::OcpProblem<DynamicsFn, CostFn,
                               transcription::NoAlgebraicResiduals, PathConstraintFn>
    build_with_path_constraints(
            DynamicsFn          dynamics_functor,
            CostFn              cost_functor,
            PathConstraintFn    path_constraint_functor,
            std::size_t         num_path_constraints_arg,
            std::vector<double> path_constraint_lower_arg,
            std::vector<double> path_constraint_upper_arg) const {
        // Reuse existing build() for all validation (state count, mesh, bounds checks).
        // Returns OcpProblem<DynamicsFn, CostFn> == <Dyn, Cost, NoAlgebraicResiduals,
        // NoPathConstraints>, which we strip to pull the 13 base fields (and param fields).
        auto base_ocp = build(std::move(dynamics_functor), std::move(cost_functor));

        // Aggregate-init the 4-param OcpProblem.
        // Field order matches ocp_problem.hpp EXACTLY (26 fields total):
        //   Fields  1-13 : base (num_states...final_state_fixed)
        //   Fields 14-17 : algebraic defaults (0, NoAlgebraicResiduals{}, {}, {})
        //   Fields 18-21 : path constraints (count, lower, upper, functor)
        //   Fields 22-26 : parameter metadata (num_parameters, names, defaults, lower, upper)
        return transcription::OcpProblem<DynamicsFn, CostFn,
                                         transcription::NoAlgebraicResiduals,
                                         PathConstraintFn>{
            // --- 13 base fields ---
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
            // --- 4 algebraic defaults (fields 14-17) ---
            // num_algebraic == 0: no DAE algebraic variables in this problem.
            std::size_t{0},
            transcription::NoAlgebraicResiduals{},
            std::vector<double>{},  // algebraic_lower_bounds (empty — no algebraic vars)
            std::vector<double>{},  // algebraic_upper_bounds (empty — no algebraic vars)
            // --- 4 path-constraint fields (fields 18-21) ---
            num_path_constraints_arg,
            std::move(path_constraint_lower_arg),
            std::move(path_constraint_upper_arg),
            std::move(path_constraint_functor),
            // --- 5 parameter fields (fields 22-26) — carry through from base_ocp ---
            base_ocp.num_parameters,
            std::move(base_ocp.parameter_names),
            std::move(base_ocp.parameter_defaults),
            std::move(base_ocp.parameter_lower),
            std::move(base_ocp.parameter_upper)
        };
    }

 private:
    void check_state_index(std::size_t i, const char* who) const {
        if (i >= state_names_.size())
            throw ModelError(std::string(who) + ": state index out of range");
    }
    void check_control_index(std::size_t i, const char* who) const {
        if (i >= control_names_.size())
            throw ModelError(std::string(who) + ": control index out of range");
    }

    /// Throws ModelError if name is already registered as a state, control, or parameter.
    void ensure_unique_name(const std::string& name) const {
        for (const auto& existing : state_names_) {
            if (existing == name) {
                throw ModelError("Model: duplicate name '" + name + "' (already a state)");
            }
        }
        for (const auto& existing : control_names_) {
            if (existing == name) {
                throw ModelError("Model: duplicate name '" + name + "' (already a control)");
            }
        }
        for (const auto& existing_spec : parameter_specs_) {
            if (existing_spec.name == name) {
                throw ModelError("Model: duplicate name '" + name + "' (already a parameter)");
            }
        }
    }

    std::vector<std::string> state_names_;
    std::vector<std::string> control_names_;
    std::vector<double> state_lower_;
    std::vector<double> state_upper_;
    std::vector<double> control_lower_;
    std::vector<double> control_upper_;
    std::vector<double> initial_value_;
    std::vector<bool>   initial_fixed_;
    std::vector<double> final_value_;
    std::vector<bool>   final_fixed_;

    std::vector<ParameterSpec> parameter_specs_;

    bool                 mesh_set_ = false;
    transcription::Mesh  mesh_{0.0, 1.0, 1};
};

}  // namespace goss::model
