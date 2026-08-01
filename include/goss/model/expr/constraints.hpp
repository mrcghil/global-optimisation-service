// include/goss/model/expr/constraints.hpp
// Comparison operators that LOWER directly to Model setters — no solver-level
// constraint mechanism required. Only StateHandle/ControlHandle comparison with
// double literals is supported; general nonlinear path constraints (e.g.
// (q + 1.0) >= 0.0) are deferred (see task-4-brief.md for the extension point).
//
// ADL NOTE: The comparison operators (>=, <=, ==) are placed in namespace
// goss::model (same as StateHandle, ControlHandle, BoundaryPoint) so that
// Argument-Dependent Lookup (ADL) finds them when a caller writes `q >= 0.0`
// without any using-directive. The apply_bound / apply_boundary helpers and
// the constraint structs live in goss::model::expr as the DSL sub-library.
#pragma once
#include "goss/model/handles.hpp"
#include "goss/model/model.hpp"
#include "goss/transcription/transcription.hpp"  // goss::transcription::kInf

namespace goss::model::expr {

/// Represents a box constraint on a single state: lower_bound <= state <= upper_bound.
/// Constructed by operator>= / operator<= on a StateHandle; lowered to
/// Model::set_state_bounds by apply_bound().
struct BoundConstraint {
    goss::model::StateHandle state_handle;
    double lower_bound;
    double upper_bound;
};

/// Represents a box constraint on a single control: lower_bound <= control <= upper_bound.
/// Constructed by operator>= / operator<= on a ControlHandle; lowered to
/// Model::set_control_bounds by apply_bound().
struct ControlBoundConstraint {
    goss::model::ControlHandle control_handle;
    double lower_bound;
    double upper_bound;
};

/// Represents a fixed boundary condition on a state.
/// Constructed by operator==(BoundaryPoint, double); lowered to
/// Model::set_initial_state or Model::set_final_state by apply_boundary().
struct BoundaryConstraint {
    goss::model::StateHandle state_handle;
    double fixed_value;
    enum class Kind { Initial, Final };
    Kind kind;
};

// --- Lowering helpers: apply constraints to a Model ---
// These are the "lowering" step: they translate DSL constraint structs into
// the imperative Model API calls that the transcription layer reads.

/// Lower a BoundConstraint to Model::set_state_bounds.
inline void apply_bound(goss::model::Model& model, const BoundConstraint& constraint) {
    model.set_state_bounds(constraint.state_handle, constraint.lower_bound, constraint.upper_bound);
}

/// Lower a ControlBoundConstraint to Model::set_control_bounds.
inline void apply_bound(goss::model::Model& model, const ControlBoundConstraint& constraint) {
    model.set_control_bounds(constraint.control_handle, constraint.lower_bound, constraint.upper_bound);
}

/// Lower a BoundaryConstraint to Model::set_initial_state or Model::set_final_state.
/// The kind enum selects which boundary the fixed value applies to.
inline void apply_boundary(goss::model::Model& model, const BoundaryConstraint& constraint) {
    if (constraint.kind == BoundaryConstraint::Kind::Initial) {
        model.set_initial_state(constraint.state_handle, constraint.fixed_value);
    } else {
        model.set_final_state(constraint.state_handle, constraint.fixed_value);
    }
}

}  // namespace goss::model::expr

// --- Comparison operators placed in goss::model for ADL ---
// These operators live in the same namespace as StateHandle, ControlHandle,
// and BoundaryPoint so that ADL finds them without any using-directive.
// They return types from goss::model::expr — that is intentional and fine.
namespace goss::model {

/// q >= value  =>  BoundConstraint{q, value, +kInf}
inline goss::model::expr::BoundConstraint operator>=(
        goss::model::StateHandle state_handle, double lower_value) {
    return goss::model::expr::BoundConstraint{
        state_handle, lower_value, goss::transcription::kInf};
}

/// q <= value  =>  BoundConstraint{q, -kInf, value}
inline goss::model::expr::BoundConstraint operator<=(
        goss::model::StateHandle state_handle, double upper_value) {
    return goss::model::expr::BoundConstraint{
        state_handle, -goss::transcription::kInf, upper_value};
}

/// u >= value  =>  ControlBoundConstraint{u, value, +kInf}
inline goss::model::expr::ControlBoundConstraint operator>=(
        goss::model::ControlHandle control_handle, double lower_value) {
    return goss::model::expr::ControlBoundConstraint{
        control_handle, lower_value, goss::transcription::kInf};
}

/// u <= value  =>  ControlBoundConstraint{u, -kInf, value}
inline goss::model::expr::ControlBoundConstraint operator<=(
        goss::model::ControlHandle control_handle, double upper_value) {
    return goss::model::expr::ControlBoundConstraint{
        control_handle, -goss::transcription::kInf, upper_value};
}

/// q.initial() == value  =>  BoundaryConstraint{q, value, Kind::Initial}
/// q.final()   == value  =>  BoundaryConstraint{q, value, Kind::Final}
/// Maps BoundaryPoint::Kind::Initial -> BoundaryConstraint::Kind::Initial,
/// and BoundaryPoint::Kind::Final -> BoundaryConstraint::Kind::Final.
inline goss::model::expr::BoundaryConstraint operator==(
        goss::model::BoundaryPoint boundary_point, double fixed_value) {
    const goss::model::expr::BoundaryConstraint::Kind constraint_kind =
        (boundary_point.kind == goss::model::BoundaryPoint::Kind::Initial)
            ? goss::model::expr::BoundaryConstraint::Kind::Initial
            : goss::model::expr::BoundaryConstraint::Kind::Final;
    return goss::model::expr::BoundaryConstraint{
        boundary_point.state_handle, fixed_value, constraint_kind};
}

}  // namespace goss::model
