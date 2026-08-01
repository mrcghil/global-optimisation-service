#pragma once
#include <cstddef>
namespace goss::model {

/// Opaque handle to a declared state. Implicitly converts to std::size_t so it
/// can index the x-vector inside dynamics/cost lambdas (`x[q]`). It is a struct
/// (not a bare size_t) so the expression DSL can attach operators to it
/// (e.g. q >= 0.0, q.initial() == 10.0) without changing call sites.
struct StateHandle {
    std::size_t index;
    constexpr operator std::size_t() const noexcept { return index; }

    // Declared here; defined inline below after BoundaryPoint is complete.
    // BoundaryPoint is a nested type so its definition can follow StateHandle
    // in this file without a separate forward-declaration header.
    struct BoundaryPoint;  // forward declaration
    BoundaryPoint initial() const;
    BoundaryPoint final()   const;
};

/// Opaque handle to a declared control. See StateHandle.
struct ControlHandle {
    std::size_t index;
    constexpr operator std::size_t() const noexcept { return index; }
};

/// A boundary attachment point — returned by StateHandle::initial() and
/// StateHandle::final(). Used with operator== to build a BoundaryConstraint
/// that lowers to set_initial_state / set_final_state.
/// Defined as a nested type of StateHandle so it can follow StateHandle's
/// definition in the same header (BoundaryPoint holds StateHandle by value,
/// so it must be defined AFTER StateHandle is complete).
struct StateHandle::BoundaryPoint {
    StateHandle state_handle;
    enum class Kind { Initial, Final };
    Kind kind;
};

/// Convenience alias so callers can refer to goss::model::BoundaryPoint
/// without qualifying with the enclosing StateHandle scope.
using BoundaryPoint = StateHandle::BoundaryPoint;

/// Inline out-of-line definitions of StateHandle::initial() and final().
/// These must come after BoundaryPoint is fully defined, because they
/// return BoundaryPoint by value. C++17 inline functions in headers are
/// guaranteed to have exactly one definition across all translation units.
inline BoundaryPoint StateHandle::initial() const {
    return BoundaryPoint{*this, BoundaryPoint::Kind::Initial};
}
inline BoundaryPoint StateHandle::final() const {
    return BoundaryPoint{*this, BoundaryPoint::Kind::Final};
}

}  // namespace goss::model
