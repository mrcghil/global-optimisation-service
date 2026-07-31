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
