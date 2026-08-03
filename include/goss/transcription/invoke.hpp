// include/goss/transcription/invoke.hpp
#pragma once
#include <type_traits>
#include <utility>
#include <vector>

namespace goss::transcription::detail {

// Calls dynamics(x,u,p,t) when that overload exists, otherwise dynamics(x,u,t).
template <typename Fn, typename T>
auto call_dynamics(const Fn& fn, const std::vector<T>& x, const std::vector<T>& u,
                   const std::vector<T>& p, T t)
    -> std::vector<T> {
    if constexpr (std::is_invocable_v<const Fn&, const std::vector<T>&,
                                      const std::vector<T>&, const std::vector<T>&, T>) {
        return fn(x, u, p, t);
    } else {
        (void)p;
        return fn(x, u, t);
    }
}

template <typename Fn, typename T>
auto call_cost(const Fn& fn, const std::vector<T>& x, const std::vector<T>& u,
               const std::vector<T>& p, T t) -> T {
    if constexpr (std::is_invocable_v<const Fn&, const std::vector<T>&,
                                      const std::vector<T>&, const std::vector<T>&, T>) {
        return fn(x, u, p, t);
    } else {
        (void)p;
        return fn(x, u, t);
    }
}

}  // namespace goss::transcription::detail
