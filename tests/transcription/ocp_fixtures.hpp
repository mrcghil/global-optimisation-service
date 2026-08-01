// tests/transcription/ocp_fixtures.hpp
#pragma once
#include <cmath>
#include <cstddef>
#include <vector>
#include "goss/transcription/ocp_problem.hpp"

namespace goss::transcription::test {

// dx/dt = -x ; analytic x(t) = x0 * exp(-t). 1 state, 0 controls, zero cost.
struct ExpDecayDynamics {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x, const std::vector<T>& /*u*/, T /*t*/) const {
        return { -x[0] };
    }
};
struct ZeroCost {
    template <typename T>
    T operator()(const std::vector<T>& /*x*/, const std::vector<T>& /*u*/, T /*t*/) const {
        return T(0);
    }
};

inline auto make_exponential_decay(double x0, double tf, std::size_t intervals) {
    OcpProblem<ExpDecayDynamics, ZeroCost> ocp;
    ocp.num_states = 1;
    ocp.num_controls = 0;
    ocp.dynamics = ExpDecayDynamics{};
    ocp.cost = ZeroCost{};
    ocp.mesh = Mesh{0.0, tf, intervals};
    ocp.state_lower = { -1e19 };
    ocp.state_upper = { 1e19 };
    ocp.control_lower = {};
    ocp.control_upper = {};
    ocp.initial_state = { x0 };
    ocp.initial_state_fixed = { 1.0 };   // pin x(0) = x0
    ocp.final_state = { 0.0 };
    ocp.final_state_fixed = { 0.0 };      // free
    return ocp;
}

inline double exp_decay_solution(double x0, double t) { return x0 * std::exp(-t); }

// dx0=x1, dx1=-x0 ; analytic x0(t)=a cos t + b sin t with a=x0(0), b=x1(0). 2 states.
struct HarmonicDynamics {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x, const std::vector<T>& /*u*/, T /*t*/) const {
        return { x[1], -x[0] };
    }
};

inline auto make_harmonic(double x0_0, double x1_0, double tf, std::size_t intervals) {
    OcpProblem<HarmonicDynamics, ZeroCost> ocp;
    ocp.num_states = 2;
    ocp.num_controls = 0;
    ocp.dynamics = HarmonicDynamics{};
    ocp.cost = ZeroCost{};
    ocp.mesh = Mesh{0.0, tf, intervals};
    ocp.state_lower = { -1e19, -1e19 };
    ocp.state_upper = { 1e19, 1e19 };
    ocp.control_lower = {};
    ocp.control_upper = {};
    ocp.initial_state = { x0_0, x1_0 };
    ocp.initial_state_fixed = { 1.0, 1.0 };
    ocp.final_state = { 0.0, 0.0 };
    ocp.final_state_fixed = { 0.0, 0.0 };
    return ocp;
}

inline double harmonic_x0_solution(double a, double b, double t) {
    return a * std::cos(t) + b * std::sin(t);
}

// dx/dt = -10*x ; analytic x(t) = x0 * exp(-10*t). Fast decay, 1 state, 0 controls, zero cost.
// On a coarse uniform mesh the early intervals carry large truncation error; used to exercise AMR.
struct FastDecayDynamics {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x, const std::vector<T>& /*u*/, T /*t*/) const {
        return { T(-10.0) * x[0] };
    }
};

inline auto make_fast_decay(double x0, double tf, std::size_t intervals) {
    OcpProblem<FastDecayDynamics, ZeroCost> ocp;
    ocp.num_states = 1;
    ocp.num_controls = 0;
    ocp.dynamics = FastDecayDynamics{};
    ocp.cost = ZeroCost{};
    ocp.mesh = Mesh{0.0, tf, intervals};
    ocp.state_lower = { -1e19 };
    ocp.state_upper = { 1e19 };
    ocp.control_lower = {};
    ocp.control_upper = {};
    ocp.initial_state = { x0 };
    ocp.initial_state_fixed = { 1.0 };
    ocp.final_state = { 0.0 };
    ocp.final_state_fixed = { 0.0 };
    return ocp;
}

}  // namespace goss::transcription::test
