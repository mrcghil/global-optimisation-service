// include/goss/transcription/ocp_problem.hpp
#pragma once
#include <cstddef>
#include <vector>
#include "goss/transcription/errors.hpp"

namespace goss::transcription {

struct Mesh {
    double t_initial;
    double t_final;
    std::size_t num_intervals;
    std::size_t num_nodes() const { return num_intervals + 1; }
    double interval_width() const { return (t_final - t_initial) / static_cast<double>(num_intervals); }
    void validate() const {
        if (num_intervals == 0) throw TranscriptionError("Mesh: num_intervals must be >= 1");
        if (t_final <= t_initial) throw TranscriptionError("Mesh: t_final must be > t_initial");
    }
};

template <typename DynamicsFn, typename CostFn>
struct OcpProblem {
    std::size_t num_states;
    std::size_t num_controls;
    DynamicsFn dynamics;   // template<T> vector<T> (const vector<T>& x, const vector<T>& u, T t)
    CostFn cost;           // template<T> T (const vector<T>& x, const vector<T>& u, T t) — running cost
    Mesh mesh;
    std::vector<double> state_lower;
    std::vector<double> state_upper;
    std::vector<double> control_lower;
    std::vector<double> control_upper;
    std::vector<double> initial_state;
    std::vector<double> initial_state_fixed;   // nonzero => pin node 0 state i
    std::vector<double> final_state;
    std::vector<double> final_state_fixed;      // nonzero => pin last node state i
};

}  // namespace goss::transcription
