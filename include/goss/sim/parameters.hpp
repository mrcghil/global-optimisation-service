// include/goss/sim/parameters.hpp
#pragma once
#include <vector>
#include "goss/model/parameter.hpp"
#include "goss/nlp/nlp_problem.hpp"

namespace goss::sim {

/// Validates a proposed parameter set against the compiled problem's validator
/// (throwing ModelError with an explicit, parameter-naming message on failure),
/// then injects it into the NLP (may throw ADError on a backend size mismatch).
/// This is the single entry point a sweep runner uses per parameter point.
inline void apply_parameters(nlp::NLPProblem& problem,
                             const model::ParameterValidator& validator,
                             const std::vector<double>& values) {
    validator.validate(values);       // explicit errors, before any solver work
    problem.set_parameters(values);   // compile-once injection
}

}  // namespace goss::sim
