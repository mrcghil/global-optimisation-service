// include/goss/transcription/transcription.hpp
#pragma once
#include <memory>
#include "goss/model/parameter.hpp"
#include "goss/nlp/nlp_problem.hpp"
#include "goss/transcription/variable_layout.hpp"

namespace goss::transcription {

/// Free-bound sentinel matching NLPProblem's convention.
constexpr double kInf = 2e19;

/// Bundle returned by a scheme's compile(): the transcribed NLP plus the
/// layout needed to unpack a solution vector back into states/controls.
///
/// NOTE ON DESIGN: OcpProblem is templated on its dynamics/cost functor types
/// (so the SAME functor records under CppAD AD types and evaluates under
/// double). A runtime-polymorphic Transcription base with a virtual compile()
/// cannot accept a templated OcpProblem, so v1 exposes each scheme's compile()
/// as a templated static function (Trapezoidal::compile, HermiteSimpson::compile)
/// rather than through virtual dispatch. Runtime scheme selection is not needed
/// yet (YAGNI); if required later, a type-erased OcpProblem wrapper can be added.
struct CompiledOcp {
    std::unique_ptr<nlp::NLPProblem> problem;
    VariableLayout layout;
    model::ParameterValidator validator{ std::vector<model::ParameterSpec>{} };   // empty (no params) by default
};

}  // namespace goss::transcription
