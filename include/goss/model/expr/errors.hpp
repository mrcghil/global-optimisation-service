// include/goss/model/expr/errors.hpp
#pragma once
#include <stdexcept>
#include <string>
namespace goss::model::expr {

/// Thrown when the expression DSL detects misuse: mismatched model handles,
/// duplicate dynamics registration, missing dynamics at build time, etc.
/// Distinct from ModelError so callers can distinguish expression-layer
/// problems from model-metadata problems.
class ExprError : public std::runtime_error {
 public:
    explicit ExprError(const std::string& message) : std::runtime_error(message) {}
};

}  // namespace goss::model::expr
