// include/goss/spec/errors.hpp
#pragma once
#include <stdexcept>
#include <string>
namespace goss::spec {
class SpecError : public std::runtime_error {
 public:
    explicit SpecError(const std::string& message) : std::runtime_error(message) {}
};
}  // namespace goss::spec
