#pragma once
#include <stdexcept>
#include <string>
namespace goss::model {
class ModelError : public std::runtime_error {
 public:
    explicit ModelError(const std::string& message) : std::runtime_error(message) {}
};
class ComponentError : public std::runtime_error {
 public:
    explicit ComponentError(const std::string& message)
        : std::runtime_error(message) {}
};
}  // namespace goss::model
