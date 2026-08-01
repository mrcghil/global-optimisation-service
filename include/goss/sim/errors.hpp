// include/goss/sim/errors.hpp
#pragma once
#include <stdexcept>
#include <string>
namespace goss::sim {
class SimError : public std::runtime_error {
 public:
    explicit SimError(const std::string& message) : std::runtime_error(message) {}
};
}  // namespace goss::sim
