// include/goss/solver/errors.hpp
#pragma once
#include <stdexcept>
#include <string>
namespace goss::solver {
class SolverError : public std::runtime_error {
 public:
    explicit SolverError(const std::string& message) : std::runtime_error(message) {}
};
}  // namespace goss::solver
