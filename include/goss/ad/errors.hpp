// include/goss/ad/errors.hpp
#pragma once
#include <stdexcept>
#include <string>
namespace goss::ad {
class ADError : public std::runtime_error {
 public:
    explicit ADError(const std::string& message) : std::runtime_error(message) {}
};
}  // namespace goss::ad
