#pragma once
#include <stdexcept>
#include <string>
namespace goss::nlp {
class NLPError : public std::runtime_error {
 public:
    explicit NLPError(const std::string& message) : std::runtime_error(message) {}
};
}  // namespace goss::nlp
