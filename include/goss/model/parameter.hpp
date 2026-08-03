// include/goss/model/parameter.hpp
#pragma once
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>
#include "goss/model/errors.hpp"

namespace goss::model {

struct ParameterHandle {
    std::size_t index;
    constexpr operator std::size_t() const noexcept { return index; }
};

struct ParameterSpec {
    std::string name;
    double default_value;
    double lower_bound;
    double upper_bound;
};

/// Build-time artifact bundled with a compiled problem. Validates a proposed
/// parameter set against declared size and per-parameter bounds, throwing
/// ModelError with a message that names the offending parameter and its bound.
class ParameterValidator {
 public:
    explicit ParameterValidator(std::vector<ParameterSpec> specs)
        : specs_(std::move(specs)) {
        defaults_.reserve(specs_.size());
        for (const ParameterSpec& spec : specs_) defaults_.push_back(spec.default_value);
    }

    std::size_t size() const { return specs_.size(); }
    const std::vector<double>& defaults() const { return defaults_; }

    void validate(const std::vector<double>& values) const {
        if (values.size() != specs_.size())
            throw ModelError("ParameterValidator: expected " +
                             std::to_string(specs_.size()) + " parameter(s), got " +
                             std::to_string(values.size()));
        for (std::size_t i = 0; i < specs_.size(); ++i) {
            const ParameterSpec& spec = specs_[i];
            if (std::isnan(values[i]))
                throw ModelError("ParameterValidator: parameter '" + spec.name +
                                 "' is NaN");
            if (values[i] < spec.lower_bound)
                throw ModelError("ParameterValidator: parameter '" + spec.name +
                                 "' value " + std::to_string(values[i]) +
                                 " is below its lower bound " +
                                 std::to_string(spec.lower_bound));
            if (values[i] > spec.upper_bound)
                throw ModelError("ParameterValidator: parameter '" + spec.name +
                                 "' value " + std::to_string(values[i]) +
                                 " exceeds its upper bound " +
                                 std::to_string(spec.upper_bound));
        }
    }

 private:
    std::vector<ParameterSpec> specs_;
    std::vector<double> defaults_;
};

}  // namespace goss::model
