// include/goss/bench/errors.hpp
#pragma once
#include <stdexcept>
#include <string>

namespace goss::bench {

/// Thrown for setup/usage errors in the benchmark harness (e.g. empty solver
/// list, invalid configuration). Individual solve failures are captured in
/// BenchmarkResult.solve_status and do NOT throw.
class BenchError : public std::runtime_error {
 public:
    explicit BenchError(const std::string& message) : std::runtime_error(message) {}
};

}  // namespace goss::bench
