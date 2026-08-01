// include/goss/bench/report.hpp
#pragma once
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include "goss/bench/benchmark_result.hpp"
#include "goss/bench/errors.hpp"

namespace goss::bench {

/// Returns a human-readable fixed-width comparison table.
/// Each row shows: Scheme | Solver | Status | Objective | Time(s) | ValidationErr | NVars
/// The table is PURE: given the same results vector, the output is deterministic
/// (except that elapsed_seconds values are whatever was stored in the results —
/// callers must not assert on their exact magnitude in tests).
inline std::string to_table(const std::vector<BenchmarkResult>& results) {
    // Fixed column widths for readability.
    constexpr int kColScheme    = 18;
    constexpr int kColSolver    = 16;
    constexpr int kColStatus    = 20;
    constexpr int kColObjective = 14;
    constexpr int kColTime      = 12;
    constexpr int kColValErr    = 16;
    constexpr int kColNVars     = 8;

    std::ostringstream out;
    auto row_separator = [&]() {
        // Width is exactly the sum of all column widths: no inter-column separators,
        // so the separator is flush with the data rows.
        out << std::string(kColScheme + kColSolver + kColStatus + kColObjective
                           + kColTime + kColValErr + kColNVars, '-') << "\n";
    };

    // Header row.
    row_separator();
    out << std::left
        << std::setw(kColScheme)    << "Scheme"
        << std::setw(kColSolver)    << "Solver"
        << std::setw(kColStatus)    << "Status"
        << std::setw(kColObjective) << "Objective"
        << std::setw(kColTime)      << "Time(s)"
        << std::setw(kColValErr)    << "ValidationErr"
        << std::setw(kColNVars)     << "NVars"
        << "\n";
    row_separator();

    for (const auto& r : results) {
        out << std::left
            << std::setw(kColScheme)    << r.scheme_name
            << std::setw(kColSolver)    << r.solver_name
            << std::setw(kColStatus)    << solver_status_name(r.solve_status)
            << std::setw(kColObjective) << std::setprecision(6) << r.objective_value
            << std::setw(kColTime)      << std::setprecision(4) << r.elapsed_seconds
            << std::setw(kColValErr)    << std::scientific << std::setprecision(3) << r.validation_error
            << std::setw(kColNVars)     << r.num_variables
            << "\n";
        out << std::defaultfloat;
    }
    row_separator();
    return out.str();
}

/// Returns all results as CSV text, mirroring sim::to_csv's pattern.
/// Header: scheme,solver,status,objective,elapsed_s,validation_error,num_variables
/// One data row per result. Full double precision (setprecision(17)).
inline std::string to_csv(const std::vector<BenchmarkResult>& results) {
    std::ostringstream out;
    out << std::setprecision(17);
    out << "scheme,solver,status,objective,elapsed_s,validation_error,num_variables\n";
    for (const auto& r : results) {
        out << r.scheme_name << ","
            << r.solver_name << ","
            << solver_status_name(r.solve_status) << ","
            << r.objective_value << ","
            << r.elapsed_seconds << ","
            << r.validation_error << ","
            << r.num_variables << "\n";
    }
    return out.str();
}

/// Writes to_csv output to the file at path.
/// Throws BenchError if the file cannot be opened.
inline void write_csv(const std::vector<BenchmarkResult>& results,
                      const std::string& path) {
    std::ofstream file(path);
    if (!file) throw BenchError("write_csv: cannot open '" + path + "' for writing");
    file << to_csv(results);
    file.flush();
    if (!file) throw BenchError("write_csv: write failed for '" + path + "'");
}

}  // namespace goss::bench
