// include/goss/bench/harness.hpp
#pragma once
#include <chrono>
#include <cstddef>
#include <string>
#include <vector>
#include "goss/bench/benchmark_result.hpp"
#include "goss/bench/errors.hpp"
#include "goss/model/model.hpp"
#include "goss/sim/initial_guess.hpp"
#include "goss/sim/validation.hpp"
#include "goss/solver/solver.hpp"
#include "goss/solver/solver_result.hpp"
#include "goss/transcription/ocp_problem.hpp"

namespace goss::bench {

/// Run a single transcription scheme across all provided solvers, collecting
/// timing and accuracy metrics for each (scheme, solver) pair.
///
/// SchemeTag must be a scheme struct with a static compile() method
/// (e.g. goss::transcription::Trapezoidal, goss::transcription::HermiteSimpson).
/// Because compile() is a templated static on a concrete struct — not a virtual
/// on a base class — the scheme axis must be unrolled at the call-site by the
/// caller, once per scheme type. Solvers are runtime-polymorphic (solver::Solver*),
/// so they can be looped over without compile-time unrolling.
///
/// Timing covers ONLY the solver.solve() call; compilation and initial_guess
/// are setup costs, not solver-performance metrics.
///
/// On a successful solve, validation_error is filled with the result of
/// sim::validate_by_integration(ocp, result, layout) — an independent RK4 check.
/// On a non-Success solve, validation_error is 0.0 (not meaningful).
template <typename SchemeTag, typename DynamicsFn, typename CostFn>
std::vector<BenchmarkResult> run_scheme(
        const goss::transcription::OcpProblem<DynamicsFn, CostFn>& ocp,
        const goss::model::Model& model,
        const std::string& scheme_name,
        const std::string& model_name,
        const std::vector<goss::solver::Solver*>& solvers,
        const std::vector<std::string>& solver_names) {

    // Validate caller contract before touching any solver or compilation.
    if (solvers.empty())
        throw BenchError("run_scheme: solvers vector must not be empty");
    if (solvers.size() != solver_names.size())
        throw BenchError("run_scheme: solvers.size() != solver_names.size()");

    // Compile the OCP with the given scheme once; all solvers share the same NLP.
    // Each solver gets a freshly built linear_guess (cheap, stateless, correct).
    auto compiled = SchemeTag::compile(ocp, model_name);
    const goss::transcription::VariableLayout& layout = compiled.layout;
    const std::size_t total_variables = layout.total_variables();

    std::vector<BenchmarkResult> results;
    results.reserve(solvers.size());

    for (std::size_t solver_index = 0; solver_index < solvers.size(); ++solver_index) {
        BenchmarkResult bench_result;
        bench_result.scheme_name   = scheme_name;
        bench_result.solver_name   = solver_names[solver_index];
        bench_result.num_variables = total_variables;

        // Build a fresh initial guess for each solver; solver order must not matter.
        const std::vector<double> initial_guess = goss::sim::linear_guess(model, layout);

        // Time ONLY the solve call — not compilation, not initial guess generation.
        const auto time_before = std::chrono::steady_clock::now();
        goss::solver::SolverResult solve_result =
            solvers[solver_index]->solve(*compiled.problem, initial_guess);
        const auto time_after = std::chrono::steady_clock::now();

        bench_result.solve_status    = solve_result.status;
        bench_result.objective_value = solve_result.objective_value;
        bench_result.elapsed_seconds = std::chrono::duration<double>(
            time_after - time_before).count();

        // Validate only when the solver converged; a failed solution vector
        // would produce a meaningless (possibly NaN) validation error.
        if (solve_result.status == goss::solver::SolverStatus::Success) {
            bench_result.validation_error =
                goss::sim::validate_by_integration(ocp, solve_result, layout);
        } else {
            bench_result.validation_error = 0.0;
        }

        results.push_back(bench_result);
    }

    return results;
}

}  // namespace goss::bench
