// tools/bench_export/main.cpp
//
// Standalone executable: run the flagship (scheme x solver) benchmark matrix
// on the exp-decay problem and write results to a CSV file.
//
// Usage: goss_bench_export <output.csv>
//
// Writes 4 rows (Trapezoidal x {IpoptSolver, NloptSolver},
//                HermiteSimpson x {IpoptSolver, NloptSolver}).
// Prints "wrote N rows" to stdout on success; exits non-zero on error.
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "goss/bench/harness.hpp"
#include "goss/bench/report.hpp"
#include "goss/model/model.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "goss/solver/nlopt_solver.hpp"
#include "goss/solver/solver.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/transcription/trapezoidal.hpp"

namespace {

// dx/dt = -x, x(0) = 1, zero running cost.  Analytic: x(t) = exp(-t).
struct ExpDecayDyn {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x_vec,
                              const std::vector<T>&, T) const {
        return {-x_vec[0]};
    }
};

struct ZeroCostFn {
    template <typename T>
    T operator()(const std::vector<T>&, const std::vector<T>&, T) const {
        return T(0);
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: goss_bench_export <output.csv>\n";
        return EXIT_FAILURE;
    }
    const std::string output_path = argv[1];

    // Build the exp-decay OCP: 1 state, 20 intervals, horizon [0, 1].
    goss::model::Model model;
    auto x_state = model.add_state("x");
    model.set_initial_state(x_state, 1.0);
    model.set_mesh(0.0, 1.0, 20);
    auto ocp = model.build(ExpDecayDyn{}, ZeroCostFn{});

    // Solver instances — kept alive across all run_scheme calls.
    goss::solver::IpoptSolver ipopt_solver;
    goss::solver::NloptSolver nlopt_solver;
    std::vector<goss::solver::Solver*> solvers     = {&ipopt_solver, &nlopt_solver};
    std::vector<std::string>           solver_names = {"IpoptSolver", "NloptSolver"};

    // Run both schemes; scheme axis unrolled at compile time (see harness.hpp).
    auto trap_results = goss::bench::run_scheme<goss::transcription::Trapezoidal>(
        ocp, model, "Trapezoidal", "bench_export_trap", solvers, solver_names);
    auto hs_results = goss::bench::run_scheme<goss::transcription::HermiteSimpson>(
        ocp, model, "HermiteSimpson", "bench_export_hs", solvers, solver_names);

    // Merge into a single results vector (Trapezoidal rows first).
    std::vector<goss::bench::BenchmarkResult> all_results = trap_results;
    all_results.insert(all_results.end(), hs_results.begin(), hs_results.end());

    // Write CSV; throws goss::bench::BenchError on I/O failure.
    goss::bench::write_csv(all_results, output_path);

    std::cout << "wrote " << all_results.size() << " rows\n";
    return EXIT_SUCCESS;
}
