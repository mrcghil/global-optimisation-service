// include/goss/sim/sweep_worker.hpp
#pragma once
// Fork/pipe POSIX machinery is in sweep_worker.cpp — this header is
// intentionally free of <unistd.h> / <sys/wait.h> to keep the API clean.
#include <vector>
#include "goss/model/parameter.hpp"
#include "goss/nlp/nlp_problem.hpp"
#include "goss/sim/sweep_result.hpp"
#include "goss/solver/solver.hpp"

namespace goss::sim {

/// Serializes a SweepPoint to a fixed binary layout suitable for pipe transfer.
/// Layout: [int status][double objective]
///         [size_t nparams][double * nparams]
///         [size_t nx][double * nx]
///         [size_t msglen][char * msglen]
/// Same-host fork — no cross-arch endianness concern.
std::vector<char> serialize_sweep_point(const SweepPoint& point);

/// Deserializes a SweepPoint previously produced by serialize_sweep_point.
SweepPoint deserialize_sweep_point(const std::vector<char>& bytes);

/// Runs ONE parameter point in a freshly forked child process.
///
/// The child:
///   1. Binds the parameter values via apply_parameters (validate + inject).
///   2. Calls solver.solve(problem, initial_guess).
///   3. Serializes the resulting SweepPoint to a pipe.
///   4. Calls ::_exit(0) — skipping atexit handlers and global/static
///      destructors, thereby avoiding double-flushing of IPOPT/CppADCG
///      static state or the JIT temp-dir cleanup still owned by the parent.
///
/// The parent reads the SweepPoint from the pipe after draining it to EOF,
/// then reaps the child with waitpid.
///
/// If the child crashes (signal / segfault), the pipe buffer will be empty and
/// this function returns a SweepPoint with status=Failure whose message names
/// the killing signal or exit code — it NEVER throws for a child-side failure.
///
/// Throws SimError only for setup failures (pipe() or fork() itself failing).
SweepPoint solve_point_in_child(
    nlp::NLPProblem& problem,
    const model::ParameterValidator& validator,
    solver::Solver& solver,
    const std::vector<double>& parameters,
    const std::vector<double>& initial_guess);

}  // namespace goss::sim
