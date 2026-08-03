// src/sim/sweep_worker.cpp
// POSIX fork/pipe implementation — <unistd.h> / <sys/wait.h> are isolated here
// so the public header remains platform-agnostic.
#include "goss/sim/sweep_worker.hpp"

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "goss/sim/errors.hpp"
#include "goss/sim/parameters.hpp"
#include "goss/solver/solver_result.hpp"

namespace goss::sim {

// ---------------------------------------------------------------------------
// Binary serialization helpers
// ---------------------------------------------------------------------------

namespace {

/// Appends sizeof(T) bytes of `value` to `buffer` using memcpy.
template <typename T>
void append_scalar(std::vector<char>& buffer, const T value) {
    const std::size_t offset = buffer.size();
    buffer.resize(offset + sizeof(T));
    std::memcpy(buffer.data() + offset, &value, sizeof(T));
}

/// Reads sizeof(T) bytes from `data` at byte offset `offset` into `out`,
/// then advances `offset` by sizeof(T).
template <typename T>
void read_scalar(const char* data, std::size_t& offset, T& out) {
    std::memcpy(&out, data + offset, sizeof(T));
    offset += sizeof(T);
}

/// Appends a length-prefixed vector of doubles to `buffer`.
void append_double_vector(std::vector<char>& buffer,
                          const std::vector<double>& values) {
    const std::size_t count = values.size();
    append_scalar(buffer, count);
    const std::size_t offset = buffer.size();
    buffer.resize(offset + count * sizeof(double));
    if (count > 0)
        std::memcpy(buffer.data() + offset, values.data(), count * sizeof(double));
}

/// Reads a length-prefixed vector of doubles from `data` at `offset`.
void read_double_vector(const char* data, std::size_t& offset,
                        std::vector<double>& out) {
    std::size_t count = 0;
    read_scalar(data, offset, count);
    out.resize(count);
    if (count > 0) {
        std::memcpy(out.data(), data + offset, count * sizeof(double));
        offset += count * sizeof(double);
    }
}

/// Appends a length-prefixed string to `buffer`.
void append_string(std::vector<char>& buffer, const std::string& text) {
    const std::size_t length = text.size();
    append_scalar(buffer, length);
    buffer.insert(buffer.end(), text.begin(), text.end());
}

/// Reads a length-prefixed string from `data` at `offset`.
void read_string(const char* data, std::size_t& offset, std::string& out) {
    std::size_t length = 0;
    read_scalar(data, offset, length);
    out.assign(data + offset, length);
    offset += length;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public serialization API
// ---------------------------------------------------------------------------

/// Fixed binary layout (same-host fork — no cross-arch endianness concern):
///   [int status][double objective]
///   [size_t nparams][double * nparams]
///   [size_t nx][double * nx]
///   [size_t msglen][char * msglen]
std::vector<char> serialize_sweep_point(const SweepPoint& point) {
    std::vector<char> buffer;
    // Reserve a reasonable initial capacity to reduce reallocations.
    buffer.reserve(64 + point.parameters.size() * sizeof(double)
                      + point.x.size() * sizeof(double)
                      + point.message.size());

    const int status_as_int = static_cast<int>(point.status);
    append_scalar(buffer, status_as_int);
    append_scalar(buffer, point.objective_value);
    append_double_vector(buffer, point.parameters);
    append_double_vector(buffer, point.x);
    append_string(buffer, point.message);
    return buffer;
}

SweepPoint deserialize_sweep_point(const std::vector<char>& bytes) {
    SweepPoint point;
    const char* data = bytes.data();
    std::size_t offset = 0;

    int status_as_int = 0;
    read_scalar(data, offset, status_as_int);
    point.status = static_cast<solver::SolverStatus>(status_as_int);

    read_scalar(data, offset, point.objective_value);
    read_double_vector(data, offset, point.parameters);
    read_double_vector(data, offset, point.x);
    read_string(data, offset, point.message);
    return point;
}

// ---------------------------------------------------------------------------
// Fork-based single-point worker
// ---------------------------------------------------------------------------

SweepPoint solve_point_in_child(nlp::NLPProblem& problem,
                                const model::ParameterValidator& validator,
                                solver::Solver& solver,
                                const std::vector<double>& parameters,
                                const std::vector<double>& initial_guess) {
    int pipe_fds[2];
    if (::pipe(pipe_fds) != 0)
        throw SimError("solve_point_in_child: pipe() failed");

    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(pipe_fds[0]);
        ::close(pipe_fds[1]);
        throw SimError("solve_point_in_child: fork() failed");
    }

    if (pid == 0) {
        // ---- CHILD ----
        // The compiled .so / GenericModel is inherited copy-on-write — the
        // child does NOT recompile.  apply_parameters just stores the values;
        // solver.solve() reads them.  No compile() path is triggered here.
        ::close(pipe_fds[0]);  // close read end — child only writes

        SweepPoint point;
        point.parameters = parameters;
        try {
            apply_parameters(problem, validator, parameters);
            const solver::SolverResult solve_result =
                solver.solve(problem, initial_guess);
            point.status          = solve_result.status;
            point.objective_value = solve_result.objective_value;
            point.x               = solve_result.x;
            point.message         = solve_result.message;
        } catch (const std::exception& error) {
            point.status  = solver::SolverStatus::Failure;
            point.message = std::string("child exception: ") + error.what();
        }

        const std::vector<char> serialized_bytes = serialize_sweep_point(point);

        // Write all bytes — loop to handle partial writes (POSIX guarantees
        // writes up to PIPE_BUF are atomic, but larger payloads may be split).
        std::size_t total_written = 0;
        while (total_written < serialized_bytes.size()) {
            const ssize_t bytes_written = ::write(
                pipe_fds[1],
                serialized_bytes.data() + total_written,
                serialized_bytes.size() - total_written);
            if (bytes_written <= 0) break;  // write error — parent will detect empty buffer
            total_written += static_cast<std::size_t>(bytes_written);
        }
        ::close(pipe_fds[1]);

        // _exit(0): skip atexit handlers + global/static destructors, avoiding
        // double-flush of IPOPT/CppADCG static state or the JIT temp-dir
        // cleanup that the parent process still owns.
        ::_exit(0);
    }

    // ---- PARENT ----
    // Close write end first so the child's close of its write end triggers EOF
    // on the parent's read loop (otherwise the parent would block forever).
    ::close(pipe_fds[1]);

    // Drain the pipe to EOF before waitpid to avoid deadlock: if the child's
    // output exceeds the OS pipe buffer, the child blocks in write() and the
    // parent blocks in waitpid() — mutual deadlock.  Drain first, reap after.
    std::vector<char> accumulated_buffer;
    char read_chunk[4096];
    ssize_t bytes_read;
    while ((bytes_read = ::read(pipe_fds[0], read_chunk, sizeof(read_chunk))) > 0)
        accumulated_buffer.insert(accumulated_buffer.end(),
                                  read_chunk, read_chunk + bytes_read);
    ::close(pipe_fds[0]);

    int wait_status = 0;
    ::waitpid(pid, &wait_status, 0);

    // An empty buffer means the child crashed before writing anything.
    // Return a classified Failure — never throw (child-side failures are results,
    // not setup errors).
    if (accumulated_buffer.empty()) {
        SweepPoint failure_point;
        failure_point.parameters = parameters;
        failure_point.status     = solver::SolverStatus::Failure;
        if (WIFSIGNALED(wait_status)) {
            failure_point.message =
                "worker killed by signal " + std::to_string(WTERMSIG(wait_status));
        } else {
            failure_point.message =
                "worker produced no output (exit " +
                std::to_string(WEXITSTATUS(wait_status)) + ")";
        }
        return failure_point;
    }

    return deserialize_sweep_point(accumulated_buffer);
}

}  // namespace goss::sim
