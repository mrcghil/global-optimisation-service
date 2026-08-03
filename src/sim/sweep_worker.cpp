// src/sim/sweep_worker.cpp
// POSIX fork/pipe implementation — <unistd.h> / <sys/wait.h> are isolated here
// so the public header remains platform-agnostic.
#include "goss/sim/sweep_worker.hpp"

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <sys/select.h>
#include <sys/wait.h>
#include <unistd.h>

#include "goss/sim/errors.hpp"
#include "goss/sim/parameters.hpp"
#include "goss/sim/sweep.hpp"
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
/// then advances `offset` by sizeof(T).  `buf_size` is the total buffer length;
/// throws SimError if the requested range exceeds it (truncated payload).
template <typename T>
void read_scalar(const char* data, std::size_t& offset, T& out,
                 std::size_t buf_size) {
    if (offset + sizeof(T) > buf_size)
        throw SimError("truncated sweep-point payload");
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
/// `buf_size` is the total buffer length; throws SimError on truncation or
/// on a corrupt length that would drive an oversized memcpy.
void read_double_vector(const char* data, std::size_t& offset,
                        std::vector<double>& out, std::size_t buf_size) {
    std::size_t count = 0;
    read_scalar(data, offset, count, buf_size);
    // Validate that enough bytes remain before reading.  Division form avoids
    // the integer overflow that a multiply check would have when count is near
    // SIZE_MAX / sizeof(double): a corrupt/malicious count would wrap to a small
    // value, pass the multiply check, and drive an oversized memcpy.
    // read_scalar guarantees offset <= buf_size, so buf_size - offset is safe.
    if (count > (buf_size - offset) / sizeof(double))
        throw SimError("truncated sweep-point payload");
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
/// `buf_size` is the total buffer length; throws SimError on truncation or
/// on a corrupt length that would drive an oversized copy.
void read_string(const char* data, std::size_t& offset, std::string& out,
                 std::size_t buf_size) {
    std::size_t length = 0;
    read_scalar(data, offset, length, buf_size);
    // Validate that `length` chars actually remain before reading.
    if (length > buf_size - offset)
        throw SimError("truncated sweep-point payload");
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
    const char* data       = bytes.data();
    const std::size_t size = bytes.size();
    std::size_t offset     = 0;

    int status_as_int = 0;
    read_scalar(data, offset, status_as_int, size);
    point.status = static_cast<solver::SolverStatus>(status_as_int);

    read_scalar(data, offset, point.objective_value, size);
    read_double_vector(data, offset, point.parameters, size);
    read_double_vector(data, offset, point.x, size);
    read_string(data, offset, point.message, size);

    // Defense in depth: a well-formed payload must be consumed exactly.
    if (offset != size)
        throw SimError("sweep-point payload has trailing bytes");

    return point;
}

// ---------------------------------------------------------------------------
// Fork-based single-point worker — DRY helpers shared by solve_point_in_child
// and the parallel process-pool scheduler.
// ---------------------------------------------------------------------------

/// Forks a child that applies `parameters`, solves, writes the serialized
/// SweepPoint to a pipe, and calls ::_exit(0).
///
/// Returns a WorkerHandle{pid, read_fd} so the parent can later collect the
/// result via collect_worker.  The parent's NLPProblem is NOT mutated — all
/// parameter application happens inside the child's copy-on-write address space.
///
/// Throws SimError if pipe() or fork() fails.
WorkerHandle launch_worker(nlp::NLPProblem& problem,
                           const model::ParameterValidator& validator,
                           solver::Solver& solver,
                           const std::vector<double>& parameters,
                           const std::vector<double>& initial_guess) {
    int pipe_fds[2];
    if (::pipe(pipe_fds) != 0)
        throw SimError("launch_worker: pipe() failed");

    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(pipe_fds[0]);
        ::close(pipe_fds[1]);
        throw SimError("launch_worker: fork() failed");
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
        // On error (bytes_written < 0, e.g. EPIPE), close the write fd and
        // _exit(1) immediately so the parent sees an empty buffer and returns a
        // clean Failure — never leave a partial payload for the parent to parse.
        std::size_t total_written = 0;
        while (total_written < serialized_bytes.size()) {
            const ssize_t bytes_written = ::write(
                pipe_fds[1],
                serialized_bytes.data() + total_written,
                serialized_bytes.size() - total_written);
            if (bytes_written < 0) {
                ::close(pipe_fds[1]);
                ::_exit(1);
            }
            // bytes_written == 0 should not occur on a pipe, but treat it as a
            // stall and retry rather than silently breaking.
            if (bytes_written > 0)
                total_written += static_cast<std::size_t>(bytes_written);
        }
        ::close(pipe_fds[1]);

        // _exit(0): skip atexit handlers + global/static destructors, avoiding
        // double-flush of IPOPT/CppADCG static state or the JIT temp-dir
        // cleanup that the parent process still owns.
        ::_exit(0);
    }

    // ---- PARENT ----
    // Close write end: when the child later closes its own write end, the
    // parent's read() will return 0 (EOF).  Without this close the parent's
    // read loop would block forever even after the child exits.
    ::close(pipe_fds[1]);

    return WorkerHandle{pid, pipe_fds[0]};
}

/// Drains `handle.read_fd` to EOF, reaps the child with waitpid, and
/// deserializes the SweepPoint.
///
/// Drain-before-waitpid ordering avoids the deadlock that would occur if the
/// child's serialized output exceeds the OS pipe buffer capacity: the child
/// would block in write() while the parent blocks in waitpid().  Draining
/// first lets the child complete its write and exit.
///
/// If the accumulated buffer is empty (child crashed before writing anything),
/// returns a Failure SweepPoint identifying the killing signal or exit code.
/// Never throws for child-side failures; throws SimError only if deserialization
/// itself finds a corrupt payload.
SweepPoint collect_worker(const WorkerHandle& handle,
                          const std::vector<double>& parameters) {
    std::vector<char> accumulated_buffer;
    char read_chunk[4096];
    ssize_t bytes_read = 0;
    while ((bytes_read = ::read(handle.read_fd, read_chunk, sizeof(read_chunk))) > 0)
        accumulated_buffer.insert(accumulated_buffer.end(),
                                  read_chunk, read_chunk + bytes_read);
    ::close(handle.read_fd);

    int wait_status = 0;
    ::waitpid(handle.pid, &wait_status, 0);

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

/// Convenience wrapper: launch_worker followed by an immediate collect_worker.
/// Blocks until the child exits and its result is fully deserialized.
SweepPoint solve_point_in_child(nlp::NLPProblem& problem,
                                const model::ParameterValidator& validator,
                                solver::Solver& solver,
                                const std::vector<double>& parameters,
                                const std::vector<double>& initial_guess) {
    const WorkerHandle handle =
        launch_worker(problem, validator, solver, parameters, initial_guess);
    return collect_worker(handle, parameters);
}

// ---------------------------------------------------------------------------
// Bounded process-pool parallel sweep
// ---------------------------------------------------------------------------

/// Resolves the effective worker cap from `config.max_parallel_workers`:
///   - 0  → std::thread::hardware_concurrency() (fallback 1 if that returns 0)
///   - >0 → use as-is
static std::size_t resolve_worker_cap(const SweepConfig& config) {
    if (config.max_parallel_workers > 0)
        return config.max_parallel_workers;
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    return (hardware_threads > 0) ? static_cast<std::size_t>(hardware_threads) : 1u;
}

/// Per-live-child bookkeeping: associates a WorkerHandle with the grid index
/// the child was launched for, so completed results can be written to the
/// correct position regardless of completion order.
struct LiveWorkerEntry {
    WorkerHandle handle;
    std::size_t  grid_index;
};

SweepResult run_sweep_parallel(
        nlp::NLPProblem& problem,
        const model::ParameterValidator& validator,
        solver::Solver& solver,
        const std::vector<std::vector<double>>& parameter_grid,
        const std::vector<double>& initial_guess,
        const SweepConfig& config) {
    const std::size_t total_points    = parameter_grid.size();
    const std::size_t effective_cap   = resolve_worker_cap(config);

    // Pre-size result so that each completed child can write its SweepPoint
    // directly at its original grid index — order is preserved regardless of
    // which child finishes first.
    SweepResult result;
    result.points.resize(total_points);

    // Map from pid → LiveWorkerEntry for all currently running children.
    std::unordered_map<pid_t, LiveWorkerEntry> live_workers;
    live_workers.reserve(effective_cap);

    // Index of the next grid point yet to be launched.
    std::size_t next_launch_index = 0;

    // --- Helper: launch one more pending grid point if any remain ---
    auto try_launch_next = [&]() {
        if (next_launch_index >= total_points) return;
        const std::size_t grid_index = next_launch_index++;
        // launch_worker forks and returns immediately; parameters are applied
        // INSIDE the child — the parent's `problem` is never mutated here.
        WorkerHandle handle = launch_worker(
            problem, validator, solver,
            parameter_grid[grid_index], initial_guess);
        live_workers.emplace(handle.pid, LiveWorkerEntry{handle, grid_index});
    };

    // Seed the pool: launch up to `effective_cap` children at once.
    while (live_workers.size() < effective_cap && next_launch_index < total_points)
        try_launch_next();

    // Sliding-window loop: each time a child's pipe reaches EOF (child exited
    // and closed its write end), drain it to avoid the deadlock where the child
    // fills the OS pipe buffer and blocks in write() while the parent blocks in
    // waitpid().  We use select() to find a ready fd without reaping first.
    while (!live_workers.empty()) {
        // Build the fd_set for all live pipe read-ends.
        // select() requires the highest fd + 1.
        fd_set read_set;
        FD_ZERO(&read_set);
        int max_fd = -1;
        for (const auto& kv : live_workers) {
            const int fd = kv.second.handle.read_fd;
            FD_SET(fd, &read_set);
            if (fd > max_fd) max_fd = fd;
        }

        // Block until at least one pipe becomes readable (EOF or data available).
        // No timeout — we always have live children at this point.
        const int ready_count = ::select(max_fd + 1, &read_set, nullptr, nullptr, nullptr);
        if (ready_count <= 0) {
            // select() was interrupted (EINTR) or hit an unexpected error.
            // Retry to avoid a stale loop.
            continue;
        }

        // Collect ALL ready fds in this round (avoids repeated iterations when
        // multiple children finish nearly simultaneously).
        for (auto it = live_workers.begin(); it != live_workers.end(); ) {
            const int fd = it->second.handle.read_fd;
            if (!FD_ISSET(fd, &read_set)) {
                ++it;
                continue;
            }

            const LiveWorkerEntry entry = it->second;
            it = live_workers.erase(it);  // remove before launching next

            // Drain-before-waitpid: the pipe may still have buffered data even
            // though select() said it's readable; drain to EOF fully.
            std::vector<char> accumulated_buffer;
            {
                char    read_chunk[4096];
                ssize_t bytes_read = 0;
                while ((bytes_read = ::read(fd, read_chunk, sizeof(read_chunk))) > 0)
                    accumulated_buffer.insert(accumulated_buffer.end(),
                                              read_chunk, read_chunk + bytes_read);
                ::close(fd);
            }

            // Now that the pipe is drained the child is guaranteed to have
            // exited (it closes the write end before/after _exit(0)).
            int   wait_status = 0;
            ::waitpid(entry.handle.pid, &wait_status, 0);

            SweepPoint point;
            if (accumulated_buffer.empty()) {
                // Child crashed before writing any output.
                point.parameters = parameter_grid[entry.grid_index];
                point.status     = solver::SolverStatus::Failure;
                if (WIFSIGNALED(wait_status)) {
                    point.message =
                        "worker killed by signal " +
                        std::to_string(WTERMSIG(wait_status));
                } else {
                    point.message =
                        "worker produced no output (exit " +
                        std::to_string(WEXITSTATUS(wait_status)) + ")";
                }
            } else {
                point = deserialize_sweep_point(accumulated_buffer);
            }

            // Write to the original grid index — preserves order-invariant.
            result.points[entry.grid_index] = std::move(point);

            // Immediately launch the next pending grid point to keep the pool full.
            try_launch_next();
        }
    }

    return result;
}

}  // namespace goss::sim
