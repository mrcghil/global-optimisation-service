// src/sim/sweep_worker.cpp
// POSIX fork/pipe implementation — <unistd.h> / <sys/wait.h> are isolated here
// so the public header remains platform-agnostic.
#include "goss/sim/sweep_worker.hpp"

#include <cerrno>
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
// Fork-based single-point worker — internal DRY helpers shared by
// solve_point_in_child and the parallel process-pool scheduler.  These are
// TU-private (anonymous namespace): the public API is solve_point_in_child and
// run_sweep_parallel, which own the launch/collect lifecycle so callers cannot
// accidentally leak a child by pairing launch/collect incorrectly.
// ---------------------------------------------------------------------------

namespace {

/// Identifies a live child and the parent's read-end of its result pipe.
struct WorkerHandle {
    pid_t pid;
    int   read_fd;
};

/// Result of draining a worker pipe: the deserialized point plus whether the
/// child produced no output at all (crashed before writing).  The flag lets
/// classify_crash decide whether to overwrite the message with waitpid detail
/// without coupling to a sentinel message string.
struct DrainedPoint {
    SweepPoint point;
    bool       empty_buffer = false;
};

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
        // child does NOT recompile.  Validation and solve happen atomically in
        // solve_with_parameters; no compile() path is triggered here.
        ::close(pipe_fds[0]);  // close read end — child only writes

        SweepPoint point;
        point.parameters = parameters;
        try {
            // validate then solve in one call — any ModelError from validation
            // is caught by the std::exception handler below, same as a solve failure.
            const solver::SolverResult solve_result =
                solve_with_parameters(solver, problem, validator, initial_guess, parameters);
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

// ---------------------------------------------------------------------------
// drain_and_deserialize — shared by collect_worker and the pool's second pass.
// ---------------------------------------------------------------------------

/// Drains `read_fd` to EOF (closing it when done), then deserializes and
/// returns the SweepPoint together with an `empty_buffer` flag.
///
/// If the pipe yielded no data at all (child crashed before writing anything),
/// `empty_buffer` is true and the point is a baseline Failure carrying the
/// supplied `parameters`; the caller refines the message after inspecting the
/// waitpid status (signal vs exit code) via classify_crash.
///
/// Throws SimError only if the accumulated data is non-empty but malformed
/// (truncated payload / trailing bytes) — that is a real IO error, not a
/// child-side failure.
DrainedPoint drain_and_deserialize(int read_fd,
                                   const std::vector<double>& parameters) {
    std::vector<char> accumulated_buffer;
    char read_chunk[4096];
    ssize_t bytes_read = 0;
    while ((bytes_read = ::read(read_fd, read_chunk, sizeof(read_chunk))) > 0)
        accumulated_buffer.insert(accumulated_buffer.end(),
                                  read_chunk, read_chunk + bytes_read);
    ::close(read_fd);

    if (accumulated_buffer.empty()) {
        // Child exited/crashed without writing anything.
        // Populate a baseline Failure point; the caller fills in the exact
        // message after calling waitpid.
        SweepPoint failure_point;
        failure_point.parameters = parameters;
        failure_point.status     = solver::SolverStatus::Failure;
        failure_point.message    = "worker produced no output";
        return DrainedPoint{std::move(failure_point), /*empty_buffer=*/true};
    }

    return DrainedPoint{deserialize_sweep_point(accumulated_buffer),
                        /*empty_buffer=*/false};
}

/// Refines a drained point's message from the child's waitpid status when the
/// child produced no output (`empty_buffer`).  A non-empty buffer means the
/// child wrote a result — even on a non-zero exit the deserialized message is
/// more informative, so it is left unchanged.
///
/// This helper ensures both the blocking path (solve_point_in_child) and the
/// pool path share identical crash-classification semantics.
void classify_crash(SweepPoint& point, bool empty_buffer, int wait_status) {
    if (!empty_buffer) return;

    if (WIFSIGNALED(wait_status)) {
        point.message =
            "worker killed by signal " + std::to_string(WTERMSIG(wait_status));
    } else {
        point.message =
            "worker produced no output (exit " +
            std::to_string(WEXITSTATUS(wait_status)) + ")";
    }
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
    DrainedPoint drained = drain_and_deserialize(handle.read_fd, parameters);

    int wait_status = 0;
    ::waitpid(handle.pid, &wait_status, 0);

    classify_crash(drained.point, drained.empty_buffer, wait_status);
    return std::move(drained.point);
}

}  // namespace

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

namespace {

/// Resolves the effective worker cap from `config.max_parallel_workers`:
///   - 0  → std::thread::hardware_concurrency() (fallback 1 if that returns 0)
///   - >0 → use as-is
std::size_t resolve_worker_cap(const SweepConfig& config) {
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

}  // namespace

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
    //
    // CRITICAL 2 guard: if any exception escapes the loop body (e.g. SimError
    // from a fork() failure inside try_launch_next), we must reap all still-live
    // children before propagating the exception to avoid zombie processes and fd
    // leaks.  The try/catch below handles this: on any exception it drains all
    // remaining fds and calls waitpid for each live child (best-effort — errors
    // during cleanup are ignored), then rethrows the original exception.
    try {
        while (!live_workers.empty()) {
            // Build the fd_set for all live pipe read-ends.
            // select() requires the highest fd + 1.
            fd_set read_set;
            FD_ZERO(&read_set);
            int max_fd = -1;
            for (const auto& kv : live_workers) {
                const int fd = kv.second.handle.read_fd;
                // FD_SET behaviour is undefined for fd >= FD_SETSIZE.  This
                // cannot happen at any realistic concurrency level (FD_SETSIZE
                // is typically 1024 and each worker uses one pipe fd), but the
                // guard eliminates theoretical UB and gives an actionable error.
                if (fd >= FD_SETSIZE)
                    throw SimError(
                        "run_sweep_parallel: pipe read_fd " + std::to_string(fd) +
                        " >= FD_SETSIZE (" + std::to_string(FD_SETSIZE) + ")");
                FD_SET(fd, &read_set);
                if (fd > max_fd) max_fd = fd;
            }

            // Block until at least one pipe becomes readable (EOF or data available).
            // No timeout — we always have live children at this point.
            const int ready_count = ::select(max_fd + 1, &read_set, nullptr, nullptr, nullptr);
            if (ready_count < 0) {
                // IMPORTANT 3: distinguish EINTR (harmless signal interrupt, retry)
                // from real select() errors (EBADF, ENOMEM, etc.) which would cause
                // an infinite busy-spin if silently retried.
                if (errno == EINTR) continue;
                throw SimError(std::string("select() failed: ") + std::strerror(errno));
            }
            if (ready_count == 0) {
                // Timeout — only possible if a timeout was supplied (we pass nullptr
                // so this should never happen, but guard for safety).
                continue;
            }

            // CRITICAL 1 — two-pass to avoid iterator invalidation:
            //
            // First pass: scan live_workers and collect all ready entries into a
            // local vector WITHOUT mutating the map.  Inserting into an
            // unordered_map during iteration can trigger a rehash that invalidates
            // all iterators (UB).  try_launch_next() emplaces into live_workers,
            // so it must NOT be called while iterating the map.
            //
            // Second pass: for each collected entry, erase it from live_workers,
            // drain+reap+record the result, then call try_launch_next().  By the
            // time try_launch_next emplaces a new entry the first-pass scan is
            // already complete and no live iterator exists.
            struct ReadyEntry {
                pid_t       pid;
                int         read_fd;
                std::size_t grid_index;
            };
            std::vector<ReadyEntry> ready_entries;

            // First pass: collect, no mutation.
            for (const auto& kv : live_workers) {
                const int fd = kv.second.handle.read_fd;
                if (FD_ISSET(fd, &read_set)) {
                    ready_entries.push_back(
                        ReadyEntry{kv.first, fd, kv.second.grid_index});
                }
            }

            // Second pass: process each ready entry.
            for (const ReadyEntry& ready : ready_entries) {
                // Erase first: once erased, this code path is solely responsible
                // for reaping the pid — the outer catch(...) only reaps workers
                // still IN live_workers.
                live_workers.erase(ready.pid);

                // Drain-before-waitpid: drain_and_deserialize closes the fd.
                // If drain_and_deserialize throws SimError (corrupt/truncated
                // non-empty payload — e.g. child SIGKILL'd mid-write of a large
                // payload), we MUST still reap the pid to avoid a zombie.
                // We catch, waitpid the now-erased pid, then rethrow so the
                // SimError still propagates out of run_sweep_parallel as the
                // genuine IO error it is.  The normal (non-throwing) path also
                // calls waitpid exactly once — no double-reap on either path.
                int wait_status = 0;
                DrainedPoint drained;
                try {
                    drained = drain_and_deserialize(
                        ready.read_fd, parameter_grid[ready.grid_index]);
                    // Now that the pipe is drained the child is guaranteed to have
                    // exited (it closes the write end before/after _exit(0)).
                    ::waitpid(ready.pid, &wait_status, 0);
                } catch (...) {
                    // Reap the zombie that would otherwise be orphaned because
                    // the pid was already removed from live_workers above.
                    ::waitpid(ready.pid, nullptr, 0);
                    throw;  // propagate the SimError unchanged
                }

                classify_crash(drained.point, drained.empty_buffer, wait_status);

                // Write to the original grid index — preserves order-invariant.
                result.points[ready.grid_index] = std::move(drained.point);

                // Immediately launch the next pending grid point to keep the pool
                // full.  Safe to emplace here: the first-pass scan is finished.
                try_launch_next();
            }
        }
    } catch (...) {
        // CRITICAL 2 cleanup: reap all still-live children before rethrowing.
        // Best-effort: ignore individual errors so cleanup never itself throws.
        for (auto& kv : live_workers) {
            const int fd = kv.second.handle.read_fd;
            // Drain to unblock the child if it is blocked in write().
            char discard[4096];
            while (::read(fd, discard, sizeof(discard)) > 0) { /* drain */ }
            ::close(fd);
            ::waitpid(kv.first, nullptr, 0);
        }
        live_workers.clear();
        throw;  // rethrow the original exception
    }

    return result;
}

}  // namespace goss::sim
