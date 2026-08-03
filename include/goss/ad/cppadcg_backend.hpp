// include/goss/ad/cppadcg_backend.hpp
#pragma once
// Use the umbrella header — NOT the sub-header <cppad/cg/cg.hpp>.
// Task 6 confirmed <cppad/cg.hpp> is the correct include path.
#include <cppad/cg.hpp>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "goss/ad/ad_backend.hpp"
#include "goss/ad/errors.hpp"

namespace goss::ad {

namespace detail {

using CGScalar = CppAD::cg::CG<double>;
using ADCG = CppAD::AD<CGScalar>;

/// Ownership bundle for the JIT-compiled model.
/// Both members must stay alive together: the library owns the symbols that
/// the model's function pointers point into.
///
/// jac_rows / jac_cols hold the sparse Jacobian sparsity pattern captured
/// once by compile_and_load via model->JacobianSparsity().  The k-th entry
/// represents the non-zero at (jac_rows[k], jac_cols[k]).
///
/// hess_rows / hess_cols hold the full symmetric Hessian sparsity pattern
/// captured via model->HessianSparsity().  CppADCodeGen returns the full
/// symmetric pattern here; lower-triangle filtering is applied in the
/// CppADCGBackend constructor when building hessian_sparsity_.
struct CompiledModel {
    std::unique_ptr<CppAD::cg::DynamicLib<double>> library;
    std::unique_ptr<CppAD::cg::GenericModel<double>> model;
    std::vector<std::size_t> jac_rows;
    std::vector<std::size_t> jac_cols;
    std::vector<std::size_t> hess_rows;
    std::vector<std::size_t> hess_cols;
};

/// Non-template helper declared here, defined in src/ad/cppadcg_backend.cpp.
/// Generates C source from `fun`, JIT-compiles via GccCompiler, loads the
/// resulting shared library, and returns the model handle.
CompiledModel compile_and_load(CppAD::ADFun<CGScalar>& fun,
                               const std::string& model_name);

}  // namespace detail

/// CppADCodeGen-backed automatic-differentiation implementation.
///
/// The templated constructor records the user's functor into an ADFun<CG<double>>,
/// then delegates to the non-template compile_and_load() helper for the JIT
/// pipeline (source generation → GCC compilation → dlopen).
///
/// Jacobian and Hessian evaluation use permutations precomputed at construction for O(nnz) sparse output aligned to the sparsity pattern.
class CppADCGBackend : public ADBackend {
 public:
    /// Records `function(z, p)` as a CppAD tape over a COMBINED domain
    /// z_combined = [x_0..x_{nv-1}, p_0..p_{np-1}] and JIT-compiles ONCE.
    ///
    /// The public surface of this backend operates on decision variables x ONLY
    /// (size nv = input_size).  Parameters are a separate out-of-band channel:
    ///   - input_size()  returns nv (not nv+np).
    ///   - eval(x) / eval_jacobian(x) / eval_hessian(x, w) accept x of size nv.
    ///     Internally, parameter_values_ are spliced into the tail of the
    ///     combined vector before forwarding to the compiled model.
    ///   - jacobian_sparsity() and hessian_sparsity() expose ONLY x-columns
    ///     (col < nv and, for the Hessian, also row < nv).
    ///
    /// @param function           Callable with signature
    ///                           `std::vector<T> operator()(const std::vector<T>& z,
    ///                                                       const std::vector<T>& p)`
    ///                           templated on AD scalar type T.
    /// @param input_size         Number of decision variables (nv).
    /// @param parameter_size     Number of parameters (np).
    /// @param parameter_defaults Initial parameter values (size must equal np).
    /// @param model_name         Name for the JIT-compiled shared library (must
    ///                           be a valid C identifier, unique per process to
    ///                           avoid shared-library name collisions).
    template <typename F>
    CppADCGBackend(const F& function,
                   std::size_t input_size,
                   std::size_t parameter_size,
                   const std::vector<double>& parameter_defaults,
                   const std::string& model_name = "goss_model")
        : input_size_(input_size),
          parameter_size_(parameter_size),
          parameter_values_(parameter_defaults) {
        if (parameter_defaults.size() != parameter_size) {
            throw ADError(
                "CppADCGBackend: parameter_defaults.size() (" +
                std::to_string(parameter_defaults.size()) +
                ") != parameter_size (" + std::to_string(parameter_size) + ")");
        }

        // Build combined independent vector: z_combined = [x..., p...].
        // Parameters are NOT declared as CppAD dynamic parameters — this pinned
        // CppADCG version has no new_dynamic() on GenericModel.  Instead they
        // occupy the tail of the single independent vector and are injected at
        // every evaluation call by splicing parameter_values_ into the tail.
        const std::size_t combined_size = input_size + parameter_size;
        std::vector<detail::ADCG> z_combined(combined_size);
        CppAD::Independent(z_combined);

        // Split combined independent vector into decision variables and params.
        std::vector<detail::ADCG> z_x(z_combined.begin(),
                                       z_combined.begin() +
                                           static_cast<std::ptrdiff_t>(input_size));
        std::vector<detail::ADCG> z_p(z_combined.begin() +
                                           static_cast<std::ptrdiff_t>(input_size),
                                       z_combined.end());

        // Record the functor over (z_x, z_p) with the combined domain.
        std::vector<detail::ADCG> y = function(z_x, z_p);
        output_size_ = y.size();
        CppAD::ADFun<detail::CGScalar> fun(z_combined, y);
        fun.optimize();
        // JIT-compile; compile_and_load captures full-domain sparsity patterns.
        compiled_ = detail::compile_and_load(fun, model_name);

        // ----------------------------------------------------------------
        // Build jacobian_sparsity_ for the x-columns only (col < input_size).
        //
        // compiled_.jac_rows/jac_cols cover all nv+np columns.  We keep only
        // entries where col < nv, and build a permutation that maps each
        // x-only sorted position to its raw SparseJacobian index.
        // ----------------------------------------------------------------
        {
            const std::size_t full_jac_nnz = compiled_.jac_rows.size();

            // Collect (raw_row, raw_col) entries where col < input_size,
            // paired with their raw index so we can build the permutation.
            struct RawJacEntry {
                std::size_t row;
                std::size_t col;
                std::size_t raw_idx;  // index in the raw SparseJacobian output
                bool operator<(const RawJacEntry& other) const noexcept {
                    return row != other.row ? row < other.row : col < other.col;
                }
            };
            std::vector<RawJacEntry> x_only_entries;
            x_only_entries.reserve(full_jac_nnz);

            // Probe SparseJacobian with a zero combined vector to learn the raw
            // (row, col) ordering used by the compiled model.  The ordering is
            // fixed for a given compiled model; only values change between calls.
            const std::vector<double> combined_probe(combined_size, 0.0);
            std::vector<double> probe_values;
            std::vector<std::size_t> probe_rows, probe_cols;
            compiled_.model->SparseJacobian(combined_probe, probe_values,
                                            probe_rows, probe_cols);

            if (probe_rows.size() != full_jac_nnz) {
                throw ADError(
                    "CppADCodeGen (parametric): SparseJacobian returned " +
                    std::to_string(probe_rows.size()) +
                    " entries but JacobianSparsity reported " +
                    std::to_string(full_jac_nnz));
            }

            for (std::size_t raw_k = 0; raw_k < full_jac_nnz; ++raw_k) {
                // Keep only x-column entries (drop param columns, col >= nv).
                if (probe_cols[raw_k] < input_size) {
                    x_only_entries.push_back(
                        {probe_rows[raw_k], probe_cols[raw_k], raw_k});
                }
            }

            // Sort x-only entries by (row, col) for deterministic ordering.
            std::sort(x_only_entries.begin(), x_only_entries.end());

            const std::size_t x_jac_nnz = x_only_entries.size();
            jacobian_sparsity_.reserve(x_jac_nnz);
            jac_perm_.resize(x_jac_nnz);
            for (std::size_t sorted_k = 0; sorted_k < x_jac_nnz; ++sorted_k) {
                jacobian_sparsity_.emplace_back(x_only_entries[sorted_k].row,
                                                x_only_entries[sorted_k].col);
                // Maps sorted x-only position → raw index in SparseJacobian output.
                jac_perm_[sorted_k] = x_only_entries[sorted_k].raw_idx;
            }
        }

        // ----------------------------------------------------------------
        // Build hessian_sparsity_ for x-only lower triangle:
        //   keep entries where row < input_size AND col < input_size AND row >= col.
        //
        // compiled_.hess_rows/hess_cols hold the full symmetric pattern over
        // the combined [x, p] domain.  We filter to x-only, take lower-triangle,
        // and build hess_perm_ mapping sorted lower-triangle positions to the
        // raw SparseHessian output index.
        // ----------------------------------------------------------------
        {
            const std::size_t full_hess_full_nnz = compiled_.hess_rows.size();

            // Probe SparseHessian with a zero combined vector and unit weights
            // to learn the raw ordering of the compiled model.
            const std::vector<double> x_probe(combined_size, 0.0);
            const std::vector<double> w_probe(output_size_, 1.0);
            std::vector<double> probe_values;
            std::vector<std::size_t> probe_rows, probe_cols;
            compiled_.model->SparseHessian(x_probe, w_probe,
                                           probe_values, probe_rows, probe_cols);

            if (probe_rows.size() != full_hess_full_nnz) {
                throw ADError(
                    "CppADCodeGen (parametric): SparseHessian probe returned " +
                    std::to_string(probe_rows.size()) +
                    " entries but HessianSparsity reported " +
                    std::to_string(full_hess_full_nnz));
            }

            // Collect raw entries that are in the x-only lower triangle
            // (row < nv && col < nv && row >= col), keeping the raw_idx.
            struct RawHessEntry {
                std::size_t row;
                std::size_t col;
                std::size_t raw_idx;
                bool operator<(const RawHessEntry& other) const noexcept {
                    return row != other.row ? row < other.row : col < other.col;
                }
            };
            std::vector<RawHessEntry> x_lower_entries;
            x_lower_entries.reserve(full_hess_full_nnz);

            for (std::size_t raw_k = 0; raw_k < full_hess_full_nnz; ++raw_k) {
                const std::size_t r = probe_rows[raw_k];
                const std::size_t c = probe_cols[raw_k];
                // Keep only entries entirely within the x-block and in the
                // lower triangle (r >= c).  Entries touching param indices
                // (r >= nv or c >= nv) are param-sensitivities and are dropped.
                if (r < input_size && c < input_size && r >= c) {
                    x_lower_entries.push_back({r, c, raw_k});
                }
            }

            // Sort by (row, col) for deterministic ordering.
            std::sort(x_lower_entries.begin(), x_lower_entries.end());

            const std::size_t x_hess_lower_nnz = x_lower_entries.size();
            hessian_sparsity_.reserve(x_hess_lower_nnz);
            hess_perm_.resize(x_hess_lower_nnz);
            for (std::size_t sorted_k = 0; sorted_k < x_hess_lower_nnz; ++sorted_k) {
                hessian_sparsity_.emplace_back(x_lower_entries[sorted_k].row,
                                               x_lower_entries[sorted_k].col);
                // Maps sorted x-only lower-triangle position → raw index.
                hess_perm_[sorted_k] = x_lower_entries[sorted_k].raw_idx;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Single-argument (non-parametric) constructor — UNCHANGED from original.
    // All existing non-parametric callers use this path; its behavior must
    // remain byte-identical to the pre-Task-2 version.
    // -----------------------------------------------------------------------
    template <typename F>
    CppADCGBackend(const F& function, std::size_t input_size,
                   const std::string& model_name = "goss_model")
        : input_size_(input_size) {
        // Record the function as an ADFun<CG<double>>.
        std::vector<detail::ADCG> x(input_size);
        CppAD::Independent(x);
        std::vector<detail::ADCG> y = function(x);
        output_size_ = y.size();
        CppAD::ADFun<detail::CGScalar> fun(x, y);
        // Optimize the tape for faster source generation.
        fun.optimize();
        // JIT-compile and load; compile_and_load also captures jac_rows/jac_cols.
        compiled_ = detail::compile_and_load(fun, model_name);

        // Build jacobian_sparsity_ from the captured rows/cols, sorted by
        // (row, col) so the ordering is deterministic and stable across calls.
        const std::size_t jac_nnz = compiled_.jac_rows.size();
        jacobian_sparsity_.reserve(jac_nnz);
        for (std::size_t k = 0; k < jac_nnz; ++k) {
            jacobian_sparsity_.emplace_back(compiled_.jac_rows[k],
                                            compiled_.jac_cols[k]);
        }
        std::sort(jacobian_sparsity_.begin(), jacobian_sparsity_.end());

        // Build hessian_sparsity_ (lower triangle only: row >= col) from the
        // full symmetric pattern captured by compile_and_load.
        // CppADCodeGen's HessianSparsity() returns the full symmetric pattern,
        // so we filter here.  Entries are sorted by (row, col) for stability.
        {
            const std::size_t hess_full_nnz = compiled_.hess_rows.size();
            for (std::size_t k = 0; k < hess_full_nnz; ++k) {
                const std::size_t r = compiled_.hess_rows[k];
                const std::size_t c = compiled_.hess_cols[k];
                if (r >= c) {
                    hessian_sparsity_.emplace_back(r, c);
                }
            }
            std::sort(hessian_sparsity_.begin(), hessian_sparsity_.end());
        }

        // Precompute jac_perm_ once by probing SparseJacobian with a zero
        // input to learn the exact raw (row,col) ordering the library uses.
        // CppADCodeGen returns a fixed (row,col) sequence for a given compiled
        // model regardless of input value — only values change across calls.
        //
        // jac_perm_[sorted_idx] = raw_idx means:
        //   aligned_values[sorted_idx] = raw_values[jac_perm_[sorted_idx]]
        //
        // This lets eval_jacobian run in O(nnz) with no heap allocations
        // beyond the result vector.
        {
            const std::vector<double> x_probe(input_size_, 0.0);
            std::vector<double> probe_values;
            std::vector<std::size_t> probe_rows, probe_cols;
            compiled_.model->SparseJacobian(x_probe, probe_values,
                                            probe_rows, probe_cols);

            if (probe_rows.size() != jac_nnz) {
                throw ADError(
                    "CppADCodeGen: SparseJacobian returned " +
                    std::to_string(probe_rows.size()) +
                    " entries but JacobianSparsity reported " +
                    std::to_string(jac_nnz));
            }

            // Build a lookup from (row,col) to its raw index.
            // Use the same sorted order as jacobian_sparsity_ to construct
            // a mapping from each sorted position to the raw index.
            jac_perm_.resize(jac_nnz);
            for (std::size_t raw_k = 0; raw_k < jac_nnz; ++raw_k) {
                // Binary-search for this raw coordinate in our sorted pattern.
                const auto target = std::make_pair(probe_rows[raw_k],
                                                   probe_cols[raw_k]);
                const auto it = std::lower_bound(
                    jacobian_sparsity_.begin(), jacobian_sparsity_.end(),
                    target);
                if (it == jacobian_sparsity_.end() || *it != target) {
                    throw ADError(
                        "CppADCodeGen: SparseJacobian returned coordinate (" +
                        std::to_string(probe_rows[raw_k]) + "," +
                        std::to_string(probe_cols[raw_k]) +
                        ") not present in JacobianSparsity output");
                }
                const std::size_t sorted_idx = static_cast<std::size_t>(
                    it - jacobian_sparsity_.begin());
                jac_perm_[sorted_idx] = raw_k;
            }
        }

        // Precompute hess_perm_ once by probing SparseHessian with a zero
        // input and unit weights.  SparseHessian returns (values, rows, cols)
        // in the SAME ordering as HessianSparsity (full symmetric), and the
        // ordering is fixed for the compiled model.
        //
        // We probe with zero input and uniform weights (all 1.0) to enumerate
        // the raw (row,col) pairs, then build a mapping from each position in
        // the sorted lower-triangle hessian_sparsity_ to its raw index.
        //
        // hess_perm_[sorted_lower_idx] = raw_full_idx means:
        //   aligned_values[sorted_lower_idx] = raw_values[hess_perm_[sorted_lower_idx]]
        {
            const std::vector<double> x_probe(input_size_, 0.0);
            const std::vector<double> w_probe(output_size_, 1.0);
            std::vector<double> probe_values;
            std::vector<std::size_t> probe_rows, probe_cols;
            compiled_.model->SparseHessian(x_probe, w_probe,
                                           probe_values, probe_rows, probe_cols);

            if (probe_rows.size() != compiled_.hess_rows.size()) {
                throw ADError(
                    "CppADCodeGen: SparseHessian probe returned " +
                    std::to_string(probe_rows.size()) +
                    " entries but HessianSparsity reported " +
                    std::to_string(compiled_.hess_rows.size()));
            }

            // probe_rows/cols contains the full symmetric pattern in raw order.
            // We need to map each entry of hessian_sparsity_ (lower triangle,
            // sorted) to its raw index in probe_rows/probe_cols.

            // Build a lookup: (row,col) -> raw_index, covering all raw entries.
            // Use a sorted temporary vector for binary search.
            struct RawEntry {
                std::size_t row;
                std::size_t col;
                std::size_t raw_idx;
                bool operator<(const RawEntry& o) const noexcept {
                    return row != o.row ? row < o.row : col < o.col;
                }
            };
            const std::size_t raw_nnz = probe_rows.size();
            std::vector<RawEntry> raw_sorted;
            raw_sorted.reserve(raw_nnz);
            for (std::size_t k = 0; k < raw_nnz; ++k) {
                raw_sorted.push_back({probe_rows[k], probe_cols[k], k});
            }
            std::sort(raw_sorted.begin(), raw_sorted.end());

            const std::size_t hess_lower_nnz = hessian_sparsity_.size();
            hess_perm_.resize(hess_lower_nnz);
            for (std::size_t sorted_k = 0; sorted_k < hess_lower_nnz; ++sorted_k) {
                const auto [r, c] = hessian_sparsity_[sorted_k];
                // Binary-search for (r,c) in the sorted raw entries.
                RawEntry target{r, c, 0};
                const auto it = std::lower_bound(raw_sorted.begin(),
                                                 raw_sorted.end(), target);
                if (it == raw_sorted.end() || it->row != r || it->col != c) {
                    throw ADError(
                        "CppADCodeGen: SparseHessian probe missing lower-triangle "
                        "entry (" + std::to_string(r) + "," +
                        std::to_string(c) + ")");
                }
                hess_perm_[sorted_k] = it->raw_idx;
            }
        }
    }

    std::size_t input_size()  const override { return input_size_; }
    std::size_t output_size() const override { return output_size_; }

    /// Returns the number of injectable parameters.  Zero for non-parametric
    /// backends (constructed via the single-argument constructor).
    std::size_t num_parameters() const override { return parameter_size_; }

    /// Injects the parameter vector for all subsequent eval / eval_jacobian /
    /// eval_hessian calls.  The values are stored in parameter_values_ and
    /// spliced into the tail of the combined vector at each call.
    /// Throws ADError if parameter_values.size() != parameter_size_.
    void set_parameters(const std::vector<double>& parameter_values) override {
        if (parameter_values.size() != parameter_size_) {
            throw ADError(
                "set_parameters: expected " + std::to_string(parameter_size_) +
                " parameters but got " +
                std::to_string(parameter_values.size()));
        }
        parameter_values_ = parameter_values;
    }

    /// Evaluates f(x) using the JIT-compiled forward pass.
    ///
    /// For parametric backends: builds combined = [x, parameter_values_] and
    /// passes it to ForwardZero.  The full output vector is returned (output
    /// columns are unchanged by the parametric mechanism).
    /// For non-parametric backends: passes x directly (parameter_size_ == 0,
    /// combined == x).
    std::vector<double> eval(const std::vector<double>& x) const override {
        if (x.size() != input_size_) {
            throw ADError("eval: x.size() (" + std::to_string(x.size()) +
                          ") != input_size (" + std::to_string(input_size_) + ")");
        }
        // Build combined vector [x, p] for the compiled model.
        // For non-parametric backends parameter_size_ == 0, so combined == x.
        const std::vector<double>& combined = build_combined_vector(x);
        return compiled_.model->ForwardZero(combined);
    }

    /// Returns the sorted (row, col) pairs of Jacobian non-zeros.
    /// The k-th pair corresponds to eval_jacobian()[k].
    /// For parametric backends, only x-columns (col < input_size_) are exposed.
    const SparsityPattern& jacobian_sparsity() const override {
        return jacobian_sparsity_;
    }

    /// Evaluates the Jacobian at x.
    ///
    /// SparseJacobian(combined, jac, rows, cols) fills raw_values in the
    /// library's internal ordering over the combined [x, p] domain.
    /// jac_perm_ (precomputed at construction) maps each sorted x-only pattern
    /// index to its raw index, so alignment is O(nnz) with no per-call heap
    /// allocation beyond the output vector.
    /// For non-parametric backends the combined vector equals x, and the full
    /// Jacobian is returned (no column filtering needed).
    std::vector<double> eval_jacobian(const std::vector<double>& x) const override {
        if (x.size() != input_size_) {
            throw ADError("eval_jacobian: x.size() (" + std::to_string(x.size()) +
                          ") != input_size (" + std::to_string(input_size_) + ")");
        }
        const std::vector<double>& combined = build_combined_vector(x);
        std::vector<double> raw_values;
        std::vector<std::size_t> raw_rows, raw_cols;
        compiled_.model->SparseJacobian(combined, raw_values, raw_rows, raw_cols);

        if (raw_values.size() < jac_perm_.size()) {
            throw ADError(
                "eval_jacobian: SparseJacobian returned " +
                std::to_string(raw_values.size()) +
                " values, expected at least " +
                std::to_string(jac_perm_.size()));
        }

        // Apply precomputed permutation: aligned[k] = raw[jac_perm_[k]].
        // For non-parametric backends jac_perm_ covers the full pattern.
        // For parametric backends jac_perm_ covers only x-columns.
        const std::size_t nnz = jac_perm_.size();
        std::vector<double> aligned_values(nnz);
        for (std::size_t k = 0; k < nnz; ++k) {
            aligned_values[k] = raw_values[jac_perm_[k]];
        }
        return aligned_values;
    }

    /// Returns the sorted lower-triangle (row >= col) (row, col) pairs of
    /// Hessian non-zeros.  The k-th pair corresponds to eval_hessian()[k].
    /// For parametric backends, only entries with both row < input_size_ and
    /// col < input_size_ are exposed (param-touching entries dropped).
    const SparsityPattern& hessian_sparsity() const override {
        return hessian_sparsity_;
    }

    /// Evaluates the weighted-sum Hessian ∇²(Σ wᵢ fᵢ) at x.
    ///
    /// SparseHessian(combined, weights, values, rows, cols) fills raw_values
    /// in the library's internal ordering (full symmetric over [x, p] domain).
    /// hess_perm_ (precomputed at construction) maps each sorted x-only
    /// lower-triangle position to the corresponding raw index, so alignment
    /// is O(nnz) with no per-call heap allocation beyond the output vector.
    std::vector<double> eval_hessian(const std::vector<double>& x,
                                     const std::vector<double>& weights) const override {
        if (x.size() != input_size_) {
            throw ADError("eval_hessian: x.size() (" + std::to_string(x.size()) +
                          ") != input_size (" + std::to_string(input_size_) + ")");
        }
        if (weights.size() != output_size_) {
            throw ADError(
                "eval_hessian: weights size (" +
                std::to_string(weights.size()) +
                ") must equal output_size (" +
                std::to_string(output_size_) + ")");
        }

        const std::vector<double>& combined = build_combined_vector(x);
        std::vector<double> raw_values;
        std::vector<std::size_t> raw_rows, raw_cols;
        compiled_.model->SparseHessian(combined, weights, raw_values,
                                       raw_rows, raw_cols);

        // For the non-parametric path, validate against the total raw size.
        // For the parametric path the raw output is larger (full [x,p] domain),
        // but hess_perm_ indices are valid as long as they are < raw_values.size().
        if (parameter_size_ == 0 &&
            raw_values.size() != compiled_.hess_rows.size()) {
            throw ADError(
                "eval_hessian: SparseHessian returned " +
                std::to_string(raw_values.size()) +
                " values, expected " +
                std::to_string(compiled_.hess_rows.size()));
        }
        // Defensive guard for the parametric path: each hess_perm_[k] must be a
        // valid index into raw_values. Mirrors the analogous guard in eval_jacobian.
        // A mismatch would indicate a sparsity-pattern change between construction
        // and evaluation time (should never happen for well-formed models).
        if (parameter_size_ > 0) {
            for (std::size_t k = 0; k < hess_perm_.size(); ++k) {
                if (hess_perm_[k] >= raw_values.size()) {
                    throw ADError(
                        "eval_hessian: hess_perm_ index out of range in raw SparseHessian output");
                }
            }
        }

        // Apply precomputed permutation: aligned[k] = raw[hess_perm_[k]].
        // For parametric backends hess_perm_ covers only x-only lower-triangle
        // entries; the param-touching entries in raw_values are ignored.
        const std::size_t hess_lower_nnz = hess_perm_.size();
        std::vector<double> aligned_values(hess_lower_nnz);
        for (std::size_t k = 0; k < hess_lower_nnz; ++k) {
            aligned_values[k] = raw_values[hess_perm_[k]];
        }
        return aligned_values;
    }

 protected:
    std::size_t input_size_;
    std::size_t output_size_ = 0;
    /// Number of injectable parameters.  Zero for non-parametric backends.
    std::size_t parameter_size_ = 0;
    /// Current parameter values; size == parameter_size_.  Empty for
    /// non-parametric backends.  Updated by set_parameters().
    std::vector<double> parameter_values_;
    detail::CompiledModel compiled_;
    SparsityPattern jacobian_sparsity_;
    SparsityPattern hessian_sparsity_;
    /// Precomputed permutation: jac_perm_[sorted_idx] = raw_idx.
    /// Maps each position in the sorted jacobian_sparsity_ to the corresponding
    /// index in the raw array returned by SparseJacobian().
    /// For parametric backends, sorted_idx iterates only over x-columns;
    /// raw_idx is the index in the full [x,p] raw output.
    std::vector<std::size_t> jac_perm_;
    /// Precomputed permutation: hess_perm_[sorted_lower_idx] = raw_full_idx.
    /// Maps each position in the sorted lower-triangle hessian_sparsity_ to
    /// the corresponding index in the raw array returned by SparseHessian().
    /// For parametric backends, sorted_lower_idx iterates only over x-only
    /// lower-triangle entries; param-touching raw entries are ignored.
    std::vector<std::size_t> hess_perm_;

 private:
    /// Builds the combined [x, parameter_values_] vector for the compiled model.
    /// For non-parametric backends (parameter_size_ == 0) this is a no-op that
    /// returns a reference alias; callers must not retain the reference past the
    /// next mutation of parameter_values_.  Since parameter_size_ == 0 in that
    /// case the returned reference IS x, so no copy is made.
    ///
    /// For parametric backends (parameter_size_ > 0) a new vector is allocated.
    /// We store it in combined_buffer_ to allow the const method to write it.
    // TODO: could pre-allocate combined_buffer_ at construction to avoid the per-call resize
    const std::vector<double>& build_combined_vector(
        const std::vector<double>& x) const {
        if (parameter_size_ == 0) {
            // Non-parametric: pass x directly to the compiled model.
            return x;
        }
        // Parametric: splice [x, parameter_values_] into a single buffer.
        combined_buffer_.resize(input_size_ + parameter_size_);
        std::copy(x.begin(), x.end(), combined_buffer_.begin());
        std::copy(parameter_values_.begin(), parameter_values_.end(),
                  combined_buffer_.begin() +
                      static_cast<std::ptrdiff_t>(input_size_));
        return combined_buffer_;
    }

    /// Mutable scratch buffer for build_combined_vector() in the parametric path.
    /// Declared mutable so const eval methods can write to it without losing the
    /// const qualifier on the public interface.
    /// NOTE: shared mutable state — this backend is NOT thread-safe for concurrent
    /// eval calls on the same instance; the sweep harness uses per-worker instances.
    mutable std::vector<double> combined_buffer_;
};

}  // namespace goss::ad
