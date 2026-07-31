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
    /// Records `function` as a CppAD tape and JIT-compiles it.
    ///
    /// @param function     Callable with signature
    ///                     `std::vector<T> operator()(const std::vector<T>&)`
    ///                     templated on AD scalar type T.
    /// @param input_size   Number of independent variables.
    /// @param model_name   Name for the generated shared library (must be a
    ///                     valid C identifier).  The name determines the
    ///                     JIT-compiled shared-library filename, so it must be
    ///                     unique across concurrently-built models/processes to
    ///                     avoid collision; parallel test runs must use distinct
    ///                     names.
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

    /// Evaluates f(x) using the JIT-compiled forward pass.
    std::vector<double> eval(const std::vector<double>& x) const override {
        if (x.size() != input_size_) {
            throw ADError("eval: x.size() (" + std::to_string(x.size()) +
                          ") != input_size (" + std::to_string(input_size_) + ")");
        }
        return compiled_.model->ForwardZero(x);
    }

    /// Returns the sorted (row, col) pairs of Jacobian non-zeros.
    /// The k-th pair corresponds to eval_jacobian()[k].
    const SparsityPattern& jacobian_sparsity() const override {
        return jacobian_sparsity_;
    }

    /// Evaluates the Jacobian at x.
    ///
    /// SparseJacobian(x, jac, rows, cols) fills raw_values in the library's
    /// internal ordering.  jac_perm_ (precomputed at construction) maps each
    /// sorted-pattern index to its raw index, so alignment is O(nnz) with no
    /// per-call heap allocation beyond the output vector.
    std::vector<double> eval_jacobian(const std::vector<double>& x) const override {
        if (x.size() != input_size_) {
            throw ADError("eval_jacobian: x.size() (" + std::to_string(x.size()) +
                          ") != input_size (" + std::to_string(input_size_) + ")");
        }
        std::vector<double> raw_values;
        std::vector<std::size_t> raw_rows, raw_cols;
        compiled_.model->SparseJacobian(x, raw_values, raw_rows, raw_cols);

        if (raw_values.size() != jac_perm_.size()) {
            throw ADError(
                "eval_jacobian: SparseJacobian returned " +
                std::to_string(raw_values.size()) +
                " values, expected " +
                std::to_string(jac_perm_.size()));
        }

        // Apply precomputed permutation: aligned[k] = raw[jac_perm_[k]]
        const std::size_t nnz = jac_perm_.size();
        std::vector<double> aligned_values(nnz);
        for (std::size_t k = 0; k < nnz; ++k) {
            aligned_values[k] = raw_values[jac_perm_[k]];
        }
        return aligned_values;
    }

    /// Returns the sorted lower-triangle (row >= col) (row, col) pairs of
    /// Hessian non-zeros.  The k-th pair corresponds to eval_hessian()[k].
    const SparsityPattern& hessian_sparsity() const override {
        return hessian_sparsity_;
    }

    /// Evaluates the weighted-sum Hessian ∇²(Σ wᵢ fᵢ) at x.
    ///
    /// SparseHessian(x, weights, values, rows, cols) fills raw_values in the
    /// library's internal ordering (full symmetric).  hess_perm_ (precomputed
    /// at construction) maps each sorted lower-triangle position to the
    /// corresponding raw index, so alignment is O(nnz) with no per-call heap
    /// allocation beyond the output vector.
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

        std::vector<double> raw_values;
        std::vector<std::size_t> raw_rows, raw_cols;
        compiled_.model->SparseHessian(x, weights, raw_values,
                                       raw_rows, raw_cols);

        if (raw_values.size() != compiled_.hess_rows.size()) {
            throw ADError(
                "eval_hessian: SparseHessian returned " +
                std::to_string(raw_values.size()) +
                " values, expected " +
                std::to_string(compiled_.hess_rows.size()));
        }

        // Apply precomputed permutation: aligned[k] = raw[hess_perm_[k]]
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
    detail::CompiledModel compiled_;
    SparsityPattern jacobian_sparsity_;
    SparsityPattern hessian_sparsity_;
    /// Precomputed permutation: jac_perm_[sorted_idx] = raw_idx.
    /// Maps each position in the sorted jacobian_sparsity_ to the corresponding
    /// index in the raw array returned by SparseJacobian().
    std::vector<std::size_t> jac_perm_;
    /// Precomputed permutation: hess_perm_[sorted_lower_idx] = raw_full_idx.
    /// Maps each position in the sorted lower-triangle hessian_sparsity_ to
    /// the corresponding index in the raw array returned by SparseHessian()
    /// (which uses the full symmetric ordering from HessianSparsity()).
    std::vector<std::size_t> hess_perm_;
};

}  // namespace goss::ad
