// include/goss/ad/cppadcg_backend.hpp
#pragma once
// Use the umbrella header — NOT the sub-header <cppad/cg/cg.hpp>.
// Task 6 confirmed <cppad/cg.hpp> is the correct include path.
#include <cppad/cg.hpp>
#include <memory>
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
struct CompiledModel {
    std::unique_ptr<CppAD::cg::DynamicLib<double>> library;
    std::unique_ptr<CppAD::cg::GenericModel<double>> model;
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
/// Jacobian and Hessian methods are throwing placeholders filled in Tasks 8-9.
class CppADCGBackend : public ADBackend {
 public:
    /// Records `function` as a CppAD tape and JIT-compiles it.
    ///
    /// @param function     Callable with signature
    ///                     `std::vector<T> operator()(const std::vector<T>&)`
    ///                     templated on AD scalar type T.
    /// @param input_size   Number of independent variables.
    /// @param model_name   Name for the generated shared library (must be a
    ///                     valid C identifier).
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
        // JIT-compile and load.
        compiled_ = detail::compile_and_load(fun, model_name);
    }

    std::size_t input_size()  const override { return input_size_; }
    std::size_t output_size() const override { return output_size_; }

    /// Evaluates f(x) using the JIT-compiled forward pass.
    std::vector<double> eval(const std::vector<double>& x) const override {
        return compiled_.model->ForwardZero(x);
    }

    // ---- Placeholders filled in Task 8 ---- //

    const SparsityPattern& jacobian_sparsity() const override {
        throw ADError("jacobian_sparsity not yet implemented");
    }

    std::vector<double> eval_jacobian(const std::vector<double>&) const override {
        throw ADError("eval_jacobian not yet implemented");
    }

    // ---- Placeholders filled in Task 9 ---- //

    const SparsityPattern& hessian_sparsity() const override {
        throw ADError("hessian_sparsity not yet implemented");
    }

    std::vector<double> eval_hessian(const std::vector<double>&,
                                     const std::vector<double>&) const override {
        throw ADError("eval_hessian not yet implemented");
    }

 protected:
    std::size_t input_size_;
    std::size_t output_size_ = 0;
    detail::CompiledModel compiled_;
    // Reserved for Tasks 8-9:
    SparsityPattern jacobian_sparsity_;
    SparsityPattern hessian_sparsity_;
};

}  // namespace goss::ad
