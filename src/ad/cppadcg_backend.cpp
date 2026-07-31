// src/ad/cppadcg_backend.cpp
//
// Non-template compile+load plumbing for CppADCGBackend.
// The umbrella header <cppad/cg.hpp> already includes all necessary sub-headers
// (model_c_source_gen.hpp, model_library_c_source_gen.hpp,
//  dynamic_model_library_processor.hpp, compiler/gcc_compiler.hpp, etc.),
// so we only need the single include below.
#include "goss/ad/cppadcg_backend.hpp"

namespace goss::ad::detail {

CompiledModel compile_and_load(CppAD::ADFun<CGScalar>& fun,
                               const std::string& model_name) {
    try {
        // --- Source generation ---
        // ModelCSourceGen<double>(fun, name) generates C source for the model.
        CppAD::cg::ModelCSourceGen<double> source_gen(fun, model_name);
        // Enable ForwardZero (required for eval()); the rest are pre-enabled for
        // Tasks 8-9 so that the compiled library already contains them.
        source_gen.setCreateForwardZero(true);
        source_gen.setCreateJacobian(true);
        source_gen.setCreateHessian(true);
        source_gen.setCreateSparseJacobian(true);
        source_gen.setCreateSparseHessian(true);

        // --- Library source generation ---
        CppAD::cg::ModelLibraryCSourceGen<double> library_gen(source_gen);

        // --- JIT compilation ---
        // GccCompiler<double> defaults to /usr/bin/gcc (confirmed present in container).
        CppAD::cg::GccCompiler<double> compiler;

        // DynamicModelLibraryProcessor takes ownership-by-reference of library_gen
        // and writes the .so to a temp path derived from model_name.
        CppAD::cg::DynamicModelLibraryProcessor<double> processor(library_gen, model_name);

        // createDynamicLibrary compiles, links, and dlopen-loads the library.
        // Returns unique_ptr<DynamicLib<double>>; the library must outlive the model.
        CompiledModel result;
        result.library = processor.createDynamicLibrary(compiler);

        // lib->model(name) returns unique_ptr<GenericModel<double>>.
        // (API note: method is model(), NOT getModel() — confirmed from dynamiclib.hpp.)
        result.model = result.library->model(model_name);
        if (!result.model) {
            throw ADError("CppADCodeGen: failed to load generated model '" + model_name + "'");
        }

        // Capture the sparse Jacobian sparsity pattern once at construction.
        // JacobianSparsity(equations, variables) fills parallel arrays of row
        // (equations) and column (variables) indices for all non-zero entries.
        result.model->JacobianSparsity(result.jac_rows, result.jac_cols);

        return result;

    } catch (const CppAD::cg::CGException& error) {
        // Wrap CppADCodeGen exceptions in our domain error type.
        throw ADError(std::string("CppADCodeGen failure: ") + error.what());
    }
}

}  // namespace goss::ad::detail
