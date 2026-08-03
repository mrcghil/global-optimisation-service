// tests/ad/test_dynamic_param_spike.cpp
//
// SPIKE: prove compile-once parameter injection mechanism for the
// parameter-binding feature (Task 1 of parameter-binding plan).
//
// Goal: record f(x; p) = p0 * x0^2 ONCE, JIT-compile ONCE, then evaluate at
// two different parameter values WITHOUT re-recording/re-compiling.
//
// PRIMARY mechanism (CppAD dynamic parameters via new_dynamic on GenericModel)
// was investigated first.  CppADCodeGen's GenericModel interface (checked via
// /usr/local/include/cppad/cg/model/generic_model.hpp) does NOT expose a
// new_dynamic() method: the compiled .so knows only Domain(), Range(), and the
// evaluation / derivative routines.  CppAD::ADFun<Base>::new_dynamic() exists
// in the core CppAD library but that object is discarded once
// DynamicModelLibraryProcessor::createDynamicLibrary() runs — the generated C
// source hard-codes parameter values at code-generation time, so there is no
// runtime injection hook in the compiled model.
//
// FALLBACK mechanism (pinned decision variables) is therefore proven here:
//   - Append the parameter(s) as extra independent variables z = [x..., p...].
//   - JIT-compile ONCE over the full combined vector.
//   - "Set a parameter" = supply the desired value in the p-slot of the x
//     vector at every evaluation call (lower==upper bound enforced by the
//     solver layer; the AD model itself is oblivious).
//
// This test proves f = p0 * x0^2 gives 12 (p0=3) and 20 (p0=5) on the SAME
// compiled model by changing only the p-slot of the input vector.

#include <gtest/gtest.h>
#include <cppad/cg.hpp>
#include <vector>

namespace {
using CGD  = CppAD::cg::CG<double>;
using ADCG = CppAD::AD<CGD>;
}  // namespace

TEST(DynamicParamSpike, ValueAndJacobianTrackInjectedParameterWithoutRecompile) {
    // Layout of the combined independent vector z = [x0, p0].
    // x0 is the decision variable; p0 is the "parameter" treated as a pinned var.
    const std::size_t num_vars   = 1;  // x0
    const std::size_t num_params = 1;  // p0
    const std::size_t num_total  = num_vars + num_params;  // 2

    // --- RECORD ONCE ---
    // Record f(z) = z[1] * z[0]^2 over the combined vector z = [x0, p0].
    // Parameters are appended as extra independent variables; the solver layer
    // pins them via equal lower/upper bounds and never lets the optimizer move them.
    std::vector<ADCG> az(num_total);
    az[0] = 2.0;  // x0 initial value (used only during recording)
    az[1] = 1.0;  // p0 initial value (used only during recording)
    CppAD::Independent(az);
    std::vector<ADCG> ay(1);
    // f(x; p) = p0 * x0^2  — expressed over the combined vector
    ay[0] = az[1] * az[0] * az[0];
    CppAD::ADFun<CGD> fun(az, ay);
    fun.optimize();

    // --- JIT-COMPILE ONCE via the same pipeline as compile_and_load ---
    CppAD::cg::ModelCSourceGen<double> source_gen(fun, "spike_model");
    source_gen.setCreateForwardZero(true);
    source_gen.setCreateSparseJacobian(true);
    CppAD::cg::ModelLibraryCSourceGen<double> library_gen(source_gen);
    CppAD::cg::GccCompiler<double> compiler;
    CppAD::cg::DynamicModelLibraryProcessor<double> processor(library_gen, "spike_model");
    auto library = processor.createDynamicLibrary(compiler);
    auto model   = library->model("spike_model");
    ASSERT_TRUE(model);

    // The compiled model covers the combined domain [x..., p...].
    ASSERT_EQ(model->Domain(), num_total);

    // --- INJECT p0 = 3.0 by passing it in the p-slot of the x-vector ---
    // z = [x0=2.0, p0=3.0]  ->  f = 3 * 4 = 12
    {
        std::vector<double> z{2.0, 3.0};
        std::vector<double> y = model->ForwardZero(z);
        EXPECT_NEAR(y[0], 12.0, 1e-12);
    }

    // --- INJECT p0 = 5.0 on the SAME compiled model ---
    // z = [x0=2.0, p0=5.0]  ->  f = 5 * 4 = 20
    {
        std::vector<double> z{2.0, 5.0};
        std::vector<double> y = model->ForwardZero(z);
        EXPECT_NEAR(y[0], 20.0, 1e-12);
    }

    // --- Verify Jacobian also reflects the injected parameter value ---
    // df/dx0 = 2*p0*x0 = 2*5*2 = 20;  df/dp0 = x0^2 = 4
    {
        std::vector<double>   z{2.0, 5.0};
        std::vector<size_t>   jac_rows, jac_cols;
        model->JacobianSparsity(jac_rows, jac_cols);
        std::vector<double> jac_vals;
        model->SparseJacobian(z, jac_vals, jac_rows, jac_cols);

        // Find df/dx0 (col 0) in the sparse result
        double dfdx0 = 0.0;
        for (std::size_t k = 0; k < jac_rows.size(); ++k) {
            if (jac_rows[k] == 0 && jac_cols[k] == 0) dfdx0 = jac_vals[k];
        }
        // df/dx0 = 2 * p0 * x0 = 2 * 5 * 2 = 20
        EXPECT_NEAR(dfdx0, 20.0, 1e-12);
    }
}
