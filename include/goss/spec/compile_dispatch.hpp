// include/goss/spec/compile_dispatch.hpp
#pragma once
#include <string>
#include "goss/spec/errors.hpp"
#include "goss/spec/specs.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/transcription/legendre_gauss_lobatto.hpp"
#include "goss/transcription/transcription.hpp"  // CompiledOcp
#include "goss/transcription/trapezoidal.hpp"

namespace goss::spec {

/// Maps the DiscretizationSpec.scheme STRING to the correct compile-time
/// transcription type and compiles `ocp`.  This is the single place a runtime
/// scheme name becomes a static scheme choice, so the transcription layer keeps
/// its "no runtime scheme dispatch" invariant while callers get a string knob.
///
/// The mesh must already be baked into `ocp` (via Model::set_mesh from the same
/// DiscretizationSpec) before calling.  Throws SpecError for an unknown scheme.
template <typename OcpT>
transcription::CompiledOcp compile_dispatch(const OcpT& ocp,
                                            const DiscretizationSpec& discretization,
                                            const std::string& model_name) {
    if (discretization.scheme == "hermite_simpson")
        return transcription::HermiteSimpson::compile(ocp, model_name);
    if (discretization.scheme == "trapezoidal")
        return transcription::Trapezoidal::compile(ocp, model_name);
    if (discretization.scheme == "lgl")
        return transcription::LegendreGaussLobatto::compile(ocp, model_name);
    throw SpecError("compile_dispatch: unknown scheme '" + discretization.scheme +
                    "' (expected 'hermite_simpson', 'trapezoidal', or 'lgl')");
}

}  // namespace goss::spec
