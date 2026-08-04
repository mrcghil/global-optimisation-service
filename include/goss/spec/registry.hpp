// include/goss/spec/registry.hpp
#pragma once
#include <functional>
#include <map>
#include <string>
#include <vector>
#include "goss/model/model.hpp"
#include "goss/spec/errors.hpp"
#include "goss/spec/specs.hpp"
#include "goss/transcription/ocp_problem.hpp"  // Mesh
#include "goss/transcription/transcription.hpp"  // CompiledOcp

namespace goss::spec {

/// Everything the executor needs to solve a run and unpack its trajectory.
/// Move-only (CompiledOcp owns a unique_ptr<NLPProblem>).
struct BuiltProblem {
    model::Model               model;     // names, bounds — drives sim::linear_guess
    transcription::CompiledOcp compiled;  // problem + layout + validator
    transcription::Mesh        mesh;      // node times for trajectory extraction
    std::string                scheme;    // resolved scheme name (provenance)
};

/// Name+version → builder registry.  A builder compiles a fresh BuiltProblem for
/// a given DiscretizationSpec, so ONE registered problem can be compiled at many
/// meshes/schemes.  Multiple versions of the same problem coexist under distinct
/// ProblemKeys, so a spec always resolves to the intended compiled artifact.
///
/// Builder contract: the model_name passed to compile/compile_dispatch doubles as
/// the JIT .so path, so a builder MUST derive a name unique to the discretization
/// (e.g. include scheme + num_intervals) — otherwise a second build at a different
/// mesh collides with the first cached library and evaluations fail with a size
/// mismatch.
class ProblemRegistry {
 public:
    using Builder = std::function<BuiltProblem(const DiscretizationSpec&)>;

    /// Registers a builder under `key`.  Throws SpecError if the key is already
    /// registered (a version bump, not a silent overwrite, is the intended way
    /// to change a problem).
    void register_problem(const ProblemKey& key, Builder builder) {
        if (builders_.count(key))
            throw SpecError("ProblemRegistry: '" + key.name + "@" + key.version +
                            "' is already registered");
        builders_.emplace(key, std::move(builder));
    }

    bool contains(const ProblemKey& key) const { return builders_.count(key) != 0; }

    /// Builds the problem for `key` at `discretization`.  Throws SpecError if the
    /// key is unknown.
    BuiltProblem build(const ProblemKey& key,
                       const DiscretizationSpec& discretization) const {
        const auto it = builders_.find(key);
        if (it == builders_.end())
            throw SpecError("ProblemRegistry: unknown problem '" + key.name + "@" +
                            key.version + "' (register it before executing)");
        return it->second(discretization);
    }

    std::vector<ProblemKey> keys() const {
        std::vector<ProblemKey> out;
        out.reserve(builders_.size());
        for (const auto& entry : builders_) out.push_back(entry.first);
        return out;
    }

 private:
    std::map<ProblemKey, Builder> builders_;
};

}  // namespace goss::spec
