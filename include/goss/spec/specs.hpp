// include/goss/spec/specs.hpp
#pragma once
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace goss::spec {

/// Identifies a registered problem by name AND version. Multiple versions of the
/// same problem coexist so a spec always resolves to the correct compiled
/// artifact.
struct ProblemKey {
    std::string name;
    std::string version;

    bool operator==(const ProblemKey& other) const {
        return name == other.name && version == other.version;
    }
    bool operator<(const ProblemKey& other) const {
        return name < other.name || (name == other.name && version < other.version);
    }
};

/// Discretization / transcription settings. `scheme` selects the transcription
/// at compile time via compile_dispatch (hermite_simpson | trapezoidal | lgl).
/// `lgl_degree` is consulted only when scheme == "lgl".
struct DiscretizationSpec {
    std::string scheme = "hermite_simpson";
    double      t_initial = 0.0;
    double      t_final = 1.0;
    std::size_t num_intervals = 20;
    std::size_t lgl_degree = 0;
};

/// Solver selection and its tunables. Fields not relevant to the chosen `kind`
/// are ignored (ipopt uses tolerance/max_iterations/print_level; nlopt uses
/// max_evaluations/xtol_rel).
struct SolverSpec {
    std::string kind = "ipopt";        // "ipopt" | "nlopt"
    double      tolerance = 1e-8;      // ipopt "tol"
    int         max_iterations = 3000;  // ipopt "max_iter"
    int         print_level = 0;        // ipopt "print_level"
    int         max_evaluations = 20000;  // nlopt
    double      xtol_rel = 1e-8;        // nlopt
};

/// Initial-guess strategy. "linear" defers to sim::linear_guess; "explicit"
/// uses `values` verbatim (must equal the compiled layout's total_variables).
struct GuessSpec {
    std::string         kind = "linear";  // "linear" | "explicit"
    std::vector<double> values;
};

/// Where results are written. Empty `root` resolves at execute time
/// (GOSS_RESULTS_DIR env, else ./goss-results). `skip_if_exists` enables
/// content-addressed resume: an existing archive for the run_id is not re-solved.
struct StorageSpec {
    std::string root;
    bool        skip_if_exists = true;
};

/// A single solve: which problem/version, its named parameter bindings, the
/// discretization, solver, guess, storage, and an optional named image pipeline
/// (recorded only; not executed in this layer).
struct RunSpec {
    ProblemKey                    problem;
    std::map<std::string, double> parameters;
    DiscretizationSpec            discretization;
    SolverSpec                    solver;
    GuessSpec                     guess;
    StorageSpec                   storage;
    std::string                   image_pipeline;
    std::string                   label;
};

/// One named parameter axis of a sweep.
struct Axis {
    std::string         parameter;
    std::vector<double> values;
};

/// A group of axes whose values advance together (index-wise "zip"). All axes in
/// a group MUST have equal-length `values`. Groups are the building block of a
/// grid: axes zip WITHIN a group, and groups are combined by cartesian PRODUCT.
struct AxisGroup {
    std::vector<Axis> axes;
};

/// A parametric family over a base RunSpec. The preferred way to describe a
/// sweep is via `groups` (zip within, product across). The legacy `axes` +
/// `combinator` fields are kept for backward compatibility only.
struct SweepSpec {
    RunSpec base;

    /// Grouped axes: zip within a group, cartesian product across groups. This is
    /// the preferred way to describe a sweep. When non-empty, `axes`/`combinator`
    /// below are ignored.
    std::vector<AxisGroup> groups;

    /// DEPRECATED — backward-compat only; slated for removal once `groups` is
    /// validated as the superior model (a single group reproduces "zip",
    /// single-axis groups reproduce "product"). Do not add new callers.
    std::vector<Axis> axes;
    /// DEPRECATED — backward-compat only (see `axes`). "product" | "zip".
    std::string combinator = "product";

    std::string label;

    /// Expands to one concrete RunSpec per parameter combination. If `groups` is
    /// non-empty, axes zip within each group (equal-length required) and groups
    /// combine by cartesian product. Otherwise the legacy `axes` + `combinator`
    /// path is used. Throws SpecError on an empty axis, unequal-length axes in a
    /// group (or a zip), or an unknown combinator.
    std::vector<RunSpec> expand() const;
};

/// A named group of sweeps — the top-level campaign the user configures once.
struct CampaignSpec {
    std::string            name;
    std::vector<SweepSpec> sweeps;
};

}  // namespace goss::spec
