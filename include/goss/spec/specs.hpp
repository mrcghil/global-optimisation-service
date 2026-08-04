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

/// A parametric family over a base RunSpec. `combinator` is "product" (Cartesian
/// product of all axes) or "zip" (index-wise pairing; axes must be equal length).
struct SweepSpec {
    RunSpec           base;
    std::vector<Axis> axes;
    std::string       combinator = "product";
    std::string       label;

    /// Expands to one concrete RunSpec per parameter combination. Each expanded
    /// spec is `base` with the axis parameters overlaid onto base.parameters by
    /// name. Throws SpecError on an empty axis, an unknown combinator, or (for
    /// "zip") axes of differing length.
    std::vector<RunSpec> expand() const;
};

/// A named group of sweeps — the top-level campaign the user configures once.
struct CampaignSpec {
    std::string            name;
    std::vector<SweepSpec> sweeps;
};

}  // namespace goss::spec
