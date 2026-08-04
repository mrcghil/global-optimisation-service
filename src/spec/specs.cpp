// src/spec/specs.cpp
#include "goss/spec/specs.hpp"
#include <cstddef>
#include <vector>
#include "goss/sim/sweep.hpp"
#include "goss/spec/errors.hpp"

namespace goss::spec {

std::vector<RunSpec> SweepSpec::expand() const {
    if (axes.empty())
        return {base};  // a degenerate sweep with no axes is just the base run

    for (const Axis& axis : axes)
        if (axis.values.empty())
            throw SpecError("SweepSpec::expand: axis '" + axis.parameter + "' is empty");

    // Collect the value combinations as index-aligned rows: combos[r][a] is the
    // value of axis a in combination r.
    std::vector<std::vector<double>> combos;
    if (combinator == "product") {
        std::vector<std::vector<double>> axis_values;
        axis_values.reserve(axes.size());
        for (const Axis& axis : axes) axis_values.push_back(axis.values);
        combos = goss::sim::make_grid(axis_values);  // reuse the one grid algorithm
    } else if (combinator == "zip") {
        const std::size_t length = axes.front().values.size();
        for (const Axis& axis : axes)
            if (axis.values.size() != length)
                throw SpecError("SweepSpec::expand: zip requires equal-length axes");
        combos.reserve(length);
        for (std::size_t row = 0; row < length; ++row) {
            std::vector<double> combo;
            combo.reserve(axes.size());
            for (const Axis& axis : axes) combo.push_back(axis.values[row]);
            combos.push_back(std::move(combo));
        }
    } else {
        throw SpecError("SweepSpec::expand: unknown combinator '" + combinator +
                        "' (expected 'product' or 'zip')");
    }

    std::vector<RunSpec> runs;
    runs.reserve(combos.size());
    for (const std::vector<double>& combo : combos) {
        RunSpec run = base;
        for (std::size_t a = 0; a < axes.size(); ++a)
            run.parameters[axes[a].parameter] = combo[a];  // overlay by name
        runs.push_back(std::move(run));
    }
    return runs;
}

}  // namespace goss::spec
