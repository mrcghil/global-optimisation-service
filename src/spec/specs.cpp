// src/spec/specs.cpp
#include "goss/spec/specs.hpp"
#include <cstddef>
#include <vector>
#include "goss/sim/sweep.hpp"
#include "goss/spec/errors.hpp"

namespace goss::spec {

std::vector<RunSpec> SweepSpec::expand() const {
    // Preferred path: grouped axes (zip within a group, product across groups).
    if (!groups.empty()) {
        // Each group zips its axes into a list of per-combination rows, and
        // contributes one "meta-axis" whose entries are those rows. We then take
        // the cartesian product of the groups' row-indices via make_grid.
        std::vector<std::vector<std::vector<double>>> group_rows;  // group -> rows -> axis values
        std::vector<std::vector<std::string>> group_param_names;   // group -> axis parameter names
        group_rows.reserve(groups.size());
        group_param_names.reserve(groups.size());

        for (const AxisGroup& group : groups) {
            if (group.axes.empty())
                throw SpecError("SweepSpec::expand: an axis group is empty");
            const std::size_t length = group.axes.front().values.size();
            if (length == 0)
                throw SpecError("SweepSpec::expand: axis '" +
                                group.axes.front().parameter + "' is empty");
            std::vector<std::string> names;
            names.reserve(group.axes.size());
            for (const Axis& axis : group.axes) {
                if (axis.values.size() != length)
                    throw SpecError(
                        "SweepSpec::expand: axes in a group must be equal length");
                names.push_back(axis.parameter);
            }
            std::vector<std::vector<double>> rows;
            rows.reserve(length);
            for (std::size_t row = 0; row < length; ++row) {
                std::vector<double> values;
                values.reserve(group.axes.size());
                for (const Axis& axis : group.axes) values.push_back(axis.values[row]);
                rows.push_back(std::move(values));
            }
            group_rows.push_back(std::move(rows));
            group_param_names.push_back(std::move(names));
        }

        // Build integer index-axes (one entry per row in each group) and product
        // them with the shared grid algorithm; group 0 varies slowest.
        std::vector<std::vector<double>> index_axes;
        index_axes.reserve(group_rows.size());
        for (const auto& rows : group_rows) {
            std::vector<double> indices(rows.size());
            for (std::size_t i = 0; i < rows.size(); ++i)
                indices[i] = static_cast<double>(i);
            index_axes.push_back(std::move(indices));
        }
        const std::vector<std::vector<double>> index_combos =
            goss::sim::make_grid(index_axes);

        std::vector<RunSpec> runs;
        runs.reserve(index_combos.size());
        for (const std::vector<double>& index_combo : index_combos) {
            RunSpec run = base;
            for (std::size_t g = 0; g < groups.size(); ++g) {
                const std::size_t row_index =
                    static_cast<std::size_t>(index_combo[g]);
                const std::vector<double>& row = group_rows[g][row_index];
                for (std::size_t a = 0; a < group_param_names[g].size(); ++a)
                    run.parameters[group_param_names[g][a]] = row[a];
            }
            runs.push_back(std::move(run));
        }
        return runs;
    }

    // ---- Legacy path (DEPRECATED, backward-compat only): flat axes + combinator.
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
