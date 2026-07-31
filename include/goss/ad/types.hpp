// include/goss/ad/types.hpp
#pragma once
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

namespace goss::ad {
using Scalar = double;
using SparsityPattern = std::vector<std::pair<std::size_t, std::size_t>>;
using SparseTriplets = std::vector<std::tuple<std::size_t, std::size_t, double>>;
}  // namespace goss::ad
