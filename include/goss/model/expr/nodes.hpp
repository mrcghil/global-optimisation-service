// include/goss/model/expr/nodes.hpp
#pragma once
#include <cstddef>
#include <type_traits>
#include <vector>

namespace goss::model::expr {

/// Leaf node: a compile-time constant. eval<ScalarT> wraps the stored double
/// in ScalarT(...) so CppAD records a constant on the AD tape — never use a
/// bare double literal inside eval when ScalarT may be an AD type.
struct ConstantExpr {
    double value;

    template <typename ScalarT>
    ScalarT eval(const std::vector<ScalarT>& /*x*/,
                 const std::vector<ScalarT>& /*u*/,
                 ScalarT /*t*/) const {
        return ScalarT(value);
    }
};

/// Leaf node: reads state at index i from the x vector.
struct StateLeaf {
    std::size_t state_index;

    template <typename ScalarT>
    ScalarT eval(const std::vector<ScalarT>& x,
                 const std::vector<ScalarT>& /*u*/,
                 ScalarT /*t*/) const {
        return x[state_index];
    }
};

/// Leaf node: reads control at index i from the u vector.
struct ControlLeaf {
    std::size_t control_index;

    template <typename ScalarT>
    ScalarT eval(const std::vector<ScalarT>& /*x*/,
                 const std::vector<ScalarT>& u,
                 ScalarT /*t*/) const {
        return u[control_index];
    }
};

/// Leaf node: returns the current time t.
struct TimeLeaf {
    template <typename ScalarT>
    ScalarT eval(const std::vector<ScalarT>& /*x*/,
                 const std::vector<ScalarT>& /*u*/,
                 ScalarT t) const {
        return t;
    }
};

/// Binary operator tags — structs used as template parameters to select
/// the arithmetic operation in BinaryExpr::eval without virtual dispatch.
struct AddTag {};
struct SubTag {};
struct MulTag {};

/// Binary expression node. Stores left and right subtrees by value (the tree
/// is stack-allocated; nodes are value types, not pointers). The OpTag
/// selects addition, subtraction, or multiplication via if constexpr so the
/// compiler inlines the correct operation under any scalar type T.
template <typename OpTag, typename LeftExpr, typename RightExpr>
struct BinaryExpr {
    LeftExpr left_operand;
    RightExpr right_operand;

    template <typename ScalarT>
    ScalarT eval(const std::vector<ScalarT>& x,
                 const std::vector<ScalarT>& u,
                 ScalarT t) const {
        const ScalarT left_value  = left_operand.eval(x, u, t);
        const ScalarT right_value = right_operand.eval(x, u, t);
        if constexpr (std::is_same_v<OpTag, AddTag>) {
            return left_value + right_value;
        } else if constexpr (std::is_same_v<OpTag, SubTag>) {
            return left_value - right_value;
        } else {
            // MulTag — only three tags exist in v1; a static_assert catches
            // accidental instantiation with an unknown tag at compile time.
            static_assert(std::is_same_v<OpTag, MulTag>,
                "BinaryExpr: unknown OpTag — only AddTag, SubTag, MulTag are supported in v1");
            return left_value * right_value;
        }
    }
};

/// Unary negation node. Returns ScalarT(-1) * operand so CppAD records the
/// negation as a multiplication, which is correctly differentiated.
template <typename OperandExpr>
struct UnaryNegExpr {
    OperandExpr operand;

    template <typename ScalarT>
    ScalarT eval(const std::vector<ScalarT>& x,
                 const std::vector<ScalarT>& u,
                 ScalarT t) const {
        return ScalarT(-1) * operand.eval(x, u, t);
    }
};

}  // namespace goss::model::expr
