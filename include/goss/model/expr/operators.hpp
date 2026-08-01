// include/goss/model/expr/operators.hpp
// Free-function operator overloads (+, -, *) for the expression DSL.
// These operators build AST node structs at composition time — NO arithmetic
// happens here. Arithmetic only runs when eval<ScalarT>() is called, which
// is why CppAD can record every operation on its tape correctly.
#pragma once
#include <utility>
#include "goss/model/expr/nodes.hpp"

namespace goss::model::expr {

// Helper: wrap a double in ConstantExpr so double literals compose with nodes.
// This is a private implementation detail; it is not part of the public API.
namespace detail {
    inline ConstantExpr wrap_double(double value) { return ConstantExpr{value}; }
}  // namespace detail

// --- operator+ ---

template <typename LeftExpr, typename RightExpr>
BinaryExpr<AddTag, LeftExpr, RightExpr>
operator+(LeftExpr left_operand, RightExpr right_operand) {
    return BinaryExpr<AddTag, LeftExpr, RightExpr>{
        std::move(left_operand), std::move(right_operand)};
}

template <typename LeftExpr>
BinaryExpr<AddTag, LeftExpr, ConstantExpr>
operator+(LeftExpr left_operand, double right_value) {
    return BinaryExpr<AddTag, LeftExpr, ConstantExpr>{
        std::move(left_operand), detail::wrap_double(right_value)};
}

template <typename RightExpr>
BinaryExpr<AddTag, ConstantExpr, RightExpr>
operator+(double left_value, RightExpr right_operand) {
    return BinaryExpr<AddTag, ConstantExpr, RightExpr>{
        detail::wrap_double(left_value), std::move(right_operand)};
}

// --- operator- (binary) ---

template <typename LeftExpr, typename RightExpr>
BinaryExpr<SubTag, LeftExpr, RightExpr>
operator-(LeftExpr left_operand, RightExpr right_operand) {
    return BinaryExpr<SubTag, LeftExpr, RightExpr>{
        std::move(left_operand), std::move(right_operand)};
}

template <typename LeftExpr>
BinaryExpr<SubTag, LeftExpr, ConstantExpr>
operator-(LeftExpr left_operand, double right_value) {
    return BinaryExpr<SubTag, LeftExpr, ConstantExpr>{
        std::move(left_operand), detail::wrap_double(right_value)};
}

template <typename RightExpr>
BinaryExpr<SubTag, ConstantExpr, RightExpr>
operator-(double left_value, RightExpr right_operand) {
    return BinaryExpr<SubTag, ConstantExpr, RightExpr>{
        detail::wrap_double(left_value), std::move(right_operand)};
}

// --- operator* ---

template <typename LeftExpr, typename RightExpr>
BinaryExpr<MulTag, LeftExpr, RightExpr>
operator*(LeftExpr left_operand, RightExpr right_operand) {
    return BinaryExpr<MulTag, LeftExpr, RightExpr>{
        std::move(left_operand), std::move(right_operand)};
}

template <typename LeftExpr>
BinaryExpr<MulTag, LeftExpr, ConstantExpr>
operator*(LeftExpr left_operand, double right_value) {
    return BinaryExpr<MulTag, LeftExpr, ConstantExpr>{
        std::move(left_operand), detail::wrap_double(right_value)};
}

template <typename RightExpr>
BinaryExpr<MulTag, ConstantExpr, RightExpr>
operator*(double left_value, RightExpr right_operand) {
    return BinaryExpr<MulTag, ConstantExpr, RightExpr>{
        detail::wrap_double(left_value), std::move(right_operand)};
}

// --- unary operator- ---

template <typename OperandExpr>
UnaryNegExpr<OperandExpr>
operator-(OperandExpr operand) {
    return UnaryNegExpr<OperandExpr>{std::move(operand)};
}

}  // namespace goss::model::expr
