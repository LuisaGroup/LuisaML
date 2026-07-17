#pragma once
#include "luisa_ml_config.h"
#include <luisa/dsl/sugar.h>
#include "onnx/type_dispatch.h"
#include "onnx/dynamic_array/dynamic_array.h"
#include "onnx/onnx.h"
#include "onnx/fp_quantized.h"

#include <luisa/core/stl/vector.h>

namespace lcml::onnx {
using namespace luisa::compute;
using refl::Value;
using refl::visit_type_index;


template<typename T>
struct RemoveVar {
    using type = T;
};
template<typename T>
struct RemoveVar<Var<T>> {
    using type = T;
};
template<typename T>
using RemoveVarT = typename RemoveVar<T>::type;

// Type trait: demote DSL-unsupported 64-bit types to 32-bit equivalents.
// double -> float, slong (int64) -> int, ulong (uint64) -> uint
template<typename T>
struct nn_storage_type {
    using type = T;
};
template<>
struct nn_storage_type<double> {
    using type = float;
};
template<>
struct nn_storage_type<slong> {
    using type = int32_t;
};
template<>
struct nn_storage_type<ulong> {
    using type = uint;
};

template<typename T>
using nn_storage_type_t = typename nn_storage_type<T>::type;

// NNTensor<T> uses the demoted storage type internally.
// e.g. NNTensor<double> is actually Tensor<float, DynamicArray<float>>.
template<typename T>
using NNTensor = Tensor<nn_storage_type_t<T>, DynamicArray<nn_storage_type_t<T>>>;
template<typename T>
using NNConstTensor = ConstTensor<nn_storage_type_t<T>, DynamicArray<nn_storage_type_t<T>>>;

// List of all supported onnx DataType <-> C++ type mappings
// Each entry is std::pair<Value<onnx::DataType>, CppType>
using NNTypeMapList = std::tuple<
    std::pair<Value<onnx::DataType::FLOAT>, float>,
    std::pair<Value<onnx::DataType::FLOAT16>, half>,
    std::pair<Value<onnx::DataType::DOUBLE>, double>,
    std::pair<Value<onnx::DataType::BOOL>, bool>,
    std::pair<Value<onnx::DataType::INT16>, int16_t>,
    std::pair<Value<onnx::DataType::UINT16>, ushort>,
    std::pair<Value<onnx::DataType::INT32>, int32_t>,
    std::pair<Value<onnx::DataType::UINT32>, uint>,
    std::pair<Value<onnx::DataType::INT64>, slong>,
    std::pair<Value<onnx::DataType::UINT64>, ulong>,
    std::pair<Value<onnx::DataType::FLOAT4E2M1>, FP4E2M1>,
    std::pair<Value<onnx::DataType::FLOAT8E4M3FN>, FP8E4M3FN>,
    std::pair<Value<onnx::DataType::FLOAT8E5M2>, FP8E5M2>,
    std::pair<Value<onnx::DataType::FLOAT16QUANTIZED>, FP16Quantized>>;

// Derive NNTypeList (std::tuple<float, half, ...>) from NNTypeMapList
namespace detail {
template<typename Tuple>
struct ExtractTypes;
template<typename... Maps>
struct ExtractTypes<std::tuple<Maps...>> {
    using type = std::tuple<typename Maps::second_type...>;
};
}// namespace detail
using NNTypeList = typename detail::ExtractTypes<NNTypeMapList>::type;

// Filter a type tuple by a template class Cond (requires Cond<T>::value to be bool)
namespace detail {
template<template<typename> class Cond, typename Result, typename Remaining>
struct FilterTypes;

template<template<typename> class Cond, typename... Rs>
struct FilterTypes<Cond, std::tuple<Rs...>, std::tuple<>> {
    using type = std::tuple<Rs...>;
};

template<template<typename> class Cond, typename... Rs, typename Head, typename... Tail>
struct FilterTypes<Cond, std::tuple<Rs...>, std::tuple<Head, Tail...>> {
    using type = std::conditional_t<
        Cond<Head>::value,
        typename FilterTypes<Cond, std::tuple<Rs..., Head>, std::tuple<Tail...>>::type,
        typename FilterTypes<Cond, std::tuple<Rs...>, std::tuple<Tail...>>::type>;
};
}// namespace detail

// NNFilteredTypeList<Cond>: subset of NNTypeList where Cond<T>::value is true
template<template<typename> class Cond>
using NNFilteredTypeList = typename detail::FilterTypes<Cond, std::tuple<>, NNTypeList>::type;

template<typename T>
struct IsFloatingPoint : std::bool_constant<luisa::is_floating_point_v<T>> {};
template<typename T>
struct IsInteger : std::bool_constant<luisa::is_integral_v<T>> {};
template<typename T>
struct IsBool : std::bool_constant<std::is_same_v<T, bool>> {};
template<typename T>
struct IsNumeric : std::bool_constant<!std::is_same_v<T, bool>> {};
template<typename T>
struct IsSigned : std::bool_constant<luisa::is_floating_point_v<T> || luisa::is_signed_v<T>> {};
// Native arithmetic: excludes bool and FP quantized types (which don't support DSL arithmetic)
template<typename T>
struct IsNativeArithmetic : std::bool_constant<
    !std::is_same_v<T, bool> &&
    !std::is_same_v<T, FP4E2M1> &&
    !std::is_same_v<T, FP8E4M3FN> &&
    !std::is_same_v<T, FP8E5M2> &&
    !std::is_same_v<T, FP16Quantized>> {};

// constexpr: C++ type -> onnx::DataType
namespace detail {
template<typename T, typename Tuple, size_t I = 0>
constexpr onnx::DataType to_onnx_dtype_impl() {
    static_assert(I < std::tuple_size_v<Tuple>, "Type not found in NNTypeMapList");
    if constexpr (std::is_same_v<T, typename std::tuple_element_t<I, Tuple>::second_type>) {
        return std::tuple_element_t<I, Tuple>::first_type::value;
    } else {
        return to_onnx_dtype_impl<T, Tuple, I + 1>();
    }
}
}// namespace detail

template<typename T>
constexpr onnx::DataType to_onnx_dtype = detail::to_onnx_dtype_impl<T, NNTypeMapList>();

// Runtime: onnx::DataType -> type index
inline luisa::TypeIndex onnx_dtype_to_type_index(onnx::DataType dt) {
    luisa::TypeIndex result = 0;
    auto found = [&]<size_t... Is>(std::index_sequence<Is...>) {
        return ((std::tuple_element_t<Is, NNTypeMapList>::first_type::value == dt ? (result = refl::type_index_of<typename std::tuple_element_t<Is, NNTypeMapList>::second_type>(), true) : false) || ...);
    }(std::make_index_sequence<std::tuple_size_v<NNTypeMapList>>{});
    LUISA_ASSERT(found, "Unsupported ONNX DataType: {}", magic_enum::enum_name(dt));
    return result;
}

// Runtime: onnx::DataType -> type name
inline std::string_view onnx_dtype_to_type_name(onnx::DataType dt) {
    std::string_view result;
    auto found = [&]<size_t... Is>(std::index_sequence<Is...>) {
        return ((std::tuple_element_t<Is, NNTypeMapList>::first_type::value == dt ? (result = refl::type_name_v<typename std::tuple_element_t<Is, NNTypeMapList>::second_type>, true) : false) || ...);
    }(std::make_index_sequence<std::tuple_size_v<NNTypeMapList>>{});
    LUISA_ASSERT(found, "Unsupported ONNX DataType: {}", magic_enum::enum_name(dt));
    return result;
}

// Runtime: onnx::DataType -> visit with typed lambda (like visit_type_index but keyed by enum)
template<typename F>
auto visit_onnx_dtype(onnx::DataType dt, F &&f) -> decltype(auto) {
    return visit_type_index<NNTypeList>(onnx_dtype_to_type_index(dt), std::forward<F>(f));
}

// Forward declaration for vectorized component-wise application
namespace detail {
template<typename VecT, typename F>
inline auto vectorized_unary_scalar(VecT v, F &&f);
}

// --- Erf: Abramowitz & Stegun approximation (max error ~1.5e-7) ---
template<typename T>
inline T erf(T x) {
    using RawT = RemoveVarT<T>;
    if constexpr (std::is_same_v<RawT, float4> || std::is_same_v<RawT, half4>) {
        return detail::vectorized_unary_scalar(x, [](auto c) {
            auto ax = abs(c);
            auto t = decltype(c){1.0f} / (decltype(c){1.0f} + decltype(c){0.3275911f} * ax);
            auto poly = t * (decltype(c){0.254829592f} + t * (decltype(c){-0.284496736f} + t * (decltype(c){1.421413741f} + t * (decltype(c){-1.453152027f} + t * decltype(c){1.061405429f}))));
            auto result = decltype(c){1.0f} - poly * exp(-ax * ax);
            return select(result, -result, c < decltype(c){0.0f});
        });
    } else {
        auto ax = abs(x);
        auto t = T{1} / (T{1} + T{0.3275911f} * ax);
        auto poly = t * (T{0.254829592f} + t * (T{-0.284496736f} + t * (T{1.421413741f} + t * (T{-1.453152027f} + t * T{1.061405429f}))));
        auto result = T{1} - poly * exp(-ax * ax);
        return select(result, -result, x < T{0});
    }
}

// ==========================================================================
// Shape helpers
// ==========================================================================

// Check if a tensor shape is effectively 1-D, i.e. at most one dimension > 1.
// Examples: (5,), (1,3,1), (1,1,1), (7,1) -> true; (2,3) -> false.
inline bool is_effectively_1d(ITensor::shape_type const &shape, uint32_t ndim) {
    uint32_t non_trivial = 0;
    for (uint32_t i = 0; i < ndim; ++i) {
        if (shape[i] > 1) ++non_trivial;
    }
    return non_trivial <= 1;
}

// ==========================================================================
// Index decomposition helpers
// ==========================================================================

// Extract the coordinate of a single dimension from a linear index.
// coord_d = (linear_idx / strides[d]) % shape[d]
// Optimized: stride==1 returns linear_idx % shape[d]; shape[d]==1 returns 0.
template<typename LinearIdx>
inline auto extract_coord(LinearIdx linear_idx,
                          ITensor::shape_type const &strides,
                          ITensor::shape_type const &shape,
                          uint32_t d) {
    if (shape[d] == 1) {
        return def(0u);
    }
    if (strides[d] == 1) {
        return def(cast<uint>(linear_idx)) % shape[d];
    }
    return (def(cast<uint>(linear_idx)) / strides[d]) % shape[d];
}

// Decompose a linear index into per-dimension coordinates using strides
// and invoke callback(d, coord) for each dimension d.
// Optimized: stride==1 (last dim) uses remaining directly; skips redundant ops.
// Also tracks the upper bound of `remaining` to eliminate redundant div/mod
// when the index range guarantees the quotient is always 0.
// Pass total_elements (the loop bound) to enable this upper-bound optimization.
template<typename LinearIdx, typename Callback>
inline void for_each_dim(LinearIdx linear_idx,
                         ITensor::shape_type const &strides,
                         uint32_t ndim,
                         Callback &&callback,
                         uint32_t total_elements = 0) {
    auto remaining = def(cast<uint>(linear_idx));
    // remaining_max tracks the maximum possible value of `remaining`.
    // If total_elements is provided, remaining ∈ [0, total_elements).
    // After remaining %= s, remaining ∈ [0, s).
    uint32_t remaining_max = (total_elements > 0) ? (total_elements - 1) : UINT32_MAX;
    for (uint32_t d = 0; d < ndim; ++d) {
        auto s = strides[d];
        if (s == 1) {
            // stride==1: coord == remaining, no division needed
            callback(d, remaining);
            // After consuming remaining, set to 0 for any subsequent dims
            if (d + 1 < ndim) {
                remaining = def(0u);
                remaining_max = 0;
            }
        } else if (s == 0) {
            // Degenerate stride — coord is always 0
            callback(d, def(0u));
        } else if (remaining_max < s) {
            // remaining < s guaranteed: coord is always 0, remaining unchanged.
            // Eliminates redundant div/mod for size-1 dimensions.
            callback(d, def(0u));
        } else {
            auto coord = remaining / s;
            remaining = remaining % s;
            callback(d, coord);
            remaining_max = s - 1;
        }
    }
}

// Shape-aware overload: uses shape to eliminate redundant div/mod operations.
// When shape[d]==1, the coordinate is trivially 0 — no division/modulo needed.
// This avoids generating code like `x / 24; x % 24` when the index range
// guarantees the result is always (0, x).
template<typename LinearIdx, typename Callback>
inline void for_each_dim(LinearIdx linear_idx,
                         ITensor::shape_type const &strides,
                         ITensor::shape_type const &shape,
                         uint32_t ndim,
                         Callback &&callback) {
    auto remaining = def(cast<uint>(linear_idx));
    for (uint32_t d = 0; d < ndim; ++d) {
        auto s = strides[d];
        if (shape[d] == 1) {
            // Dimension has size 1: coord is always 0, remaining unchanged.
            callback(d, def(0u));
        } else if (s == 1) {
            callback(d, remaining);
            if (d + 1 < ndim) {
                remaining = def(0u);
            }
        } else if (s == 0) {
            callback(d, def(0u));
        } else {
            auto coord = remaining / s;
            remaining = remaining % s;
            callback(d, coord);
        }
    }
}

// Remap a linear index from one stride layout to another.
// Decomposes linear_idx by src_strides and recomposes with dst_strides.
// If strides are identical, returns linear_idx directly (no decomposition).
template<typename LinearIdx>
inline auto linear_remap(LinearIdx linear_idx,
                         ITensor::shape_type const &src_strides,
                         ITensor::shape_type const &dst_strides,
                         uint32_t ndim) {
    if (src_strides == dst_strides) {
        return def(cast<uint>(linear_idx));
    }
    auto result = def(0u);
    for_each_dim(linear_idx, src_strides, ndim,
                 [&](uint32_t d, auto coord) {
                     auto ds = dst_strides[d];
                     if (ds == 0) {
                         // skip: broadcast dim contributes nothing
                     } else if (ds == 1) {
                         result += coord;
                     } else {
                         result += coord * ds;
                     }
                 });
    return result;
}

// Broadcast remap: decompose linear_idx by out_strides, then for each
// input stride array, accumulate coord * in_stride[d] (stride=0 ⇒ broadcast).
// Returns std::tuple<Var<uint>, ...> with one remapped index per input.
// If all input strides equal out_strides, returns linear_idx directly for each.
template<typename LinearIdx, typename... StridesArrays>
inline auto broadcast_linear_remap(LinearIdx linear_idx,
                                   ITensor::shape_type const &out_strides,
                                   uint32_t ndim,
                                   StridesArrays const &...in_strides) {
    // Fast path: if every input stride array equals out_strides, skip decomposition
    if ((... && (in_strides == out_strides))) {
        auto as_linear = [&](auto const &) { return def(cast<uint>(linear_idx)); };
        return std::tuple<decltype(as_linear(in_strides))...>{as_linear(in_strides)...};
    }
    // Slow path: decompose by output strides, accumulate per-input index
    // Optimized: skip stride==0 dims (broadcast), use coord directly for stride==1
    auto make_zero = [](auto const &) { return def(0u); };
    std::tuple<decltype(make_zero(in_strides))...> results{make_zero(in_strides)...};
    for_each_dim(linear_idx, out_strides, ndim,
                 [&](uint32_t d, auto coord) {
                     auto accumulate_one = [&]<size_t I>(std::integral_constant<size_t, I>,
                                                         auto const &strides_arr) {
                         auto s = strides_arr[d];
                         if (s == 0) {
                             // Broadcast dim: contributes nothing
                         } else if (s == 1) {
                             // stride==1: just add coord, no multiply
                             std::get<I>(results) += coord;
                         } else {
                             std::get<I>(results) += coord * s;
                         }
                     };
                     [&]<size_t... Is>(std::index_sequence<Is...>) {
                         (accumulate_one(std::integral_constant<size_t, Is>{}, in_strides), ...);
                     }(std::index_sequence_for<StridesArrays...>{});
                 });
    return results;
}

// ============================================================================
// Vectorization helpers
// ============================================================================

namespace detail {

template<typename T>
struct VecDispatch {
    static constexpr bool supported = false;
};

template<>
struct VecDispatch<float> {
    static constexpr bool supported = true;
    using VecT = float4;
    static constexpr uint32_t N = 4u;
    static auto broadcast(Var<float> v) { return make_float4(v); }
};

template<>
struct VecDispatch<half> {
    static constexpr bool supported = true;
    using VecT = half4;
    static constexpr uint32_t N = 4u;
    static auto broadcast(Var<half> v) { return make_half4(v); }
};

template<typename A, typename B, typename O>
[[nodiscard]] bool all_byte_buffer(A &a, B &b, O &out) noexcept {
    return a.container().is_byte_buffer() &&
           b.container().is_byte_buffer() &&
           out.container().is_byte_buffer();
}

template<typename A, typename O>
[[nodiscard]] bool all_byte_buffer(A &a, O &out) noexcept {
    return a.container().is_byte_buffer() && out.container().is_byte_buffer();
}

template<typename ST, typename A, typename B, typename O, typename F>
void vectorized_same_shape(A &a, B &b, O &out, F &&f) {
    using VecT = typename VecDispatch<ST>::VecT;
    auto buf_a = a.container().get_byte_buffer();
    auto buf_b = b.container().get_byte_buffer();
    auto buf_o = out.container().get_byte_buffer();
    auto off_a = static_cast<uint>(a.container().get_byte_offset());
    auto off_b = static_cast<uint>(b.container().get_byte_offset());
    auto off_o = static_cast<uint>(out.container().get_byte_offset());
    auto n = out.size();
    auto vec_n = static_cast<uint>(n / VecDispatch<ST>::N);
    auto rem = static_cast<uint>(n % VecDispatch<ST>::N);
    for (auto i : dynamic_range(vec_n)) {
        auto byte_idx = i * static_cast<uint>(sizeof(VecT));
        auto va = buf_a->read<VecT>(off_a + byte_idx);
        auto vb = buf_b->read<VecT>(off_b + byte_idx);
        auto vr = f(va, vb);
        buf_o->write(off_o + byte_idx, vr);
    }
    for (auto i : dynamic_range(rem)) {
        auto idx = vec_n * VecDispatch<ST>::N + i;
        out[idx] = f(a[idx], b[idx]);
    }
}

template<typename ST, typename A, typename O, typename F>
void vectorized_scalar_b(A &a, Var<ST> scalar_b, O &out, F &&f) {
    using VecT = typename VecDispatch<ST>::VecT;
    auto buf_a = a.container().get_byte_buffer();
    auto buf_o = out.container().get_byte_buffer();
    auto off_a = static_cast<uint>(a.container().get_byte_offset());
    auto off_o = static_cast<uint>(out.container().get_byte_offset());
    auto vec_b = VecDispatch<ST>::broadcast(scalar_b);
    auto n = out.size();
    auto vec_n = static_cast<uint>(n / VecDispatch<ST>::N);
    auto rem = static_cast<uint>(n % VecDispatch<ST>::N);
    for (auto i : dynamic_range(vec_n)) {
        auto byte_idx = i * static_cast<uint>(sizeof(VecT));
        auto va = buf_a->read<VecT>(off_a + byte_idx);
        auto vr = f(va, vec_b);
        buf_o->write(off_o + byte_idx, vr);
    }
    for (auto i : dynamic_range(rem)) {
        auto idx = vec_n * VecDispatch<ST>::N + i;
        out[idx] = f(a[idx], scalar_b);
    }
}

template<typename ST, typename B, typename O, typename F>
void vectorized_scalar_a(Var<ST> scalar_a, B &b, O &out, F &&f) {
    using VecT = typename VecDispatch<ST>::VecT;
    auto buf_b = b.container().get_byte_buffer();
    auto buf_o = out.container().get_byte_buffer();
    auto off_b = static_cast<uint>(b.container().get_byte_offset());
    auto off_o = static_cast<uint>(out.container().get_byte_offset());
    auto vec_a = VecDispatch<ST>::broadcast(scalar_a);
    auto n = out.size();
    auto vec_n = static_cast<uint>(n / VecDispatch<ST>::N);
    auto rem = static_cast<uint>(n % VecDispatch<ST>::N);
    for (auto i : dynamic_range(vec_n)) {
        auto byte_idx = i * static_cast<uint>(sizeof(VecT));
        auto vb = buf_b->read<VecT>(off_b + byte_idx);
        auto vr = f(vec_a, vb);
        buf_o->write(off_o + byte_idx, vr);
    }
    for (auto i : dynamic_range(rem)) {
        auto idx = vec_n * VecDispatch<ST>::N + i;
        out[idx] = f(scalar_a, b[idx]);
    }
}

template<typename ST, typename I, typename O, typename F>
void vectorized_unary(I &in, O &out, F &&f) {
    using VecT = typename VecDispatch<ST>::VecT;
    auto buf_in = in.container().get_byte_buffer();
    auto buf_o = out.container().get_byte_buffer();
    auto off_in = static_cast<uint>(in.container().get_byte_offset());
    auto off_o = static_cast<uint>(out.container().get_byte_offset());
    auto n = out.size();
    auto vec_n = static_cast<uint>(n / VecDispatch<ST>::N);
    auto rem = static_cast<uint>(n % VecDispatch<ST>::N);
    for (auto i : dynamic_range(vec_n)) {
        auto byte_idx = i * static_cast<uint>(sizeof(VecT));
        auto v = buf_in->read<VecT>(off_in + byte_idx);
        auto vr = f(v);
        buf_o->write(off_o + byte_idx, vr);
    }
    for (auto i : dynamic_range(rem)) {
        auto idx = vec_n * VecDispatch<ST>::N + i;
        auto byte_idx_in = idx * static_cast<uint>(sizeof(ST));
        auto byte_idx_o = idx * static_cast<uint>(sizeof(ST));
        auto v = buf_in->read<ST>(off_in + byte_idx_in);
        buf_o->write(off_o + byte_idx_o, f(v));
    }
}

template<typename ST, typename O, typename F>
void vectorized_in_place(O &out, F &&f) {
    using VecT = typename VecDispatch<ST>::VecT;
    auto buf_o = out.container().get_byte_buffer();
    auto off_o = static_cast<uint>(out.container().get_byte_offset());
    auto n = out.size();
    auto vec_n = static_cast<uint>(n / VecDispatch<ST>::N);
    auto rem = static_cast<uint>(n % VecDispatch<ST>::N);
    for (auto i : dynamic_range(vec_n)) {
        auto byte_idx = i * static_cast<uint>(sizeof(VecT));
        auto v = buf_o->read<VecT>(off_o + byte_idx);
        auto vr = f(v);
        buf_o->write(off_o + byte_idx, vr);
    }
    for (auto i : dynamic_range(rem)) {
        auto idx = vec_n * VecDispatch<ST>::N + i;
        out[idx] = f(out[idx]);
    }
}

// Vectorized inner-dimension broadcast for 1 input -> 1 output.
// Requires output innermost stride==1 and both tensors are ByteBuffer.
// Input innermost stride must be 0 (broadcast) or 1 (contiguous).
template<typename ST, typename I, typename O, typename F>
void vectorized_broadcast_unary(I &in, O &out,
                                ITensor::shape_type const &out_shape,
                                uint32_t out_ndim,
                                luisa::vector<uint32_t> const &in_stride,
                                F &&f) {
    using VecT = typename VecDispatch<ST>::VecT;
    auto buf_in = in.container().get_byte_buffer();
    auto buf_o = out.container().get_byte_buffer();
    auto off_in = static_cast<uint>(in.container().get_byte_offset());
    auto off_o = static_cast<uint>(out.container().get_byte_offset());
    auto last_dim = static_cast<uint>(out_shape[out_ndim - 1]);
    if (last_dim == 0 || out.size() == 0) return;
    auto in_stride_last = in_stride[out_ndim - 1];
    auto vec_n = last_dim / VecDispatch<ST>::N;
    auto rem = last_dim % VecDispatch<ST>::N;
    auto row_count = static_cast<uint>(out.size() / last_dim);
    for (auto row : dynamic_range(row_count)) {
        auto row_base = row * last_dim;
        for (auto v : dynamic_range(vec_n)) {
            auto base = row_base + v * VecDispatch<ST>::N;
            auto [idx_in_base] = broadcast_linear_remap(
                base, out.strides(), out_ndim, in_stride);
            Var<VecT> vin;
            if (in_stride_last == 1) {
                vin = buf_in->read<VecT>(off_in + idx_in_base * static_cast<uint>(sizeof(ST)));
            } else {
                auto s = buf_in->read<ST>(off_in + idx_in_base * static_cast<uint>(sizeof(ST)));
                vin = VecDispatch<ST>::broadcast(s);
            }
            auto vr = f(vin);
            buf_o->write(off_o + base * static_cast<uint>(sizeof(ST)), vr);
        }
        for (auto r : dynamic_range(rem)) {
            auto idx = row_base + vec_n * VecDispatch<ST>::N + r;
            auto [idx_in] = broadcast_linear_remap(
                idx, out.strides(), out_ndim, in_stride);
            auto s = buf_in->read<ST>(off_in + idx_in * static_cast<uint>(sizeof(ST)));
            buf_o->write(off_o + idx * static_cast<uint>(sizeof(ST)), f(s));
        }
    }
}

// Vectorized inner-dimension broadcast for 2 inputs + 1 output.
// Requires output innermost stride==1 and all tensors are ByteBuffer.
// Input innermost strides must be 0 (broadcast) or 1 (contiguous).
template<typename ST, typename A, typename B, typename O, typename F>
void vectorized_broadcast_2in(A &a, B &b, O &out,
                              ITensor::shape_type const &out_shape,
                              uint32_t out_ndim,
                              luisa::vector<uint32_t> const &stride_a,
                              luisa::vector<uint32_t> const &stride_b,
                              F &&f) {
    using VecT = typename VecDispatch<ST>::VecT;
    auto buf_a = a.container().get_byte_buffer();
    auto buf_b = b.container().get_byte_buffer();
    auto buf_o = out.container().get_byte_buffer();
    auto off_a = static_cast<uint>(a.container().get_byte_offset());
    auto off_b = static_cast<uint>(b.container().get_byte_offset());
    auto off_o = static_cast<uint>(out.container().get_byte_offset());
    auto last_dim = static_cast<uint>(out_shape[out_ndim - 1]);
    auto stride_a_last = stride_a[out_ndim - 1];
    auto stride_b_last = stride_b[out_ndim - 1];
    auto vec_n = last_dim / VecDispatch<ST>::N;
    auto rem = last_dim % VecDispatch<ST>::N;
    auto row_count = static_cast<uint>(out.size() / last_dim);
    for (auto row : dynamic_range(row_count)) {
        auto row_base = row * last_dim;
        for (auto v : dynamic_range(vec_n)) {
            auto base = row_base + v * VecDispatch<ST>::N;
            auto [idx_a_base, idx_b_base] = broadcast_linear_remap(
                base, out.strides(), out_ndim, stride_a, stride_b);
            Var<VecT> va, vb;
            if (stride_a_last == 1) {
                va = buf_a->read<VecT>(off_a + idx_a_base * static_cast<uint>(sizeof(ST)));
            } else {
                auto sa = buf_a->read<ST>(off_a + idx_a_base * static_cast<uint>(sizeof(ST)));
                va = VecDispatch<ST>::broadcast(sa);
            }
            if (stride_b_last == 1) {
                vb = buf_b->read<VecT>(off_b + idx_b_base * static_cast<uint>(sizeof(ST)));
            } else {
                auto sb = buf_b->read<ST>(off_b + idx_b_base * static_cast<uint>(sizeof(ST)));
                vb = VecDispatch<ST>::broadcast(sb);
            }
            auto vr = f(va, vb);
            buf_o->write(off_o + base * static_cast<uint>(sizeof(ST)), vr);
        }
        for (auto r : dynamic_range(rem)) {
            auto idx = row_base + vec_n * VecDispatch<ST>::N + r;
            auto [idx_a, idx_b] = broadcast_linear_remap(
                idx, out.strides(), out_ndim, stride_a, stride_b);
            auto va = buf_a->read<ST>(off_a + idx_a * static_cast<uint>(sizeof(ST)));
            auto vb = buf_b->read<ST>(off_b + idx_b * static_cast<uint>(sizeof(ST)));
            buf_o->write(off_o + idx * static_cast<uint>(sizeof(ST)), f(va, vb));
        }
    }
}

// Vectorized inner-dimension broadcast for 1 input + accumulator (read-modify-write).
// Requires accumulator innermost stride==1 and both tensors are ByteBuffer.
// Input innermost stride must be 0 (broadcast) or 1 (contiguous).
template<typename ST, typename ACC, typename B, typename F>
void vectorized_broadcast_1in_acc(ACC &acc, B &b,
                                  ITensor::shape_type const &out_shape,
                                  uint32_t out_ndim,
                                  luisa::vector<uint32_t> const &stride_b,
                                  F &&f) {
    using VecT = typename VecDispatch<ST>::VecT;
    auto buf_acc = acc.container().get_byte_buffer();
    auto buf_b = b.container().get_byte_buffer();
    auto off_acc = static_cast<uint>(acc.container().get_byte_offset());
    auto off_b = static_cast<uint>(b.container().get_byte_offset());
    auto last_dim = static_cast<uint>(out_shape[out_ndim - 1]);
    auto stride_b_last = stride_b[out_ndim - 1];
    auto vec_n = last_dim / VecDispatch<ST>::N;
    auto rem = last_dim % VecDispatch<ST>::N;
    auto row_count = static_cast<uint>(acc.size() / last_dim);
    for (auto row : dynamic_range(row_count)) {
        auto row_base = row * last_dim;
        for (auto v : dynamic_range(vec_n)) {
            auto base = row_base + v * VecDispatch<ST>::N;
            auto [idx_b_base] = broadcast_linear_remap(
                base, acc.strides(), out_ndim, stride_b);
            auto vacc = buf_acc->read<VecT>(off_acc + base * static_cast<uint>(sizeof(ST)));
            Var<VecT> vb;
            if (stride_b_last == 1) {
                vb = buf_b->read<VecT>(off_b + idx_b_base * static_cast<uint>(sizeof(ST)));
            } else {
                auto sb = buf_b->read<ST>(off_b + idx_b_base * static_cast<uint>(sizeof(ST)));
                vb = VecDispatch<ST>::broadcast(sb);
            }
            auto vr = f(vacc, vb);
            buf_acc->write(off_acc + base * static_cast<uint>(sizeof(ST)), vr);
        }
        for (auto r : dynamic_range(rem)) {
            auto idx = row_base + vec_n * VecDispatch<ST>::N + r;
            auto [idx_b] = broadcast_linear_remap(
                idx, acc.strides(), out_ndim, stride_b);
            auto vacc = buf_acc->read<ST>(off_acc + idx * static_cast<uint>(sizeof(ST)));
            auto vb = buf_b->read<ST>(off_b + idx_b * static_cast<uint>(sizeof(ST)));
            buf_acc->write(off_acc + idx * static_cast<uint>(sizeof(ST)), f(vacc, vb));
        }
    }
}

// Apply a scalar lambda to each component of a vector type and reconstruct the vector.
// This avoids requiring the lambda itself to support vector types (e.g. operators with T{0} literals).
template<typename VecT, typename F>
inline auto vectorized_unary_scalar(VecT v, F &&f) {
    using RawT = RemoveVarT<VecT>;
    if constexpr (std::is_same_v<RawT, float4>) {
        return make_float4(f(v.x), f(v.y), f(v.z), f(v.w));
    } else if constexpr (std::is_same_v<RawT, half4>) {
        return make_half4(f(v.x), f(v.y), f(v.z), f(v.w));
    } else {
        static_assert(std::is_same_v<RawT, float4> || std::is_same_v<RawT, half4>, "Unsupported vector type");
        return v;
    }
}

// Vectorized element-wise cast between two ByteBuffer-backed tensors.
// Requires both SrcT and DstT to have VecDispatch support (currently float and half).
template<typename SrcT, typename DstT, typename I, typename O, typename F>
void vectorized_cast(I &in, O &out, F &&f) {
    using VecSrcT = typename VecDispatch<SrcT>::VecT;
    using VecDstT = typename VecDispatch<DstT>::VecT;
    auto buf_in = in.container().get_byte_buffer();
    auto buf_o = out.container().get_byte_buffer();
    auto off_in = static_cast<uint>(in.container().get_byte_offset());
    auto off_o = static_cast<uint>(out.container().get_byte_offset());
    auto n = out.size();
    auto vec_n = static_cast<uint>(n / VecDispatch<SrcT>::N);
    auto rem = static_cast<uint>(n % VecDispatch<SrcT>::N);
    for (auto i : dynamic_range(vec_n)) {
        auto byte_idx_in = i * static_cast<uint>(sizeof(VecSrcT));
        auto byte_idx_o = i * static_cast<uint>(sizeof(VecDstT));
        auto v = buf_in->read<VecSrcT>(off_in + byte_idx_in);
        Var<VecDstT> vr;
        if constexpr (std::is_same_v<VecSrcT, float4> && std::is_same_v<VecDstT, float4>) {
            vr = make_float4(f(v.x), f(v.y), f(v.z), f(v.w));
        } else if constexpr (std::is_same_v<VecSrcT, float4> && std::is_same_v<VecDstT, half4>) {
            vr = make_half4(f(v.x), f(v.y), f(v.z), f(v.w));
        } else if constexpr (std::is_same_v<VecSrcT, half4> && std::is_same_v<VecDstT, float4>) {
            vr = make_float4(f(v.x), f(v.y), f(v.z), f(v.w));
        } else if constexpr (std::is_same_v<VecSrcT, half4> && std::is_same_v<VecDstT, half4>) {
            vr = make_half4(f(v.x), f(v.y), f(v.z), f(v.w));
        }
        buf_o->write(off_o + byte_idx_o, vr);
    }
    for (auto i : dynamic_range(rem)) {
        auto idx = vec_n * VecDispatch<SrcT>::N + i;
        auto byte_idx_in = idx * static_cast<uint>(sizeof(SrcT));
        auto byte_idx_o = idx * static_cast<uint>(sizeof(DstT));
        auto v = buf_in->read<SrcT>(off_in + byte_idx_in);
        auto vr = f(v);
        buf_o->write(off_o + byte_idx_o, vr);
    }
}

}// namespace detail

}// namespace lcml::onnx
