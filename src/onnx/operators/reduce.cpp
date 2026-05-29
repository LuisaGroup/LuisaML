#include <limits>

#include "onnx/operator.h"
#include "onnx/network_instance.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

namespace lcml::onnx {

// Generic Reduce base class with axes and keepdims support
// ONNX spec (opset 18+): axes is an optional second input; keepdims attribute (default 1); noop_with_empty_axes (default 0)
template<typename Derived>
class ReduceBase : public Operator {
protected:
    std::vector<int> axes_;
    int keepdims_;
    int noop_with_empty_axes_;

public:
    uint warp_size_ = 0;

public:
    ReduceBase(std::string name, std::vector<int> axes, int keepdims, int noop_with_empty_axes)
        : Operator(std::move(name)), axes_(std::move(axes)),
          keepdims_(keepdims), noop_with_empty_axes_(noop_with_empty_axes) {}

    void set_environment(NetworkInstance &env, TensorTable &) override {
        warp_size_ = env.warp_size();
    }

    void set_warp_size(uint32_t size) override {
        warp_size_ = size;
    }

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
        LUISA_ASSERT(inputs.size() >= 1 && outputs.size() == 1, "Reduce op requires >=1 input and 1 output.");
        auto &X = inputs[0].get();
        auto &Y = outputs[0].get();

        LUISA_ASSERT(X.element_type() == Y.element_type(), "Reduce op: input and output must have the same element type.");

        auto const &x_shape = X.shape();
        auto ndim = x_shape.size();

        // Resolve axes
        std::vector<int> axes = axes_;

        // If axes is empty and noop_with_empty_axes, just copy
        if (axes.empty() && noop_with_empty_axes_) {
            visit_typeid<NNTypeList>(X.element_type(), [&]<typename T>() {
                auto &x = static_cast<NNTensor<T> &>(X);
                auto &y = static_cast<NNTensor<T> &>(Y);
                using VT = nn_storage_type_t<T>;
                if constexpr (detail::VecDispatch<VT>::supported) {
                    if (detail::all_byte_buffer(x, y)) {
                        detail::vectorized_unary<VT>(x, y, [](auto v) { return v; });
                        return;
                    }
                }
                for (auto i : dynamic_range(x.size())) {
                    y[i] = x[i];
                }
            });
            return;
        }

        // If axes empty, reduce all dims
        if (axes.empty()) {
            for (uint32_t d = 0; d < ndim; ++d)
                axes.push_back(static_cast<int>(d));
        }

        // Normalize negative axes
        for (auto &a : axes) {
            if (a < 0) a += static_cast<int>(ndim);
        }

        // Create a boolean mask for reduced dims
        std::vector<bool> reduce_dim(ndim, false);
        for (auto a : axes) reduce_dim[a] = true;

        // Compute reduced size
        uint32_t reduced_size = 1;
        for (uint32_t d = 0; d < ndim; ++d) {
            if (reduce_dim[d]) reduced_size *= x_shape[d];
        }

        // Build stride arrays for kept dims (output->input remap)
        // and reduced dims (reduced-space linear index -> input offset remap)
        ITensor::shape_type kept_out_strides, kept_in_strides;
        ITensor::shape_type reduced_strides, reduced_in_strides;
        std::vector<uint32_t> reduced_sizes;

        uint32_t out_d = 0;
        auto const &y_strides = Y.strides();
        auto const &x_strides = X.strides();

        for (uint32_t d = 0; d < ndim; ++d) {
            if (reduce_dim[d]) {
                reduced_sizes.push_back(x_shape[d]);
                reduced_in_strides.push_back(x_strides[d]);
                if (keepdims_) out_d++;
            } else {
                kept_out_strides.push_back(y_strides[out_d]);
                kept_in_strides.push_back(x_strides[d]);
                out_d++;
            }
        }

        // Compute strides for the reduced-space: last reduced dim has stride 1,
        // previous ones have stride = product of subsequent reduced dim sizes
        if (!reduced_sizes.empty()) {
            reduced_strides.resize(reduced_sizes.size());
            reduced_strides.back() = 1;
            for (int i = static_cast<int>(reduced_sizes.size()) - 2; i >= 0; --i) {
                reduced_strides[i] = reduced_strides[i + 1] * reduced_sizes[i + 1];
            }
        }

        auto kept_ndim = static_cast<uint32_t>(kept_out_strides.size());
        auto reduced_ndim = static_cast<uint32_t>(reduced_strides.size());

        static_cast<Derived *>(this)->reduce_impl(X, Y,
                                                  kept_out_strides, kept_in_strides, kept_ndim,
                                                  reduced_strides, reduced_in_strides, reduced_ndim, reduced_size);
    }
};

// Helper tags for warp reduce intrinsics
struct WarpReduceNone {
    // Dummy apply — never called, exists only so if-constexpr-discarded branches compile
    template<typename V>
    static auto apply(V &&v) { return std::forward<V>(v); }
};
struct WarpReduceSum {
    template<typename V>
    static auto apply(V &&v) { return warp_active_sum(std::forward<V>(v)); }
};
struct WarpReduceMax {
    template<typename V>
    static auto apply(V &&v) { return warp_active_max(std::forward<V>(v)); }
};
struct WarpReduceMin {
    template<typename V>
    static auto apply(V &&v) { return warp_active_min(std::forward<V>(v)); }
};
struct WarpReduceProduct {
    template<typename V>
    static auto apply(V &&v) { return warp_active_product(std::forward<V>(v)); }
};

// Type-safe identity values for reduce operators
namespace detail {
template<typename VT>
constexpr VT identity_max() noexcept {
    if constexpr (std::is_integral_v<VT>) {
        return std::numeric_limits<VT>::lowest();
    } else {
        return -std::numeric_limits<VT>::infinity();
    }
}
template<typename VT>
constexpr VT identity_min() noexcept {
    if constexpr (std::is_integral_v<VT>) {
        return std::numeric_limits<VT>::max();
    } else {
        return std::numeric_limits<VT>::infinity();
    }
}

// Append FP8/FP4 quantized types to a type tuple
template<typename Tuple>
struct AppendFPQuantized;
template<typename... Ts>
struct AppendFPQuantized<std::tuple<Ts...>> {
    using type = std::tuple<Ts..., FP8E4M3FN, FP8E5M2, FP4E2M1>;
};
template<typename Tuple>
using AppendFPQuantizedT = typename AppendFPQuantized<Tuple>::type;
}// namespace detail

// Macro for defining reduce operators with optional warp optimization and vectorized scalar path
// WarpTag: one of WarpReduceNone/Sum/Max/Min/Product
// WarpIdentity: identity value for inactive lanes in warp reduce (e.g. 0 for sum, identity_max<VT>() for max)
// WarpBlockAccumExpr: expression using 'acc' and 'blk_val' to merge warp block result into accumulator
// VecAccumExpr: expression using 'vec_acc' and 'v' for float4/half4 vectorized accumulation
// VecReduceExpr: expression reducing 'vec_acc' vector to scalar 'acc'
// QInitExpr/QAccumExpr/QFinalExpr: half-precision expressions for FP8/FP4 quantized path
#define DEFINE_REDUCE_OP(Name, TypeList, InitExpr, AccumExpr, FinalExpr,                                                                                          \
                         WarpTag, WarpIdentity, WarpBlockAccumExpr,                                                                                               \
                         VecAccumExpr, VecReduceExpr,                                                                                                             \
                         QInitExpr, QAccumExpr, QFinalExpr)                                                                                                       \
    class Name : public ReduceBase<Name> {                                                                                                    \
    public:                                                                                                    \
        Name(std::vector<int> axes, int keepdims, int noop)                                                                                                    \
            : ReduceBase(#Name, std::move(axes), keepdims, noop) {}                                                                                               \
                                                                                                    \
        void reduce_impl(ITensor &X, ITensor &Y,                                                                                                    \
                         ITensor::shape_type const &kept_out_strides,                                                                                             \
                         ITensor::shape_type const &kept_in_strides, uint32_t kept_ndim,                                                                          \
                         ITensor::shape_type const &reduced_strides,                                                                                              \
                         ITensor::shape_type const &reduced_in_strides, uint32_t reduced_ndim,                                                                    \
                         uint32_t reduced_size) {                                                                                                    \
            visit_typeid<TypeList>(X.element_type(), [&]<typename T>() {                                                                                          \
                using VT = nn_storage_type_t<T>;                                                                                                    \
                auto &x = static_cast<NNTensor<T> &>(X);                                                                                                    \
                auto &y = static_cast<NNTensor<T> &>(Y);                                                                                                    \
                                                                                                    \
                /* Quantized FP8/FP4 path: dequantize to half, accumulate, requantize */                                                                          \
                if constexpr (std::is_same_v<T, FP8E4M3FN> || std::is_same_v<T, FP8E5M2> || std::is_same_v<T, FP4E2M1>) {                                        \
                    auto deq = [&]() {                                                                                                    \
                        if constexpr (std::is_same_v<T, FP8E4M3FN>) return fp8e4m3_to_float();                                                                   \
                        else if constexpr (std::is_same_v<T, FP8E5M2>) return fp8e5m2_to_float();                                                                 \
                        else return fp4e2m1_to_float();                                                                                                    \
                    }();                                                                                                    \
                    auto q = [&]() {                                                                                                    \
                        if constexpr (std::is_same_v<T, FP8E4M3FN>) return fp8e4m3_from_float();                                                                   \
                        else if constexpr (std::is_same_v<T, FP8E5M2>) return fp8e5m2_from_float();                                                                 \
                        else return fp4e2m1_from_float();                                                                                                    \
                    }();                                                                                                    \
                    for (auto out_linear : dynamic_range(y.size())) {                                                                                         \
                        auto base = def(0u);                                                                                                    \
                        for_each_dim(out_linear, kept_out_strides, kept_ndim, [&](uint32_t d, auto coord) { base += coord * kept_in_strides[d]; }, y.size()); \
                        auto acc = def(QInitExpr);                                                                                                    \
                        for (auto r : dynamic_range(reduced_size)) {                                                                                                  \
                            auto offset = def(0u);                                                                                                    \
                            for_each_dim(r, reduced_strides, reduced_ndim, [&](uint32_t d, auto coord) { offset += coord * reduced_in_strides[d]; }, reduced_size);   \
                            auto bits = x[base + offset].bits.cast<ushort>();                                                                                                    \
                            auto val = deq(bits);                                                                                                    \
                            acc = (QAccumExpr).template as<half>();                                                                                                    \
                        }                                                                                                    \
                        auto result = (QFinalExpr).template as<half>();                                                                                                    \
                        y[out_linear].bits = q(result).cast<uint16_t>();                                                                                                    \
                    }                                                                                                    \
                    return;                                                                                                    \
                }                                                                                                    \
                                                                                                    \
                /* Native arithmetic path */                                                                                                    \
                if constexpr (IsNativeArithmetic<T>::value) {                                                                                                    \
                    /* Warp-optimized path: single contiguous reduce dimension */                                                                                     \
                    if constexpr (!std::is_same_v<WarpTag, WarpReduceNone>) {                                                                                         \
                        if (false && this->warp_size_ > 1 &&                                                                                                    \
                            reduced_ndim == 1 && reduced_in_strides[0] == 1) {                                                                                        \
                            auto lane_id = warp_lane_id();                                                                                                    \
                            uint num_blocks = reduced_size / this->warp_size_;                                                                                        \
                            uint tail = reduced_size % this->warp_size_;                                                                                              \
                            bool use_buf_vec = false;                                                                                                    \
                            Var<ByteBuffer> *buf_x = nullptr;                                                                                                    \
                            uint off_x = 0;                                                                                                    \
                            if constexpr (std::is_same_v<VT, float> || std::is_same_v<VT, half>) {                                                                   \
                                use_buf_vec = x.container().is_byte_buffer();                                                                                         \
                                if (use_buf_vec) {                                                                                                    \
                                    buf_x = x.container().get_byte_buffer();                                                                                            \
                                    off_x = static_cast<uint>(x.container().get_byte_offset());                                                                       \
                                }                                                                                                    \
                            }                                                                                                    \
                            auto type_size = static_cast<uint>(sizeof(VT));                                                                                           \
                            for (auto out_linear : dynamic_range(y.size())) {                                                                                         \
                                auto base = def(0u);                                                                                                    \
                                for_each_dim(out_linear, kept_out_strides, kept_ndim, [&](uint32_t d, auto coord) { base += coord * kept_in_strides[d]; }, y.size()); \
                                auto acc = def(VT(WarpIdentity));                                                                                                     \
                                if (use_buf_vec) {                                                                                                    \
                                    if constexpr (std::is_same_v<VT, float> || std::is_same_v<VT, half>) {                                                           \
                                        uint vec_blocks = num_blocks / 4u;                                                                                                \
                                        uint rem_blocks = num_blocks % 4u;                                                                                                \
                                        if constexpr (std::is_same_v<VT, float>) {                                                                                        \
                                            for (auto blk : dynamic_range(vec_blocks)) {                                                                                  \
                                                auto v = buf_x->read<float4>(off_x + (base + lane_id * 4u + blk * this->warp_size_ * 4u) * type_size);                    \
                                                auto vec_acc = def(make_float4(VT(WarpIdentity)));                                                                            \
                                                vec_acc = (VecAccumExpr).template as<float4>();                                                                           \
                                                auto blk_sum = (VecReduceExpr).template as<VT>();                                                                                      \
                                                auto blk_val = WarpTag::apply(blk_sum);                                                                                      \
                                                acc = (WarpBlockAccumExpr).template as<VT>();                                                                                      \
                                            }                                                                                                    \
                                        } else {                                                                                                    \
                                            for (auto blk : dynamic_range(vec_blocks)) {                                                                                  \
                                                auto v = buf_x->read<half4>(off_x + (base + lane_id * 4u + blk * this->warp_size_ * 4u) * type_size);                     \
                                                auto vec_acc = def(make_half4(VT(WarpIdentity)));                                                                             \
                                                vec_acc = (VecAccumExpr).template as<half4>();                                                                            \
                                                auto blk_sum = (VecReduceExpr).template as<VT>();                                                                                      \
                                                auto blk_val = WarpTag::apply(blk_sum);                                                                                      \
                                                acc = (WarpBlockAccumExpr).template as<VT>();                                                                                      \
                                            }                                                                                                    \
                                        }                                                                                                    \
                                        for (auto blk : dynamic_range(rem_blocks)) {                                                                                      \
                                            auto val = buf_x->read<VT>(off_x + (base + lane_id + vec_blocks * this->warp_size_ * 4u + blk * this->warp_size_) * type_size); \
                                            auto blk_val = WarpTag::apply(val);                                                                                          \
                                            acc = (WarpBlockAccumExpr).template as<VT>();                                                                                          \
                                        }                                                                                                    \
                                    }                                                                                                    \
                                } else {                                                                                                    \
                                    for (auto blk : dynamic_range(num_blocks)) {                                                                                      \
                                        auto val = x[base + lane_id + blk * this->warp_size_];                                                                        \
                                        auto blk_val = WarpTag::apply(val);                                                                                          \
                                        acc = (WarpBlockAccumExpr).template as<VT>();                                                                                          \
                                    }                                                                                                    \
                                }                                                                                                    \
                                /* Tail elements (< warp_size): inactive lanes use identity */                                                                        \
                                if (tail > 0) {                                                                                                    \
                                    auto tail_idx = lane_id + num_blocks * this->warp_size_;                                                                          \
                                    auto val = def(VT(WarpIdentity));                                                                                                 \
                                    $if (tail_idx < reduced_size) {                                                                                                   \
                                        if (use_buf_vec) {                                                                                                    \
                                            val = buf_x->read<VT>(off_x + (base + tail_idx) * type_size);                                                             \
                                        } else {                                                                                                    \
                                            val = x[base + tail_idx];                                                                                                 \
                                        }                                                                                                    \
                                    };                                                                                                    \
                                    auto blk_val = WarpTag::apply(val);                                                                                              \
                                    acc = (WarpBlockAccumExpr).template as<VT>();                                                                                              \
                                }                                                                                                    \
                                y[out_linear] = (FinalExpr).template as<VT>();                                                                                        \
                            }                                                                                                    \
                            return;                                                                                                    \
                        }                                                                                                    \
                    }                                                                                                    \
                                                                                                    \
                    /* Vectorized scalar path: contiguous reduced dimension, single thread */                                                                     \
                    if (reduced_ndim == 1 && reduced_in_strides[0] == 1) {                                                                                        \
                        if constexpr (std::is_same_v<VT, float> || std::is_same_v<VT, half>) {                                                                   \
                            bool use_buf_vec = x.container().is_byte_buffer();                                                                                        \
                            Var<ByteBuffer> *buf_x = nullptr;                                                                                                    \
                            uint off_x = 0;                                                                                                    \
                            if (use_buf_vec) {                                                                                                    \
                                buf_x = x.container().get_byte_buffer();                                                                                                \
                                off_x = static_cast<uint>(x.container().get_byte_offset());                                                                           \
                            }                                                                                                    \
                            auto type_size = static_cast<uint>(sizeof(VT));                                                                                           \
                            uint32_t vec_n = (reduced_size / 4u) * 4u;                                                                                          \
                            for (auto out_linear : dynamic_range(y.size())) {                                                                                         \
                                auto base = def(0u);                                                                                                    \
                                for_each_dim(out_linear, kept_out_strides, kept_ndim, [&](uint32_t d, auto coord) { base += coord * kept_in_strides[d]; }, y.size()); \
                                auto acc = def(VT(InitExpr));                                                                                                    \
                                if (use_buf_vec) {                                                                                                    \
                                    if constexpr (std::is_same_v<VT, float>) {                                                                                        \
                                        auto vec_acc = def(make_float4(VT(InitExpr)));                                                                                \
                                        for (uint32_t r = 0; r < vec_n; r += 4u) {                                                                                  \
                                            auto v = buf_x->read<float4>(off_x + (base + r) * type_size);                                                           \
                                            vec_acc = (VecAccumExpr).template as<float4>();                                                                           \
                                        }                                                                                                    \
                                        acc = (VecReduceExpr).template as<VT>();                                                                                      \
                                    } else {                                                                                                    \
                                        auto vec_acc = def(make_half4(VT(InitExpr)));                                                                                 \
                                        for (uint32_t r = 0; r < vec_n; r += 4u) {                                                                                  \
                                            auto v = buf_x->read<half4>(off_x + (base + r) * type_size);                                                            \
                                            vec_acc = (VecAccumExpr).template as<half4>();                                                                            \
                                        }                                                                                                    \
                                        acc = (VecReduceExpr).template as<VT>();                                                                                      \
                                    }                                                                                                    \
                                } else {                                                                                                    \
                                    if constexpr (std::is_same_v<VT, float>) {                                                                                        \
                                        auto vec_acc = def(make_float4(VT(InitExpr)));                                                                                \
                                        for (uint32_t r = 0; r < vec_n; r += 4u) {                                                                                  \
                                            auto v = make_float4(x[base + r + 0u], x[base + r + 1u], x[base + r + 2u], x[base + r + 3u]);                           \
                                            vec_acc = (VecAccumExpr).template as<float4>();                                                                           \
                                        }                                                                                                    \
                                        acc = (VecReduceExpr).template as<VT>();                                                                                      \
                                    } else {                                                                                                    \
                                        auto vec_acc = def(make_half4(VT(InitExpr)));                                                                                 \
                                        for (uint32_t r = 0; r < vec_n; r += 4u) {                                                                                  \
                                            auto v = make_half4(x[base + r + 0u], x[base + r + 1u], x[base + r + 2u], x[base + r + 3u]);                            \
                                            vec_acc = (VecAccumExpr).template as<half4>();                                                                            \
                                        }                                                                                                    \
                                        acc = (VecReduceExpr).template as<VT>();                                                                                      \
                                    }                                                                                                    \
                                }                                                                                                    \
                                if (use_buf_vec) {                                                                                                    \
                                    for (auto r : dynamic_range(vec_n, reduced_size)) {                                                                             \
                                        auto val = buf_x->read<VT>(off_x + (base + r) * type_size);                                                                 \
                                        acc = (AccumExpr).template as<VT>();                                                                                          \
                                    }                                                                                                    \
                                } else {                                                                                                    \
                                    for (auto r : dynamic_range(vec_n, reduced_size)) {                                                                             \
                                        auto val = x[base + r];                                                                                                    \
                                        acc = (AccumExpr).template as<VT>();                                                                                                    \
                                    }                                                                                                    \
                                }                                                                                                    \
                                y[out_linear] = (FinalExpr).template as<VT>();                                                                                        \
                            }                                                                                                    \
                            return;                                                                                                    \
                        }                                                                                                    \
                    }                                                                                                    \
                                                                                                    \
                    /* Scalar fallback path */                                                                                                    \
                    for (auto out_linear : dynamic_range(y.size())) {                                                                                                 \
                        auto base = def(0u);                                                                                                    \
                        for_each_dim(out_linear, kept_out_strides, kept_ndim, [&](uint32_t d, auto coord) { base += coord * kept_in_strides[d]; }, y.size());         \
                        auto acc = def(VT(InitExpr));                                                                                                    \
                        for (auto r : dynamic_range(reduced_size)) {                                                                                                  \
                            auto offset = def(0u);                                                                                                    \
                            for_each_dim(r, reduced_strides, reduced_ndim, [&](uint32_t d, auto coord) { offset += coord * reduced_in_strides[d]; }, reduced_size);   \
                            auto val = x[base + offset];                                                                                                    \
                            acc = (AccumExpr).template as<VT>();                                                                                                    \
                        }                                                                                                    \
                        y[out_linear] = (FinalExpr).template as<VT>();                                                                                                \
                    }                                                                                                    \
                }                                                                                                    \
            });                                                                                                    \
        }                                                                                                    \
    };                                                                                                    \
    REGISTER_TO_DEFAULT_OPSET(Name) {                                                                                                    \
        std::vector<int> axes;                                                                                                    \
        int keepdims = 1, noop = 0;                                                                                                    \
        if (auto p = node.try_get_attr("axes"))                                                                                                    \
            axes = p->get<onnx::AttributeType::INTS>();                                                                                                    \
        if (auto p = node.try_get_attr("keepdims"))                                                                                                    \
            keepdims = p->get<onnx::AttributeType::INT>();                                                                                                    \
        if (auto p = node.try_get_attr("noop_with_empty_axes"))                                                                                                   \
            noop = p->get<onnx::AttributeType::INT>();                                                                                                    \
        return std::make_unique<Name>(std::move(axes), keepdims, noop);                                                                                           \
    }

DEFINE_REDUCE_OP(ReduceSum,
                 detail::AppendFPQuantizedT<NNFilteredTypeList<IsNativeArithmetic>>,
                 0, acc + val, acc,
                 WarpReduceSum, 0, acc + blk_val,
                 vec_acc + v, vec_acc.x + vec_acc.y + vec_acc.z + vec_acc.w,
                 half(0), acc + val, acc);

DEFINE_REDUCE_OP(ReduceMean,
                 detail::AppendFPQuantizedT<NNFilteredTypeList<IsFloatingPoint>>,
                 0, acc + val, acc / VT(reduced_size),
                 WarpReduceSum, 0, acc + blk_val,
                 vec_acc + v, vec_acc.x + vec_acc.y + vec_acc.z + vec_acc.w,
                 half(0), acc + val, acc / half(reduced_size));

DEFINE_REDUCE_OP(ReduceMax,
                 detail::AppendFPQuantizedT<NNFilteredTypeList<IsNativeArithmetic>>,
                 detail::identity_max<VT>(), max(acc, val), acc,
                 WarpReduceMax, detail::identity_max<VT>(), max(acc, blk_val),
                 max(vec_acc, v), max(max(vec_acc.x, vec_acc.y), max(vec_acc.z, vec_acc.w)),
                 half(-65504.0f), max(acc, val), acc);

DEFINE_REDUCE_OP(ReduceMin,
                 detail::AppendFPQuantizedT<NNFilteredTypeList<IsNativeArithmetic>>,
                 detail::identity_min<VT>(), min(acc, val), acc,
                 WarpReduceMin, detail::identity_min<VT>(), min(acc, blk_val),
                 min(vec_acc, v), min(min(vec_acc.x, vec_acc.y), min(vec_acc.z, vec_acc.w)),
                 half(65504.0f), min(acc, val), acc);

DEFINE_REDUCE_OP(ReduceL1,
                 detail::AppendFPQuantizedT<NNFilteredTypeList<IsSigned>>,
                 0, acc + abs(val).template as<VT>(), acc,
                 WarpReduceNone, 0, acc,
                 vec_acc + abs(v), vec_acc.x + vec_acc.y + vec_acc.z + vec_acc.w,
                 half(0), acc + abs(val), acc);

DEFINE_REDUCE_OP(ReduceL2,
                 detail::AppendFPQuantizedT<NNFilteredTypeList<IsFloatingPoint>>,
                 0, luisa::compute::fma(val, val, acc), sqrt(acc),
                 WarpReduceNone, 0, acc,
                 luisa::compute::fma(v, v, vec_acc), vec_acc.x + vec_acc.y + vec_acc.z + vec_acc.w,
                 half(0), luisa::compute::fma(val, val, acc), sqrt(acc));

}// namespace lcml::onnx
