#include <limits>

#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

namespace lcml::onnx {

// Helper: compute stride for a given axis in a shape
static uint32_t axis_stride(ITensor::shape_type const &shape, uint32_t axis) {
    uint32_t s = 1;
    for (size_t d = axis + 1; d < shape.size(); ++d) s *= shape[d];
    return s;
}

// ArgMax: returns indices of maximum values along an axis.
// ONNX spec: attribute axis (default 0), keepdims (default 1), select_last_index (default 0)
class ArgMax : public Operator {
private:
    int64_t axis_;
    int keepdims_;
    int select_last_index_;

public:
    ArgMax(int64_t axis, int keepdims, int select_last_index)
        : Operator("ArgMax"), axis_(axis), keepdims_(keepdims), select_last_index_(select_last_index) {}

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 1 && outputs.size() == 1,
                     "ArgMax requires 1 input and 1 output.");
        auto &X = inputs[0].get();
        auto &Y = outputs[0].get();
        LUISA_ASSERT(Y.element_type() == typeid(int) || Y.element_type() == typeid(slong),
                     "ArgMax: output must be int or int64 type.");
#else
        auto &X = inputs[0].get();
        auto &Y = outputs[0].get();
#endif
        auto const &x_shape = X.shape();
        auto ndim = x_shape.size();
        auto axis = static_cast<uint32_t>(axis_ < 0 ? axis_ + ndim : axis_);
        auto axis_size = static_cast<uint32_t>(x_shape[axis]);
        auto ax_stride = axis_stride(x_shape, axis);

        visit_typeid<NNFilteredTypeList<IsFloatingPoint>>(X.element_type(), [&]<typename T>() {
            using VT = nn_storage_type_t<T>;
            auto &x = static_cast<NNTensor<T> &>(X);
            auto &y = static_cast<NNTensor<int> &>(Y);

            bool x_is_buf = x.container().is_byte_buffer();
            Var<ByteBuffer> *buf_x = nullptr;
            uint off_x = 0;
            if (x_is_buf) {
                buf_x = x.container().get_byte_buffer();
                off_x = static_cast<uint>(x.container().get_byte_offset());
            }
            auto type_size = static_cast<uint>(sizeof(VT));

            for (auto out_linear : dynamic_range(static_cast<uint>(y.size()))) {
                // Map output linear to input position (excluding axis)
                auto remaining = out_linear;
                auto in_base = def(0u);
                for (uint32_t d = 0; d < ndim; ++d) {
                    if (d == axis) {
                        if (keepdims_) {
                            // keepdims: output dim size is 1, coordinate is 0,
                            // remaining should not be consumed.
                        }
                        continue;
                    }
                    auto coord = remaining / Y.strides()[keepdims_ ? d : (d > axis ? d - 1 : d)];
                    remaining = remaining % Y.strides()[keepdims_ ? d : (d > axis ? d - 1 : d)];
                    in_base += coord * X.strides()[d];
                }

                auto max_val = def(std::is_same_v<VT, half> ? VT(-65504.0f) : VT(-std::numeric_limits<float>::infinity()));
                auto max_idx = def(0);
                for (auto i : dynamic_range(axis_size)) {
                    auto val = def(VT{0});
                    if (x_is_buf) {
                        val = buf_x->read<VT>(off_x + (in_base + i * ax_stride) * type_size);
                    } else {
                        val = x[in_base + i * ax_stride];
                    }
                    if (select_last_index_) {
                        $if (val >= max_val) {
                            max_val = val;
                            max_idx = i.cast<int>();
                        };
                    } else {
                        $if (val > max_val) {
                            max_val = val;
                            max_idx = i.cast<int>();
                        };
                    }
                }
                y[out_linear] = max_idx;
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(ArgMax) {
    int64_t axis = 0;
    int keepdims = 1, sli = 0;
    if (auto p = node.try_get_attr("axis")) axis = p->get<onnx::AttributeType::INT>();
    if (auto p = node.try_get_attr("keepdims")) keepdims = p->get<onnx::AttributeType::INT>();
    if (auto p = node.try_get_attr("select_last_index")) sli = p->get<onnx::AttributeType::INT>();
    return std::make_unique<ArgMax>(axis, keepdims, sli);
};

// ArgMin: same as ArgMax but finds minimum
class ArgMin : public Operator {
private:
    int64_t axis_;
    int keepdims_;

public:
    ArgMin(int64_t axis, int keepdims) : Operator("ArgMin"), axis_(axis), keepdims_(keepdims) {}

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 1 && outputs.size() == 1,
                     "ArgMin requires 1 input and 1 output.");
        auto &X = inputs[0].get();
        auto &Y = outputs[0].get();
        LUISA_ASSERT(Y.element_type() == typeid(int) || Y.element_type() == typeid(slong),
                     "ArgMin: output must be int or int64 type.");
#else
        auto &X = inputs[0].get();
        auto &Y = outputs[0].get();
#endif
        auto const &x_shape = X.shape();
        auto ndim = x_shape.size();
        auto axis = static_cast<uint32_t>(axis_ < 0 ? axis_ + ndim : axis_);
        auto axis_size = static_cast<uint32_t>(x_shape[axis]);
        auto ax_stride = axis_stride(x_shape, axis);

        visit_typeid<NNFilteredTypeList<IsFloatingPoint>>(X.element_type(), [&]<typename T>() {
            using VT = nn_storage_type_t<T>;
            auto &x = static_cast<NNTensor<T> &>(X);
            auto &y = static_cast<NNTensor<int> &>(Y);

            bool x_is_buf = x.container().is_byte_buffer();
            Var<ByteBuffer> *buf_x = nullptr;
            uint off_x = 0;
            if (x_is_buf) {
                buf_x = x.container().get_byte_buffer();
                off_x = static_cast<uint>(x.container().get_byte_offset());
            }
            auto type_size = static_cast<uint>(sizeof(VT));

            for (auto out_linear : dynamic_range(static_cast<uint>(y.size()))) {
                auto remaining = out_linear;
                auto in_base = def(0u);
                for (uint32_t d = 0; d < ndim; ++d) {
                    if (d == axis) {
                        if (keepdims_) {
                            // keepdims: coordinate is 0, do not consume remaining
                        }
                        continue;
                    }
                    auto coord = remaining / Y.strides()[keepdims_ ? d : (d > axis ? d - 1 : d)];
                    remaining = remaining % Y.strides()[keepdims_ ? d : (d > axis ? d - 1 : d)];
                    in_base += coord * X.strides()[d];
                }

                auto min_val = def(std::is_same_v<VT, half> ? VT(65504.0f) : VT(std::numeric_limits<float>::infinity()));
                auto min_idx = def(0);
                for (auto i : dynamic_range(axis_size)) {
                    auto val = def(VT{0});
                    if (x_is_buf) {
                        val = buf_x->read<VT>(off_x + (in_base + i * ax_stride) * type_size);
                    } else {
                        val = x[in_base + i * ax_stride];
                    }
                    $if (val < min_val) {
                        min_val = val;
                        min_idx = i.cast<int>();
                    };
                }
                y[out_linear] = min_idx;
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(ArgMin) {
    int64_t axis = 0;
    int keepdims = 1;
    if (auto p = node.try_get_attr("axis")) axis = p->get<onnx::AttributeType::INT>();
    if (auto p = node.try_get_attr("keepdims")) keepdims = p->get<onnx::AttributeType::INT>();
    return std::make_unique<ArgMin>(axis, keepdims);
};

namespace detail {
template<typename Tuple>
struct SoftmaxAppendFPQuantized;
template<typename... Ts>
struct SoftmaxAppendFPQuantized<std::tuple<Ts...>> {
    using type = std::tuple<Ts..., FP8E4M3FN, FP8E5M2, FP4E2M1>;
};
template<typename Tuple>
using SoftmaxAppendFPQuantizedT = typename SoftmaxAppendFPQuantized<Tuple>::type;
}// namespace detail

// Type list for Softmax/LogSoftmax including quantized FP types
using SoftmaxTypeList = detail::SoftmaxAppendFPQuantizedT<NNFilteredTypeList<IsFloatingPoint>>;

// Softmax: softmax(x)_i = exp(x_i) / sum(exp(x_j)) along axis
// ONNX spec: attribute axis (default -1)
class Softmax : public Operator {
private:
    int64_t axis_;

public:
    Softmax(int64_t axis) : Operator("Softmax"), axis_(axis) {}

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 1 && outputs.size() == 1,
                     "Softmax requires 1 input and 1 output.");
        auto &X = inputs[0].get();
        auto &Y = outputs[0].get();
        LUISA_ASSERT(X.element_type() == Y.element_type(),
                     "Softmax: input and output must have the same element type.");
#else
        auto &X = inputs[0].get();
        auto &Y = outputs[0].get();
#endif
        auto const &x_shape = X.shape();
        auto ndim = x_shape.size();
        auto axis = static_cast<uint32_t>(axis_ < 0 ? axis_ + ndim : axis_);
        auto axis_size = static_cast<uint32_t>(x_shape[axis]);
        auto ax_stride = axis_stride(x_shape, axis);

        visit_typeid<SoftmaxTypeList>(X.element_type(), [&]<typename T>() {
            using VT = nn_storage_type_t<T>;
            auto &x = static_cast<NNTensor<T> &>(X);
            auto &y = static_cast<NNTensor<T> &>(Y);

            // Quantized FP8/FP4 path: dequantize to half, compute, requantize
            if constexpr (std::is_same_v<T, FP8E4M3FN> || std::is_same_v<T, FP8E5M2> || std::is_same_v<T, FP4E2M1>) {
                auto deq = [&]() {
                    if constexpr (std::is_same_v<T, FP8E4M3FN>) return fp8e4m3_to_float();
                    else if constexpr (std::is_same_v<T, FP8E5M2>) return fp8e5m2_to_float();
                    else return fp4e2m1_to_float();
                }();
                auto q = [&]() {
                    if constexpr (std::is_same_v<T, FP8E4M3FN>) return fp8e4m3_from_float();
                    else if constexpr (std::is_same_v<T, FP8E5M2>) return fp8e5m2_from_float();
                    else return fp4e2m1_from_float();
                }();
                for (auto out_linear : dynamic_range(static_cast<uint>(y.size()))) {
                    auto axis_coord = extract_coord(out_linear, y.strides(), y.shape(), axis);
                    auto base = out_linear - axis_coord * ax_stride;
                    auto max_val = def(half(-65504.0f));
                    for (auto i : dynamic_range(axis_size)) {
                        auto bits = x[base + i * ax_stride].bits.cast<ushort>();
                        auto val = deq(bits);
                        max_val = max(max_val, val);
                    }
                    auto sum_exp = def(half{0});
                    for (auto i : dynamic_range(axis_size)) {
                        auto bits = x[base + i * ax_stride].bits.cast<ushort>();
                        auto val = deq(bits);
                        sum_exp += exp(val - max_val);
                    }
                    auto out_bits = x[out_linear].bits.cast<ushort>();
                    auto out_val = deq(out_bits);
                    auto result = exp(out_val - max_val) / sum_exp;
                    y[out_linear].bits = q(result).cast<uint16_t>();
                }
                return;
            }

            // Native arithmetic path
            if constexpr (IsNativeArithmetic<T>::value) {
                bool x_is_buf = x.container().is_byte_buffer();
                Var<ByteBuffer> *buf_x = nullptr;
                uint off_x = 0;
                if (x_is_buf) {
                    buf_x = x.container().get_byte_buffer();
                    off_x = static_cast<uint>(x.container().get_byte_offset());
                }
                bool y_is_buf = y.container().is_byte_buffer();
                Var<ByteBuffer> *buf_y = nullptr;
                uint off_y = 0;
                if (y_is_buf) {
                    buf_y = y.container().get_byte_buffer();
                    off_y = static_cast<uint>(y.container().get_byte_offset());
                }
                auto type_size = static_cast<uint>(sizeof(VT));

                if (ax_stride == 1 && axis_size >= 4) {
                    uint32_t vec_n = (axis_size / 4u) * 4u;
                    for (auto out_linear : dynamic_range(static_cast<uint>(y.size()))) {
                        auto axis_coord = extract_coord(out_linear, y.strides(), y.shape(), axis);
                        auto base = out_linear - axis_coord * ax_stride;

                        auto max_val = def(std::is_same_v<VT, half> ? VT(-65504.0f) : VT(-std::numeric_limits<float>::infinity()));
                        if constexpr (std::is_same_v<VT, float>) {
                            auto vec_max = def(make_float4(max_val));
                            if (x_is_buf) {
                                for (uint32_t i = 0; i < vec_n; i += 4u) {
                                    auto v = buf_x->read<float4>(off_x + (base + i) * type_size);
                                    vec_max = max(vec_max, v);
                                }
                            } else {
                                for (uint32_t i = 0; i < vec_n; i += 4u) {
                                    auto v = make_float4(x[base + i + 0u], x[base + i + 1u], x[base + i + 2u], x[base + i + 3u]);
                                    vec_max = max(vec_max, v);
                                }
                            }
                            max_val = max(max_val, max(max(vec_max.x, vec_max.y), max(vec_max.z, vec_max.w)));
                        } else {
                            auto vec_max = def(make_half4(max_val));
                            if (x_is_buf) {
                                for (uint32_t i = 0; i < vec_n; i += 4u) {
                                    auto v = buf_x->read<half4>(off_x + (base + i) * type_size);
                                    vec_max = max(vec_max, v);
                                }
                            } else {
                                for (uint32_t i = 0; i < vec_n; i += 4u) {
                                    auto v = make_half4(x[base + i + 0u], x[base + i + 1u], x[base + i + 2u], x[base + i + 3u]);
                                    vec_max = max(vec_max, v);
                                }
                            }
                            max_val = max(max_val, max(max(vec_max.x, vec_max.y), max(vec_max.z, vec_max.w)));
                        }
                        if (x_is_buf) {
                            for (auto i : dynamic_range(vec_n, axis_size)) {
                                max_val = max(max_val, buf_x->read<VT>(off_x + (base + i) * type_size));
                            }
                        } else {
                            for (auto i : dynamic_range(vec_n, axis_size)) {
                                max_val = max(max_val, x[base + i]);
                            }
                        }

                        auto sum_exp = def(VT{0});
                        if constexpr (std::is_same_v<VT, float>) {
                            auto vec_sum = def(make_float4(VT{0}));
                            if (x_is_buf) {
                                for (uint32_t i = 0; i < vec_n; i += 4u) {
                                    auto v = buf_x->read<float4>(off_x + (base + i) * type_size);
                                    vec_sum += exp(v - make_float4(max_val));
                                }
                            } else {
                                for (uint32_t i = 0; i < vec_n; i += 4u) {
                                    auto v = make_float4(x[base + i + 0u], x[base + i + 1u], x[base + i + 2u], x[base + i + 3u]);
                                    vec_sum += exp(v - make_float4(max_val));
                                }
                            }
                            sum_exp += vec_sum.x + vec_sum.y + vec_sum.z + vec_sum.w;
                        } else {
                            auto vec_sum = def(make_half4(VT{0}));
                            if (x_is_buf) {
                                for (uint32_t i = 0; i < vec_n; i += 4u) {
                                    auto v = buf_x->read<half4>(off_x + (base + i) * type_size);
                                    vec_sum += exp(v - make_half4(max_val));
                                }
                            } else {
                                for (uint32_t i = 0; i < vec_n; i += 4u) {
                                    auto v = make_half4(x[base + i + 0u], x[base + i + 1u], x[base + i + 2u], x[base + i + 3u]);
                                    vec_sum += exp(v - make_half4(max_val));
                                }
                            }
                            sum_exp += vec_sum.x + vec_sum.y + vec_sum.z + vec_sum.w;
                        }
                        if (x_is_buf) {
                            for (auto i : dynamic_range(vec_n, axis_size)) {
                                sum_exp += exp(buf_x->read<VT>(off_x + (base + i) * type_size) - max_val);
                            }
                        } else {
                            for (auto i : dynamic_range(vec_n, axis_size)) {
                                sum_exp += exp(x[base + i] - max_val);
                            }
                        }

                        auto result = def(VT{0});
                        if (x_is_buf) {
                            auto out_val = buf_x->read<VT>(off_x + out_linear * type_size);
                            result = exp(out_val - max_val) / sum_exp;
                        } else {
                            result = exp(x[out_linear] - max_val) / sum_exp;
                        }
                        if (y_is_buf) {
                            buf_y->write(off_y + out_linear * type_size, result);
                        } else {
                            y[out_linear] = result;
                        }
                    }
                } else {
                    for (auto out_linear : dynamic_range(static_cast<uint>(y.size()))) {
                        auto axis_coord = extract_coord(out_linear, y.strides(), y.shape(), axis);
                        auto base = out_linear - axis_coord * ax_stride;

                        auto max_val = def(std::is_same_v<VT, half> ? VT(-65504.0f) : VT(-std::numeric_limits<float>::infinity()));
                        if (x_is_buf) {
                            for (auto i : dynamic_range(axis_size)) {
                                max_val = max(max_val, buf_x->read<VT>(off_x + (base + i * ax_stride) * type_size));
                            }
                        } else {
                            for (auto i : dynamic_range(axis_size)) {
                                max_val = max(max_val, x[base + i * ax_stride]);
                            }
                        }
                        auto sum_exp = def(VT{0});
                        if (x_is_buf) {
                            for (auto i : dynamic_range(axis_size)) {
                                sum_exp += exp(buf_x->read<VT>(off_x + (base + i * ax_stride) * type_size) - max_val);
                            }
                        } else {
                            for (auto i : dynamic_range(axis_size)) {
                                sum_exp += exp(x[base + i * ax_stride] - max_val);
                            }
                        }
                        auto result = def(VT{0});
                        if (x_is_buf) {
                            auto out_val = buf_x->read<VT>(off_x + out_linear * type_size);
                            result = exp(out_val - max_val) / sum_exp;
                        } else {
                            result = exp(x[out_linear] - max_val) / sum_exp;
                        }
                        if (y_is_buf) {
                            buf_y->write(off_y + out_linear * type_size, result);
                        } else {
                            y[out_linear] = result;
                        }
                    }
                }
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(Softmax) {
    int64_t axis = -1;
    if (auto p = node.try_get_attr("axis")) axis = p->get<onnx::AttributeType::INT>();
    return std::make_unique<Softmax>(axis);
};

// LogSoftmax: log(softmax(x))
class LogSoftmax : public Operator {
private:
    int64_t axis_;

public:
    LogSoftmax(int64_t axis) : Operator("LogSoftmax"), axis_(axis) {}

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 1 && outputs.size() == 1,
                     "LogSoftmax requires 1 input and 1 output.");
        auto &X = inputs[0].get();
        auto &Y = outputs[0].get();
        LUISA_ASSERT(X.element_type() == Y.element_type(),
                     "LogSoftmax: input and output must have the same element type.");
#else
        auto &X = inputs[0].get();
        auto &Y = outputs[0].get();
#endif
        auto const &x_shape = X.shape();
        auto ndim = x_shape.size();
        auto axis = static_cast<uint32_t>(axis_ < 0 ? axis_ + ndim : axis_);
        auto axis_size = static_cast<uint32_t>(x_shape[axis]);
        auto ax_stride = axis_stride(x_shape, axis);

        visit_typeid<SoftmaxTypeList>(X.element_type(), [&]<typename T>() {
            using VT = nn_storage_type_t<T>;
            auto &x = static_cast<NNTensor<T> &>(X);
            auto &y = static_cast<NNTensor<T> &>(Y);

            // Quantized FP8/FP4 path
            if constexpr (std::is_same_v<T, FP8E4M3FN> || std::is_same_v<T, FP8E5M2> || std::is_same_v<T, FP4E2M1>) {
                auto deq = [&]() {
                    if constexpr (std::is_same_v<T, FP8E4M3FN>) return fp8e4m3_to_float();
                    else if constexpr (std::is_same_v<T, FP8E5M2>) return fp8e5m2_to_float();
                    else return fp4e2m1_to_float();
                }();
                auto q = [&]() {
                    if constexpr (std::is_same_v<T, FP8E4M3FN>) return fp8e4m3_from_float();
                    else if constexpr (std::is_same_v<T, FP8E5M2>) return fp8e5m2_from_float();
                    else return fp4e2m1_from_float();
                }();
                for (auto out_linear : dynamic_range(static_cast<uint>(y.size()))) {
                    auto axis_coord = extract_coord(out_linear, y.strides(), y.shape(), axis);
                    auto base = out_linear - axis_coord * ax_stride;
                    auto max_val = def(half(-65504.0f));
                    for (auto i : dynamic_range(axis_size)) {
                        auto bits = x[base + i * ax_stride].bits.cast<ushort>();
                        auto val = deq(bits);
                        max_val = max(max_val, val);
                    }
                    auto sum_exp = def(half{0});
                    for (auto i : dynamic_range(axis_size)) {
                        auto bits = x[base + i * ax_stride].bits.cast<ushort>();
                        auto val = deq(bits);
                        sum_exp += exp(val - max_val);
                    }
                    auto out_bits = x[out_linear].bits.cast<ushort>();
                    auto out_val = deq(out_bits);
                    auto result = (out_val - max_val) - log(sum_exp);
                    y[out_linear].bits = q(result).cast<uint16_t>();
                }
                return;
            }

            // Native arithmetic path
            if constexpr (IsNativeArithmetic<T>::value) {
                bool x_is_buf = x.container().is_byte_buffer();
                Var<ByteBuffer> *buf_x = nullptr;
                uint off_x = 0;
                if (x_is_buf) {
                    buf_x = x.container().get_byte_buffer();
                    off_x = static_cast<uint>(x.container().get_byte_offset());
                }
                bool y_is_buf = y.container().is_byte_buffer();
                Var<ByteBuffer> *buf_y = nullptr;
                uint off_y = 0;
                if (y_is_buf) {
                    buf_y = y.container().get_byte_buffer();
                    off_y = static_cast<uint>(y.container().get_byte_offset());
                }
                auto type_size = static_cast<uint>(sizeof(VT));

                if (ax_stride == 1 && axis_size >= 4) {
                    uint32_t vec_n = (axis_size / 4u) * 4u;
                    for (auto out_linear : dynamic_range(static_cast<uint>(y.size()))) {
                        auto axis_coord = extract_coord(out_linear, y.strides(), y.shape(), axis);
                        auto base = out_linear - axis_coord * ax_stride;

                        auto max_val = def(std::is_same_v<VT, half> ? VT(-65504.0f) : VT(-std::numeric_limits<float>::infinity()));
                        if constexpr (std::is_same_v<VT, float>) {
                            auto vec_max = def(make_float4(max_val));
                            if (x_is_buf) {
                                for (uint32_t i = 0; i < vec_n; i += 4u) {
                                    auto v = buf_x->read<float4>(off_x + (base + i) * type_size);
                                    vec_max = max(vec_max, v);
                                }
                            } else {
                                for (uint32_t i = 0; i < vec_n; i += 4u) {
                                    auto v = make_float4(x[base + i + 0u], x[base + i + 1u], x[base + i + 2u], x[base + i + 3u]);
                                    vec_max = max(vec_max, v);
                                }
                            }
                            max_val = max(max_val, max(max(vec_max.x, vec_max.y), max(vec_max.z, vec_max.w)));
                        } else {
                            auto vec_max = def(make_half4(max_val));
                            if (x_is_buf) {
                                for (uint32_t i = 0; i < vec_n; i += 4u) {
                                    auto v = buf_x->read<half4>(off_x + (base + i) * type_size);
                                    vec_max = max(vec_max, v);
                                }
                            } else {
                                for (uint32_t i = 0; i < vec_n; i += 4u) {
                                    auto v = make_half4(x[base + i + 0u], x[base + i + 1u], x[base + i + 2u], x[base + i + 3u]);
                                    vec_max = max(vec_max, v);
                                }
                            }
                            max_val = max(max_val, max(max(vec_max.x, vec_max.y), max(vec_max.z, vec_max.w)));
                        }
                        if (x_is_buf) {
                            for (auto i : dynamic_range(vec_n, axis_size)) {
                                max_val = max(max_val, buf_x->read<VT>(off_x + (base + i) * type_size));
                            }
                        } else {
                            for (auto i : dynamic_range(vec_n, axis_size)) {
                                max_val = max(max_val, x[base + i]);
                            }
                        }

                        auto sum_exp = def(VT{0});
                        if constexpr (std::is_same_v<VT, float>) {
                            auto vec_sum = def(make_float4(VT{0}));
                            if (x_is_buf) {
                                for (uint32_t i = 0; i < vec_n; i += 4u) {
                                    auto v = buf_x->read<float4>(off_x + (base + i) * type_size);
                                    vec_sum += exp(v - make_float4(max_val));
                                }
                            } else {
                                for (uint32_t i = 0; i < vec_n; i += 4u) {
                                    auto v = make_float4(x[base + i + 0u], x[base + i + 1u], x[base + i + 2u], x[base + i + 3u]);
                                    vec_sum += exp(v - make_float4(max_val));
                                }
                            }
                            sum_exp += vec_sum.x + vec_sum.y + vec_sum.z + vec_sum.w;
                        } else {
                            auto vec_sum = def(make_half4(VT{0}));
                            if (x_is_buf) {
                                for (uint32_t i = 0; i < vec_n; i += 4u) {
                                    auto v = buf_x->read<half4>(off_x + (base + i) * type_size);
                                    vec_sum += exp(v - make_half4(max_val));
                                }
                            } else {
                                for (uint32_t i = 0; i < vec_n; i += 4u) {
                                    auto v = make_half4(x[base + i + 0u], x[base + i + 1u], x[base + i + 2u], x[base + i + 3u]);
                                    vec_sum += exp(v - make_half4(max_val));
                                }
                            }
                            sum_exp += vec_sum.x + vec_sum.y + vec_sum.z + vec_sum.w;
                        }
                        if (x_is_buf) {
                            for (auto i : dynamic_range(vec_n, axis_size)) {
                                sum_exp += exp(buf_x->read<VT>(off_x + (base + i) * type_size) - max_val);
                            }
                        } else {
                            for (auto i : dynamic_range(vec_n, axis_size)) {
                                sum_exp += exp(x[base + i] - max_val);
                            }
                        }

                        auto result = def(VT{0});
                        if (x_is_buf) {
                            auto out_val = buf_x->read<VT>(off_x + out_linear * type_size);
                            result = (out_val - max_val) - log(sum_exp);
                        } else {
                            result = (x[out_linear] - max_val) - log(sum_exp);
                        }
                        if (y_is_buf) {
                            buf_y->write(off_y + out_linear * type_size, result);
                        } else {
                            y[out_linear] = result;
                        }
                    }
                } else {
                    for (auto out_linear : dynamic_range(static_cast<uint>(y.size()))) {
                        auto axis_coord = extract_coord(out_linear, y.strides(), y.shape(), axis);
                        auto base = out_linear - axis_coord * ax_stride;

                        auto max_val = def(std::is_same_v<VT, half> ? VT(-65504.0f) : VT(-std::numeric_limits<float>::infinity()));
                        if (x_is_buf) {
                            for (auto i : dynamic_range(axis_size)) {
                                max_val = max(max_val, buf_x->read<VT>(off_x + (base + i * ax_stride) * type_size));
                            }
                        } else {
                            for (auto i : dynamic_range(axis_size)) {
                                max_val = max(max_val, x[base + i * ax_stride]);
                            }
                        }
                        auto sum_exp = def(VT{0});
                        if (x_is_buf) {
                            for (auto i : dynamic_range(axis_size)) {
                                sum_exp += exp(buf_x->read<VT>(off_x + (base + i * ax_stride) * type_size) - max_val);
                            }
                        } else {
                            for (auto i : dynamic_range(axis_size)) {
                                sum_exp += exp(x[base + i * ax_stride] - max_val);
                            }
                        }
                        auto result = def(VT{0});
                        if (x_is_buf) {
                            auto out_val = buf_x->read<VT>(off_x + out_linear * type_size);
                            result = (out_val - max_val) - log(sum_exp);
                        } else {
                            result = (x[out_linear] - max_val) - log(sum_exp);
                        }
                        if (y_is_buf) {
                            buf_y->write(off_y + out_linear * type_size, result);
                        } else {
                            y[out_linear] = result;
                        }
                    }
                }
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(LogSoftmax) {
    int64_t axis = -1;
    if (auto p = node.try_get_attr("axis")) axis = p->get<onnx::AttributeType::INT>();
    return std::make_unique<LogSoftmax>(axis);
};

// Hardmax: sets 1 at the position of the max value along axis, 0 elsewhere
class Hardmax : public Operator {
private:
    int64_t axis_;

public:
    Hardmax(int64_t axis) : Operator("Hardmax"), axis_(axis) {}

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 1 && outputs.size() == 1,
                     "Hardmax requires 1 input and 1 output.");
        auto &X = inputs[0].get();
        auto &Y = outputs[0].get();
        LUISA_ASSERT(X.element_type() == Y.element_type(),
                     "Hardmax: input and output must have the same element type.");
#else
        auto &X = inputs[0].get();
        auto &Y = outputs[0].get();
#endif
        auto const &x_shape = X.shape();
        auto ndim = x_shape.size();
        auto axis = static_cast<uint32_t>(axis_ < 0 ? axis_ + ndim : axis_);
        auto axis_size = static_cast<uint32_t>(x_shape[axis]);
        auto ax_stride = axis_stride(x_shape, axis);

        visit_typeid<NNFilteredTypeList<IsFloatingPoint>>(X.element_type(), [&]<typename T>() {
            using VT = nn_storage_type_t<T>;
            auto &x = static_cast<NNTensor<T> &>(X);
            auto &y = static_cast<NNTensor<T> &>(Y);

            bool x_is_buf = x.container().is_byte_buffer();
            Var<ByteBuffer> *buf_x = nullptr;
            uint off_x = 0;
            if (x_is_buf) {
                buf_x = x.container().get_byte_buffer();
                off_x = static_cast<uint>(x.container().get_byte_offset());
            }
            auto type_size = static_cast<uint>(sizeof(VT));

            for (auto out_linear : dynamic_range(static_cast<uint>(y.size()))) {
                // input/output have identical shape, so in_base == out_linear
                auto axis_coord = extract_coord(out_linear, y.strides(), y.shape(), axis);
                auto base = out_linear - axis_coord * ax_stride;

                // Find argmax
                auto max_val = def(std::is_same_v<VT, half> ? VT(-65504.0f) : VT(-std::numeric_limits<float>::infinity()));
                auto max_idx = def(0u);
                for (auto i : dynamic_range(axis_size)) {
                    auto val = def(VT{0});
                    if (x_is_buf) {
                        val = buf_x->read<VT>(off_x + (base + i * ax_stride) * type_size);
                    } else {
                        val = x[base + i * ax_stride];
                    }
                    $if (val > max_val) {
                        max_val = val;
                        max_idx = i;
                    };
                }
                y[out_linear] = select(Var<VT>{VT{0}}, Var<VT>{VT{1}}, axis_coord == max_idx);
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(Hardmax) {
    int64_t axis = -1;
    if (auto p = node.try_get_attr("axis")) axis = p->get<onnx::AttributeType::INT>();
    return std::make_unique<Hardmax>(axis);
};

}// namespace lcml::onnx
