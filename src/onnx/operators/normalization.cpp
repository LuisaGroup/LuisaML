#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

namespace lcml::onnx {

namespace {

template<typename VT>
inline auto make_zero_vec() {
    if constexpr (std::is_same_v<VT, float>) {
        return make_float4(0.0f);
    } else if constexpr (std::is_same_v<VT, half>) {
        return make_half4(half(0.0f));
    }
}

template<typename VT, typename TensorT>
inline auto gather_vec4(TensorT &tensor, Var<uint> off) {
    if constexpr (std::is_same_v<VT, float>) {
        return make_float4(tensor[off + 0u], tensor[off + 1u], tensor[off + 2u], tensor[off + 3u]);
    } else {
        return make_half4(tensor[off + 0u], tensor[off + 1u], tensor[off + 2u], tensor[off + 3u]);
    }
}

template<typename VT, typename TensorT, typename VecT>
inline void scatter_vec4(TensorT &tensor, Var<uint> off, VecT vec) {
    tensor[off + 0u] = vec.x;
    tensor[off + 1u] = vec.y;
    tensor[off + 2u] = vec.z;
    tensor[off + 3u] = vec.w;
}

template<typename VT>
inline auto broadcast_vec4(Var<VT> s) {
    if constexpr (std::is_same_v<VT, float>) return make_float4(s);
    else return make_half4(s);
}

// Vectorized read: use ByteBuffer vectorized load when available, else scalar gather.
template<typename VT, typename TensorT>
inline auto read_vec4(TensorT &tensor, Var<uint> elem_off,
                      luisa::compute::Var<luisa::compute::ByteBuffer> *buf, uint byte_off) {
    using VecT = typename detail::VecDispatch<VT>::VecT;
    if (buf) {
        return buf->read<VecT>(byte_off + elem_off * static_cast<uint>(sizeof(VT)));
    }
    return gather_vec4<VT>(tensor, elem_off);
}

// Vectorized write: use ByteBuffer vectorized store when available, else scalar scatter.
template<typename VT, typename TensorT, typename VecT>
inline void write_vec4(TensorT &tensor, Var<uint> elem_off, VecT vec,
                       luisa::compute::Var<luisa::compute::ByteBuffer> *buf, uint byte_off) {
    if (buf) {
        buf->write(byte_off + elem_off * static_cast<uint>(sizeof(VT)), vec);
        return;
    }
    scatter_vec4<VT>(tensor, elem_off, vec);
}

}// namespace

// BatchNormalization: normalizes input across batch dimension.
// ONNX spec: inputs: X(N,C,D1,...), scale(C), B(C), input_mean(C), input_var(C)
// Attribute: epsilon (default 1e-5), momentum (not used in inference), training_mode (default 0)
class BatchNormalization : public Operator {
private:
    float epsilon_;

public:
    BatchNormalization(float epsilon) : Operator("BatchNormalization"), epsilon_(epsilon) {}

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 5 && outputs.size() >= 1,
                     "BatchNormalization requires 5 inputs and >=1 output.");
#endif
        auto &X = inputs[0].get();
        auto &Y = outputs[0].get();

        auto const &x_shape = X.shape();
#ifndef NDEBUG
        LUISA_ASSERT(x_shape.size() >= 2, "BatchNormalization input must be at least 2D.");
#endif
        uint32_t N = x_shape[0];
        uint32_t C = x_shape[1];
        uint32_t spatial_size = 1;
        for (size_t i = 2; i < x_shape.size(); ++i)
            spatial_size *= x_shape[i];

        visit_typeid<NNFilteredTypeList<IsFloatingPoint>>(X.element_type(), [&]<typename T>() {
            using VT = nn_storage_type_t<T>;
            auto &x = static_cast<NNTensor<T> &>(X);
            auto &scale = static_cast<NNTensor<T> &>(inputs[1].get());
            auto &bias = static_cast<NNTensor<T> &>(inputs[2].get());
            auto &mean = static_cast<NNTensor<T> &>(inputs[3].get());
            auto &var = static_cast<NNTensor<T> &>(inputs[4].get());
            auto &y = static_cast<NNTensor<T> &>(Y);
            auto eps = Var<VT>{VT(sizeof(VT) == 2 ? 1e-3f : epsilon_)};

            bool x_is_bb = x.container().is_byte_buffer();
            bool y_is_bb = y.container().is_byte_buffer();
            auto *buf_x = x_is_bb ? x.container().get_byte_buffer() : nullptr;
            auto *buf_y = y_is_bb ? y.container().get_byte_buffer() : nullptr;
            uint off_x = x_is_bb ? static_cast<uint>(x.container().get_byte_offset()) : 0u;
            uint off_y = y_is_bb ? static_cast<uint>(y.container().get_byte_offset()) : 0u;

            for (auto n : dynamic_range(N)) {
                for (auto c : dynamic_range(C)) {
                    auto s = scale[c];
                    auto b = bias[c];
                    auto m = mean[c];
                    auto v = var[c];
                    auto inv_std = s / sqrt(v + eps);
                    auto base_in = n * x.strides()[0] + c * x.strides()[1];
                    auto base_out = n * y.strides()[0] + c * y.strides()[1];
                    auto linear_bias = b - m * inv_std;

                    if constexpr (std::is_same_v<VT, float> || std::is_same_v<VT, half>) {
                        constexpr uint32_t VecN = detail::VecDispatch<VT>::N;
                        auto vec_count = spatial_size / VecN;
                        auto rem = spatial_size % VecN;
                        auto inv_std_vec = broadcast_vec4<VT>(inv_std);
                        auto bias_vec = broadcast_vec4<VT>(linear_bias);

                        if (x_is_bb && y_is_bb) {
                            for (auto i : dynamic_range(vec_count)) {
                                auto off = i * VecN;
                                auto xv = read_vec4<VT>(x, base_in + off, buf_x, off_x);
                                auto yv = luisa::compute::fma(xv, inv_std_vec, bias_vec);
                                write_vec4<VT>(y, base_out + off, yv, buf_y, off_y);
                            }
                        } else {
                            for (auto i : dynamic_range(vec_count)) {
                                auto off_in = base_in + i * VecN;
                                auto off_out = base_out + i * VecN;
                                auto xv = gather_vec4<VT>(x, off_in);
                                auto yv = luisa::compute::fma(xv, inv_std_vec, bias_vec);
                                scatter_vec4<VT>(y, off_out, yv);
                            }
                        }
                        auto start = vec_count * VecN;
                        for (auto i : dynamic_range(rem)) {
                            y[base_out + start + i] = luisa::compute::fma(x[base_in + start + i], inv_std, linear_bias);
                        }
                    } else {
                        for (auto i : dynamic_range(spatial_size)) {
                            y[base_out + i] = luisa::compute::fma(x[base_in + i], inv_std, linear_bias);
                        }
                    }
                }
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(BatchNormalization) {
    float epsilon = 1e-5f;
    if (auto p = node.try_get_attr("epsilon"))
        epsilon = p->get<onnx::AttributeType::FLOAT>();
    return std::make_unique<BatchNormalization>(epsilon);
};

// InstanceNormalization: y = scale * (x - mean) / sqrt(var + epsilon) + B
// Per (N, C) pair, normalize over spatial dims.
// ONNX spec: inputs: X(N,C,D1,...), scale(C), B(C); attribute epsilon
class InstanceNormalization : public Operator {
private:
    float epsilon_;

public:
    InstanceNormalization(float epsilon) : Operator("InstanceNormalization"), epsilon_(epsilon) {}

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 3 && outputs.size() == 1,
                     "InstanceNormalization requires 3 inputs and 1 output.");
#endif
        auto &X = inputs[0].get();
        auto &Y = outputs[0].get();

        auto const &x_shape = X.shape();
#ifndef NDEBUG
        LUISA_ASSERT(x_shape.size() >= 2, "InstanceNormalization input must be at least 2D.");
#endif
        uint32_t N = x_shape[0];
        uint32_t C = x_shape[1];
        uint32_t spatial_size = 1;
        for (size_t i = 2; i < x_shape.size(); ++i)
            spatial_size *= x_shape[i];

        visit_typeid<NNFilteredTypeList<IsFloatingPoint>>(X.element_type(), [&]<typename T>() {
            using VT = nn_storage_type_t<T>;
            auto &x = static_cast<NNTensor<T> &>(X);
            auto &scale = static_cast<NNTensor<T> &>(inputs[1].get());
            auto &bias = static_cast<NNTensor<T> &>(inputs[2].get());
            auto &y = static_cast<NNTensor<T> &>(Y);
            auto eps = Var<VT>{VT(sizeof(VT) == 2 ? 1e-3f : epsilon_)};
            auto spatial_f = Var<VT>{VT(spatial_size)};

            bool x_is_bb = x.container().is_byte_buffer();
            bool y_is_bb = y.container().is_byte_buffer();
            auto *buf_x = x_is_bb ? x.container().get_byte_buffer() : nullptr;
            auto *buf_y = y_is_bb ? y.container().get_byte_buffer() : nullptr;
            uint off_x = x_is_bb ? static_cast<uint>(x.container().get_byte_offset()) : 0u;
            uint off_y = y_is_bb ? static_cast<uint>(y.container().get_byte_offset()) : 0u;

            for (auto n : dynamic_range(N)) {
                for (auto c : dynamic_range(C)) {
                    auto base = n * x.strides()[0] + c * x.strides()[1];
                    // Two-pass mean and variance with vectorized reads
                    auto mean = def(VT{0});
                    if constexpr (std::is_same_v<VT, float> || std::is_same_v<VT, half>) {
                        constexpr uint32_t VecN = detail::VecDispatch<VT>::N;
                        auto vec_count = spatial_size / VecN;
                        auto rem = spatial_size % VecN;
                        auto vec_sum = def(make_zero_vec<VT>());
                        if (x_is_bb) {
                            for (auto i : dynamic_range(vec_count)) {
                                auto off = base + i * VecN;
                                auto v = read_vec4<VT>(x, off, buf_x, off_x);
                                vec_sum = vec_sum + v;
                            }
                        } else {
                            for (auto i : dynamic_range(vec_count)) {
                                auto off = base + i * VecN;
                                auto v = gather_vec4<VT>(x, off);
                                vec_sum = vec_sum + v;
                            }
                        }
                        mean += vec_sum.x + vec_sum.y + vec_sum.z + vec_sum.w;
                        auto start = vec_count * VecN;
                        for (auto i : dynamic_range(rem)) {
                            mean += x[base + start + i];
                        }
                    } else {
                        for (auto i : dynamic_range(spatial_size)) {
                            mean += x[base + i];
                        }
                    }
                    mean = mean / spatial_f;

                    auto m2 = def(VT{0});
                    if constexpr (std::is_same_v<VT, float> || std::is_same_v<VT, half>) {
                        constexpr uint32_t VecN = detail::VecDispatch<VT>::N;
                        auto vec_count = spatial_size / VecN;
                        auto rem = spatial_size % VecN;
                        auto mean_vec = broadcast_vec4<VT>(mean);
                        auto vec_var = def(make_zero_vec<VT>());
                        if (x_is_bb) {
                            for (auto i : dynamic_range(vec_count)) {
                                auto off = base + i * VecN;
                                auto v = read_vec4<VT>(x, off, buf_x, off_x);
                                auto diff = v - mean_vec;
                                vec_var = luisa::compute::fma(diff, diff, vec_var);
                            }
                        } else {
                            for (auto i : dynamic_range(vec_count)) {
                                auto off = base + i * VecN;
                                auto v = gather_vec4<VT>(x, off);
                                auto diff = v - mean_vec;
                                vec_var = luisa::compute::fma(diff, diff, vec_var);
                            }
                        }
                        m2 += vec_var.x + vec_var.y + vec_var.z + vec_var.w;
                        auto start = vec_count * VecN;
                        for (auto i : dynamic_range(rem)) {
                            auto diff = x[base + start + i] - mean;
                            m2 += diff * diff;
                        }
                    } else {
                        for (auto i : dynamic_range(spatial_size)) {
                            auto diff = x[base + i] - mean;
                            m2 += diff * diff;
                        }
                    }
                    auto var = m2 / spatial_f;
                    auto inv_std = scale[c] / sqrt(var + eps);
                    auto linear_bias = bias[c] - mean * inv_std;
                    auto base_out = n * y.strides()[0] + c * y.strides()[1];

                    if constexpr (std::is_same_v<VT, float> || std::is_same_v<VT, half>) {
                        constexpr uint32_t VecN = detail::VecDispatch<VT>::N;
                        auto vec_count = spatial_size / VecN;
                        auto rem = spatial_size % VecN;
                        auto inv_std_vec = broadcast_vec4<VT>(inv_std);
                        auto bias_vec = broadcast_vec4<VT>(linear_bias);

                        if (x_is_bb && y_is_bb) {
                            for (auto i : dynamic_range(vec_count)) {
                                auto off = i * VecN;
                                auto xv = read_vec4<VT>(x, base + off, buf_x, off_x);
                                auto yv = luisa::compute::fma(xv, inv_std_vec, bias_vec);
                                write_vec4<VT>(y, base_out + off, yv, buf_y, off_y);
                            }
                        } else {
                            for (auto i : dynamic_range(vec_count)) {
                                auto off_in = base + i * VecN;
                                auto off_out = base_out + i * VecN;
                                auto xv = gather_vec4<VT>(x, off_in);
                                auto yv = luisa::compute::fma(xv, inv_std_vec, bias_vec);
                                scatter_vec4<VT>(y, off_out, yv);
                            }
                        }
                        auto start = vec_count * VecN;
                        for (auto i : dynamic_range(rem)) {
                            y[base_out + start + i] = luisa::compute::fma(x[base + start + i], inv_std, linear_bias);
                        }
                    } else {
                        for (auto i : dynamic_range(spatial_size)) {
                            y[base_out + i] = luisa::compute::fma(x[base + i], inv_std, linear_bias);
                        }
                    }
                }
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(InstanceNormalization) {
    float epsilon = 1e-5f;
    if (auto p = node.try_get_attr("epsilon"))
        epsilon = p->get<onnx::AttributeType::FLOAT>();
    return std::make_unique<InstanceNormalization>(epsilon);
};

// LayerNormalization: normalizes over the last N dimensions.
// ONNX spec: inputs: X, Scale, [B]; attributes: axis (default -1), epsilon, stash_type
class LayerNormalization : public Operator {
private:
    int64_t axis_;
    float epsilon_;

public:
    LayerNormalization(int64_t axis, float epsilon)
        : Operator("LayerNormalization"), axis_(axis), epsilon_(epsilon) {}

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() >= 2 && outputs.size() >= 1,
                     "LayerNormalization requires >=2 inputs and >=1 output.");
#endif
        auto &X = inputs[0].get();
        auto &Y = outputs[0].get();

        auto const &x_shape = X.shape();
        auto ndim = x_shape.size();
        auto ndim_i = static_cast<int64_t>(ndim);
        auto axis = axis_ < 0 ? axis_ + ndim_i : axis_;
#ifndef NDEBUG
        LUISA_ASSERT(axis >= 0 && axis < ndim_i, "LayerNormalization axis out of range.");
#endif

        // Outer dims: product of shape[0..axis), Inner dims: product of shape[axis..end)
        uint32_t outer_size = 1, inner_size = 1;
        for (int64_t d = 0; d < axis; ++d) outer_size *= x_shape[d];
        for (size_t d = axis; d < ndim; ++d) inner_size *= x_shape[d];

        ITensor *B_ptr = (inputs.size() >= 3) ? &inputs[2].get() : nullptr;

        visit_typeid<NNFilteredTypeList<IsFloatingPoint>>(X.element_type(), [&]<typename T>() {
            using VT = nn_storage_type_t<T>;
            auto &x = static_cast<NNTensor<T> &>(X);
            auto &scale = static_cast<NNTensor<T> &>(inputs[1].get());
            auto &y = static_cast<NNTensor<T> &>(Y);
            auto eps = Var<VT>{VT(sizeof(VT) == 2 ? 1e-3f : epsilon_)};
            auto inner_f = Var<VT>{VT(inner_size)};

            bool x_is_bb = x.container().is_byte_buffer();
            bool y_is_bb = y.container().is_byte_buffer();
            bool scale_is_bb = scale.container().is_byte_buffer();
            bool b_is_bb = false;
            auto *buf_x = x_is_bb ? x.container().get_byte_buffer() : nullptr;
            auto *buf_y = y_is_bb ? y.container().get_byte_buffer() : nullptr;
            auto *buf_scale = scale_is_bb ? scale.container().get_byte_buffer() : nullptr;
            auto *buf_b = static_cast<luisa::compute::Var<luisa::compute::ByteBuffer> *>(nullptr);
            uint off_x = x_is_bb ? static_cast<uint>(x.container().get_byte_offset()) : 0u;
            uint off_y = y_is_bb ? static_cast<uint>(y.container().get_byte_offset()) : 0u;
            uint off_scale = scale_is_bb ? static_cast<uint>(scale.container().get_byte_offset()) : 0u;
            uint off_b = 0u;
            if (B_ptr) {
                auto &b = static_cast<NNTensor<T> &>(*B_ptr);
                b_is_bb = b.container().is_byte_buffer();
                if (b_is_bb) {
                    buf_b = b.container().get_byte_buffer();
                    off_b = static_cast<uint>(b.container().get_byte_offset());
                }
            }

            for (auto outer : dynamic_range(outer_size)) {
                auto base = outer * inner_size;
                // Two-pass mean and variance with vectorized reads
                auto mean = def(VT{0});
                if constexpr (std::is_same_v<VT, float> || std::is_same_v<VT, half>) {
                    constexpr uint32_t VecN = detail::VecDispatch<VT>::N;
                    auto vec_count = inner_size / VecN;
                    auto rem = inner_size % VecN;
                    auto vec_sum = def(make_zero_vec<VT>());
                    if (x_is_bb) {
                        for (auto i : dynamic_range(vec_count)) {
                            auto off = base + i * VecN;
                            auto v = read_vec4<VT>(x, off, buf_x, off_x);
                            vec_sum = vec_sum + v;
                        }
                    } else {
                        for (auto i : dynamic_range(vec_count)) {
                            auto off = base + i * VecN;
                            auto v = gather_vec4<VT>(x, off);
                            vec_sum = vec_sum + v;
                        }
                    }
                    mean += vec_sum.x + vec_sum.y + vec_sum.z + vec_sum.w;
                    auto start = vec_count * VecN;
                    for (auto i : dynamic_range(rem)) {
                        mean += x[base + start + i];
                    }
                } else {
                    for (auto i : dynamic_range(inner_size)) {
                        mean += x[base + i];
                    }
                }
                mean = mean / inner_f;

                auto m2 = def(VT{0});
                if constexpr (std::is_same_v<VT, float> || std::is_same_v<VT, half>) {
                    constexpr uint32_t VecN = detail::VecDispatch<VT>::N;
                    auto vec_count = inner_size / VecN;
                    auto rem = inner_size % VecN;
                    auto mean_vec = broadcast_vec4<VT>(mean);
                    auto vec_var = def(make_zero_vec<VT>());
                    if (x_is_bb) {
                        for (auto i : dynamic_range(vec_count)) {
                            auto off = base + i * VecN;
                            auto v = read_vec4<VT>(x, off, buf_x, off_x);
                            auto diff = v - mean_vec;
                            vec_var = luisa::compute::fma(diff, diff, vec_var);
                        }
                    } else {
                        for (auto i : dynamic_range(vec_count)) {
                            auto off = base + i * VecN;
                            auto v = gather_vec4<VT>(x, off);
                            auto diff = v - mean_vec;
                            vec_var = luisa::compute::fma(diff, diff, vec_var);
                        }
                    }
                    m2 += vec_var.x + vec_var.y + vec_var.z + vec_var.w;
                    auto start = vec_count * VecN;
                    for (auto i : dynamic_range(rem)) {
                        auto diff = x[base + start + i] - mean;
                        m2 += diff * diff;
                    }
                } else {
                    for (auto i : dynamic_range(inner_size)) {
                        auto diff = x[base + i] - mean;
                        m2 += diff * diff;
                    }
                }
                auto inv_std = Var<VT>{VT{1}} / sqrt(m2 / inner_f + eps);

                if constexpr (std::is_same_v<VT, float> || std::is_same_v<VT, half>) {
                    constexpr uint32_t VecN = detail::VecDispatch<VT>::N;
                    auto vec_count = inner_size / VecN;
                    auto rem = inner_size % VecN;
                    auto inv_std_vec = broadcast_vec4<VT>(inv_std);
                    auto mean_std_vec = broadcast_vec4<VT>(mean * inv_std);

                    bool use_bb_vec = x_is_bb && y_is_bb && scale_is_bb &&
                                      (!B_ptr || b_is_bb);

                    if (use_bb_vec) {
                        for (auto i : dynamic_range(vec_count)) {
                            auto off = base + i * VecN;
                            auto scale_off = i * VecN;
                            auto xv = read_vec4<VT>(x, off, buf_x, off_x);
                            auto sv = read_vec4<VT>(scale, scale_off, buf_scale, off_scale);
                            auto nv = luisa::compute::fma(xv, inv_std_vec, -mean_std_vec);
                            if (B_ptr) {
                                auto &b = static_cast<NNTensor<T> &>(*B_ptr);
                                auto bv = read_vec4<VT>(b, scale_off, buf_b, off_b);
                                auto result = luisa::compute::fma(nv, sv, bv);
                                write_vec4<VT>(y, off, result, buf_y, off_y);
                            } else {
                                auto result = luisa::compute::fma(nv, sv, make_zero_vec<VT>());
                                write_vec4<VT>(y, off, result, buf_y, off_y);
                            }
                        }
                    } else {
                        for (auto i : dynamic_range(vec_count)) {
                            auto off = base + i * VecN;
                            auto scale_off = i * VecN;
                            auto xv = gather_vec4<VT>(x, off);
                            auto sv = gather_vec4<VT>(scale, scale_off);
                            auto nv = luisa::compute::fma(xv, inv_std_vec, -mean_std_vec);
                            if (B_ptr) {
                                auto &b = static_cast<NNTensor<T> &>(*B_ptr);
                                auto bv = gather_vec4<VT>(b, scale_off);
                                auto result = luisa::compute::fma(nv, sv, bv);
                                scatter_vec4<VT>(y, off, result);
                            } else {
                                auto result = luisa::compute::fma(nv, sv, make_zero_vec<VT>());
                                scatter_vec4<VT>(y, off, result);
                            }
                        }
                    }
                    auto start = vec_count * VecN;
                    for (auto i : dynamic_range(rem)) {
                        auto idx = base + start + i;
                        auto normalized = luisa::compute::fma(x[idx], inv_std, -mean * inv_std);
                        auto result = luisa::compute::fma(normalized, scale[start + i], VT{0});
                        if (B_ptr) {
                            result = result + static_cast<NNTensor<T> &>(*B_ptr)[start + i];
                        }
                        y[idx] = result;
                    }
                } else {
                    for (auto i : dynamic_range(inner_size)) {
                        auto normalized = luisa::compute::fma(x[base + i], inv_std, -mean * inv_std);
                        auto result = luisa::compute::fma(normalized, scale[i], VT{0});
                        if (B_ptr) {
                            result = result + static_cast<NNTensor<T> &>(*B_ptr)[i];
                        }
                        y[base + i] = result;
                    }
                }
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(LayerNormalization) {
    int64_t axis = -1;
    float epsilon = 1e-5f;
    if (auto p = node.try_get_attr("axis"))
        axis = p->get<onnx::AttributeType::INT>();
    if (auto p = node.try_get_attr("epsilon"))
        epsilon = p->get<onnx::AttributeType::FLOAT>();
    return std::make_unique<LayerNormalization>(axis, epsilon);
};

// GroupNormalization: divides channels into groups, normalizes within each group.
// ONNX spec: inputs: X(N,C,D1,...), scale(C), bias(C); attribute: epsilon, num_groups
class GroupNormalization : public Operator {
private:
    float epsilon_;
    int64_t num_groups_;

public:
    GroupNormalization(float epsilon, int64_t num_groups)
        : Operator("GroupNormalization"), epsilon_(epsilon), num_groups_(num_groups) {}

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 3 && outputs.size() == 1,
                     "GroupNormalization requires 3 inputs and 1 output.");
#endif
        auto &X = inputs[0].get();
        auto &Y = outputs[0].get();

        auto const &x_shape = X.shape();
#ifndef NDEBUG
        LUISA_ASSERT(x_shape.size() >= 2, "GroupNormalization input must be at least 2D.");
#endif
        uint32_t N = x_shape[0];
        uint32_t C = x_shape[1];
        uint32_t spatial_size = 1;
        for (size_t i = 2; i < x_shape.size(); ++i)
            spatial_size *= x_shape[i];

        uint32_t num_groups = static_cast<uint32_t>(num_groups_);
#ifndef NDEBUG
        LUISA_ASSERT(num_groups > 0 && C % num_groups == 0,
                     "GroupNormalization: num_groups must be >0 and divide C.");
#endif
        uint32_t channels_per_group = C / num_groups;
        uint32_t group_size = channels_per_group * spatial_size;

        visit_typeid<NNFilteredTypeList<IsFloatingPoint>>(X.element_type(), [&]<typename T>() {
            using VT = nn_storage_type_t<T>;
            auto &x = static_cast<NNTensor<T> &>(X);
            auto &scale = static_cast<NNTensor<T> &>(inputs[1].get());
            auto &bias = static_cast<NNTensor<T> &>(inputs[2].get());
            auto &y = static_cast<NNTensor<T> &>(Y);
            auto eps = Var<VT>{VT(sizeof(VT) == 2 ? 1e-3f : epsilon_)};
            auto group_f = Var<VT>{VT(group_size)};

            bool x_is_bb = x.container().is_byte_buffer();
            bool y_is_bb = y.container().is_byte_buffer();
            auto *buf_x = x_is_bb ? x.container().get_byte_buffer() : nullptr;
            auto *buf_y = y_is_bb ? y.container().get_byte_buffer() : nullptr;
            uint off_x = x_is_bb ? static_cast<uint>(x.container().get_byte_offset()) : 0u;
            uint off_y = y_is_bb ? static_cast<uint>(y.container().get_byte_offset()) : 0u;

            for (auto n : dynamic_range(N)) {
                for (uint32_t g = 0; g < num_groups; ++g) {
                    // Compute mean over channels_per_group * spatial_size (flattened)
                    auto group_base = n * x.strides()[0] + g * channels_per_group * x.strides()[1];
                    auto sum = def(VT{0});
                    if constexpr (std::is_same_v<VT, float> || std::is_same_v<VT, half>) {
                        constexpr uint32_t VecN = detail::VecDispatch<VT>::N;
                        auto group_vec_count = group_size / VecN;
                        auto group_rem = group_size % VecN;
                        auto vec_sum = def(make_zero_vec<VT>());
                        if (x_is_bb) {
                            for (auto i : dynamic_range(group_vec_count)) {
                                auto off = group_base + i * VecN;
                                auto v = read_vec4<VT>(x, off, buf_x, off_x);
                                vec_sum = vec_sum + v;
                            }
                        } else {
                            for (auto i : dynamic_range(group_vec_count)) {
                                auto off = group_base + i * VecN;
                                auto v = gather_vec4<VT>(x, off);
                                vec_sum = vec_sum + v;
                            }
                        }
                        sum += vec_sum.x + vec_sum.y + vec_sum.z + vec_sum.w;
                        auto start = group_vec_count * VecN;
                        for (auto i : dynamic_range(group_rem)) {
                            sum += x[group_base + start + i];
                        }
                    } else {
                        for (auto i : dynamic_range(group_size)) {
                            sum += x[group_base + i];
                        }
                    }
                    auto mean = sum / group_f;
                    // Variance (flattened)
                    auto var_sum = def(VT{0});
                    if constexpr (std::is_same_v<VT, float> || std::is_same_v<VT, half>) {
                        constexpr uint32_t VecN = detail::VecDispatch<VT>::N;
                        auto group_vec_count = group_size / VecN;
                        auto group_rem = group_size % VecN;
                        auto vec_var = def(make_zero_vec<VT>());
                        auto mean_vec = broadcast_vec4<VT>(mean);
                        if (x_is_bb) {
                            for (auto i : dynamic_range(group_vec_count)) {
                                auto off = group_base + i * VecN;
                                auto v = read_vec4<VT>(x, off, buf_x, off_x);
                                auto diff = v - mean_vec;
                                vec_var = luisa::compute::fma(diff, diff, vec_var);
                            }
                        } else {
                            for (auto i : dynamic_range(group_vec_count)) {
                                auto off = group_base + i * VecN;
                                auto v = gather_vec4<VT>(x, off);
                                auto diff = v - mean_vec;
                                vec_var = luisa::compute::fma(diff, diff, vec_var);
                            }
                        }
                        var_sum += vec_var.x + vec_var.y + vec_var.z + vec_var.w;
                        auto start = group_vec_count * VecN;
                        for (auto i : dynamic_range(group_rem)) {
                            auto diff = x[group_base + start + i] - mean;
                            var_sum = luisa::compute::fma(diff, diff, var_sum);
                        }
                    } else {
                        for (auto i : dynamic_range(group_size)) {
                            auto diff = x[group_base + i] - mean;
                            var_sum += diff * diff;
                        }
                    }
                    auto inv_std = Var<VT>{VT{1}} / sqrt(var_sum / group_f + eps);
                    // Apply scale and bias per channel
                    for (uint32_t c_off = 0; c_off < channels_per_group; ++c_off) {
                        uint32_t c = g * channels_per_group + c_off;
                        auto base_in = n * x.strides()[0] + c * x.strides()[1];
                        auto base_out = n * y.strides()[0] + c * y.strides()[1];
                        // ONNX GroupNorm: scale and bias are per-channel (size C)
                        auto s = scale[c];
                        auto b = bias[c];
                        auto scaled_std = inv_std * s;
                        auto out_bias = b - mean * scaled_std;

                        if constexpr (std::is_same_v<VT, float> || std::is_same_v<VT, half>) {
                            constexpr uint32_t VecN = detail::VecDispatch<VT>::N;
                            auto vec_count = spatial_size / VecN;
                            auto rem = spatial_size % VecN;
                            auto scaled_std_vec = broadcast_vec4<VT>(scaled_std);
                            auto out_bias_vec = broadcast_vec4<VT>(out_bias);

                            if (x_is_bb && y_is_bb) {
                                for (auto i : dynamic_range(vec_count)) {
                                    auto off = i * VecN;
                                    auto xv = read_vec4<VT>(x, base_in + off, buf_x, off_x);
                                    auto yv = luisa::compute::fma(xv, scaled_std_vec, out_bias_vec);
                                    write_vec4<VT>(y, base_out + off, yv, buf_y, off_y);
                                }
                            } else {
                                for (auto i : dynamic_range(vec_count)) {
                                    auto off_in = base_in + i * VecN;
                                    auto off_out = base_out + i * VecN;
                                    auto xv = gather_vec4<VT>(x, off_in);
                                    auto yv = luisa::compute::fma(xv, scaled_std_vec, out_bias_vec);
                                    scatter_vec4<VT>(y, off_out, yv);
                                }
                            }
                            auto start = vec_count * VecN;
                            for (auto i : dynamic_range(rem)) {
                                y[base_out + start + i] = luisa::compute::fma(x[base_in + start + i], scaled_std, out_bias);
                            }
                        } else {
                            for (auto i : dynamic_range(spatial_size)) {
                                y[base_out + i] = luisa::compute::fma(x[base_in + i], scaled_std, out_bias);
                            }
                        }
                    }
                }
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(GroupNormalization) {
    float epsilon = 1e-5f;
    int64_t num_groups = 1;
    if (auto p = node.try_get_attr("epsilon"))
        epsilon = p->get<onnx::AttributeType::FLOAT>();
    if (auto p = node.try_get_attr("num_groups"))
        num_groups = p->get<onnx::AttributeType::INT>();
    return std::make_unique<GroupNormalization>(epsilon, num_groups);
};

}// namespace lcml::onnx
