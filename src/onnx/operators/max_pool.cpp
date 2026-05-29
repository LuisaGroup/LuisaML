#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

namespace lcml::onnx {

template<typename T>
struct MaxPoolSupported : std::bool_constant<
    (IsFloatingPoint<T>::value && !std::is_same_v<T, double>) ||
    std::is_same_v<T, FP4E2M1> ||
    std::is_same_v<T, FP8E4M3FN> ||
    std::is_same_v<T, FP8E5M2> ||
    std::is_same_v<T, FP16Quantized>> {};

// MaxPool: computes max pooling over spatial dimensions.
// ONNX spec: attributes: auto_pad, ceil_mode, dilations, kernel_shape, pads, strides, storage_order
class MaxPool : public Operator {
private:
    std::vector<int> kernel_shape_;
    std::vector<int> pads_;
    std::vector<int> strides_;
    std::vector<int> dilations_;

public:
    MaxPool(std::vector<int> kernel_shape, std::vector<int> pads,
            std::vector<int> strides, std::vector<int> dilations)
        : Operator("MaxPool"), kernel_shape_(std::move(kernel_shape)),
          pads_(std::move(pads)), strides_(std::move(strides)),
          dilations_(std::move(dilations)) {}

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 1 && outputs.size() == 1,
                     "MaxPool requires 1 input and 1 output.");
        LUISA_ASSERT(kernel_shape_.size() == 2,
                     "MaxPool: only 2D kernel_shape is supported.");
        auto &X = inputs[0].get();
        auto &Y = outputs[0].get();

        auto const &x_shape = X.shape();// (N, C, H, W)
        auto const &y_shape = Y.shape();
        auto spatial_dims = x_shape.size() - 2;

        std::vector<int> strides = strides_;
        std::vector<int> pads = pads_;
        std::vector<int> dilations = dilations_;
        if (strides.empty()) strides.assign(spatial_dims, 1);
        if (pads.empty()) pads.assign(spatial_dims * 2, 0);
        if (dilations.empty()) dilations.assign(spatial_dims, 1);

        LUISA_ASSERT(spatial_dims == 2, "MaxPool: only 2D pooling is supported.");
#else
        auto &X = inputs[0].get();
        auto &Y = outputs[0].get();

        auto const &x_shape = X.shape();// (N, C, H, W)
        auto const &y_shape = Y.shape();
        auto spatial_dims = x_shape.size() - 2;

        std::vector<int> strides = strides_;
        std::vector<int> pads = pads_;
        std::vector<int> dilations = dilations_;
        if (strides.empty()) strides.assign(spatial_dims, 1);
        if (pads.empty()) pads.assign(spatial_dims * 2, 0);
        if (dilations.empty()) dilations.assign(spatial_dims, 1);
#endif

        visit_typeid<NNFilteredTypeList<MaxPoolSupported>>(X.element_type(), [&]<typename T>() {
            using VT = nn_storage_type_t<T>;
            using CT = std::conditional_t<
                std::is_same_v<T, FP4E2M1> || std::is_same_v<T, FP8E4M3FN> ||
                std::is_same_v<T, FP8E5M2> || std::is_same_v<T, FP16Quantized>,
                half, VT>;
            auto &x = static_cast<NNTensor<T> &>(X);
            auto &y = static_cast<NNTensor<T> &>(Y);

            uint32_t N = x_shape[0], C = x_shape[1];
            uint32_t iH = x_shape[2], iW = x_shape[3];
            uint32_t kH = kernel_shape_[0], kW = kernel_shape_[1];
            uint32_t oH = y_shape[2], oW = y_shape[3];
            int sH = strides[0], sW = strides[1];
            int dH = dilations[0], dW = dilations[1];
            int pH = pads[0], pW = pads[1];

            auto x_stride_n = x.strides()[0];
            auto x_stride_c = x.strides()[1];
            auto x_stride_h = x.strides()[2];
            auto x_stride_w = x.strides()[3];
            auto y_stride_n = y.strides()[0];
            auto y_stride_c = y.strides()[1];
            auto y_stride_h = y.strides()[2];
            auto y_stride_w = y.strides()[3];

            auto neg_inf = [&]() -> CT {
                if constexpr (std::is_same_v<CT, half>) {
                    return CT(-65504.0f);
                } else {
                    return CT(-std::numeric_limits<CT>::infinity());
                }
            }();

            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
                bool x_is_bb = x.container().is_byte_buffer();
                Var<ByteBuffer> *buf_x = nullptr;
                uint off_x = 0;
                if (x_is_bb) {
                    buf_x = x.container().get_byte_buffer();
                    off_x = static_cast<uint>(x.container().get_byte_offset());
                }
                auto read_x = [&](auto idx) -> Var<CT> {
                    if (x_is_bb) {
                        return buf_x->read<CT>(off_x + idx * static_cast<uint>(sizeof(CT)));
                    }
                    return x[idx];
                };

                // Native float/half path with optional kw-loop vectorization when dW==1
                if (dW == 1) {
                    for (auto n : dynamic_range(N)) {
                        for (auto c : dynamic_range(C)) {
                            auto base_x = n * x_stride_n + c * x_stride_c;
                            auto base_y = n * y_stride_n + c * y_stride_c;
                            for (auto oh : dynamic_range(oH)) {
                                for (auto ow : dynamic_range(oW)) {
                                    auto max_val = def(neg_inf);
                                    for (uint32_t kh = 0; kh < kH; ++kh) {
                                        auto ih = oh * sH + kh * dH - pH;
                                        auto iw0 = ow * sW - pW;
                                        auto iw_last = iw0 + (kW - 1u);
                                        $if (ih >= 0u & ih < iH & iw0 >= 0u & iw_last < iW) {
                                            auto row_base = base_x + ih * x_stride_h;
                                            if constexpr (std::is_same_v<VT, float>) {
                                                auto vec_max = def(make_float4(-std::numeric_limits<float>::infinity()));
                                                uint32_t kw_vec_end = (kW / 4u) * 4u;
                                                if (x_is_bb && x_stride_w == 1) {
                                                    for (uint32_t kw = 0; kw < kw_vec_end; kw += 4u) {
                                                        auto byte_off = off_x + (row_base + iw0 + kw) * static_cast<uint>(sizeof(float));
                                                        auto v = buf_x->read<float4>(byte_off);
                                                        vec_max = max(vec_max, v);
                                                    }
                                                    for (uint32_t kw = kw_vec_end; kw < kW; ++kw) {
                                                        auto byte_off = off_x + (row_base + iw0 + kw) * static_cast<uint>(sizeof(float));
                                                        max_val = max(max_val, buf_x->read<float>(byte_off));
                                                    }
                                                } else {
                                                    for (uint32_t kw = 0; kw < kw_vec_end; kw += 4u) {
                                                        auto v = make_float4(
                                                            x[row_base + (iw0 + kw + 0u) * x_stride_w],
                                                            x[row_base + (iw0 + kw + 1u) * x_stride_w],
                                                            x[row_base + (iw0 + kw + 2u) * x_stride_w],
                                                            x[row_base + (iw0 + kw + 3u) * x_stride_w]);
                                                        vec_max = max(vec_max, v);
                                                    }
                                                    for (uint32_t kw = kw_vec_end; kw < kW; ++kw) {
                                                        max_val = max(max_val, x[row_base + (iw0 + kw) * x_stride_w]);
                                                    }
                                                }
                                                max_val = max(max_val, max(max(vec_max.x, vec_max.y), max(vec_max.z, vec_max.w)));
                                            } else {
                                                auto vec_max = def(make_half4(half(-65504.0f)));
                                                uint32_t kw_vec_end = (kW / 4u) * 4u;
                                                if (x_is_bb && x_stride_w == 1) {
                                                    for (uint32_t kw = 0; kw < kw_vec_end; kw += 4u) {
                                                        auto byte_off = off_x + (row_base + iw0 + kw) * static_cast<uint>(sizeof(half));
                                                        auto v = buf_x->read<half4>(byte_off);
                                                        vec_max = max(vec_max, v);
                                                    }
                                                    for (uint32_t kw = kw_vec_end; kw < kW; ++kw) {
                                                        auto byte_off = off_x + (row_base + iw0 + kw) * static_cast<uint>(sizeof(half));
                                                        max_val = max(max_val, buf_x->read<half>(byte_off));
                                                    }
                                                } else {
                                                    for (uint32_t kw = 0; kw < kw_vec_end; kw += 4u) {
                                                        auto v = make_half4(
                                                            x[row_base + (iw0 + kw + 0u) * x_stride_w],
                                                            x[row_base + (iw0 + kw + 1u) * x_stride_w],
                                                            x[row_base + (iw0 + kw + 2u) * x_stride_w],
                                                            x[row_base + (iw0 + kw + 3u) * x_stride_w]);
                                                        vec_max = max(vec_max, v);
                                                    }
                                                    for (uint32_t kw = kw_vec_end; kw < kW; ++kw) {
                                                        max_val = max(max_val, x[row_base + (iw0 + kw) * x_stride_w]);
                                                    }
                                                }
                                                max_val = max(max_val, max(max(vec_max.x, vec_max.y), max(vec_max.z, vec_max.w)));
                                            }
                                        } $else {
                                            for (uint32_t kw = 0; kw < kW; ++kw) {
                                                auto iw = ow * sW + kw - pW;
                                                $if (ih >= 0u & ih < iH & iw >= 0u & iw < iW) {
                                                    max_val = max(max_val, read_x(base_x + ih * x_stride_h + iw * x_stride_w));
                                                };
                                            }
                                        };
                                    }
                                    y[base_y + oh * y_stride_h + ow * y_stride_w] = max_val;
                                }
                            }
                        }
                    }
                } else {
                    for (auto n : dynamic_range(N)) {
                        for (auto c : dynamic_range(C)) {
                            auto base_x = n * x_stride_n + c * x_stride_c;
                            auto base_y = n * y_stride_n + c * y_stride_c;
                            for (auto oh : dynamic_range(oH)) {
                                for (auto ow : dynamic_range(oW)) {
                                    auto max_val = def(neg_inf);
                                    for (uint32_t kh = 0; kh < kH; ++kh) {
                                        for (uint32_t kw = 0; kw < kW; ++kw) {
                                            auto ih = oh * sH + kh * dH - pH;
                                            auto iw = ow * sW + kw * dW - pW;
                                            $if (ih >= 0u & ih < iH & iw >= 0u & iw < iW) {
                                                max_val = max(max_val, read_x(base_x + ih * x_stride_h + iw * x_stride_w));
                                            };
                                        }
                                    }
                                    y[base_y + oh * y_stride_h + ow * y_stride_w] = max_val;
                                }
                            }
                        }
                    }
                }
            } else {
                // Quantized path: dequantize to half, compute max, quantize back
                auto deq = [&](auto &v) -> Var<half> {
                    if constexpr (std::is_same_v<T, FP4E2M1>) {
                        static auto c = fp4e2m1_to_float();
                        return c(v.bits.cast<ushort>());
                    } else if constexpr (std::is_same_v<T, FP8E4M3FN>) {
                        static auto c = fp8e4m3_to_float();
                        return c(v.bits.cast<ushort>());
                    } else if constexpr (std::is_same_v<T, FP8E5M2>) {
                        static auto c = fp8e5m2_to_float();
                        return c(v.bits.cast<ushort>());
                    } else {
                        return v.bits;
                    }
                };

                auto store_y = [&](auto idx, auto val) {
                    if constexpr (std::is_same_v<T, FP4E2M1>) {
                        static auto c = fp4e2m1_from_float();
                        y[idx].bits = c(val.template cast<half>()).cast<uint16_t>();
                    } else if constexpr (std::is_same_v<T, FP8E4M3FN>) {
                        static auto c = fp8e4m3_from_float();
                        y[idx].bits = c(val.template cast<half>()).cast<uint16_t>();
                    } else if constexpr (std::is_same_v<T, FP8E5M2>) {
                        static auto c = fp8e5m2_from_float();
                        y[idx].bits = c(val.template cast<half>()).cast<uint16_t>();
                    } else {
                        y[idx].bits = val.template cast<half>();
                    }
                };

                for (auto n : dynamic_range(N)) {
                    for (auto c : dynamic_range(C)) {
                        auto base_x = n * x_stride_n + c * x_stride_c;
                        auto base_y = n * y_stride_n + c * y_stride_c;
                        for (auto oh : dynamic_range(oH)) {
                            for (auto ow : dynamic_range(oW)) {
                                auto max_val = def(neg_inf);
                                for (uint32_t kh = 0; kh < kH; ++kh) {
                                    for (uint32_t kw = 0; kw < kW; ++kw) {
                                        auto ih = oh * sH + kh * dH - pH;
                                        auto iw = ow * sW + kw * dW - pW;
                                        $if (ih >= 0u & ih < iH & iw >= 0u & iw < iW) {
                                            max_val = max(max_val, deq(x[base_x + ih * x_stride_h + iw * x_stride_w]));
                                        };
                                    }
                                }
                                store_y(base_y + oh * y_stride_h + ow * y_stride_w, max_val);
                            }
                        }
                    }
                }
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(MaxPool) {
    std::vector<int> kernel_shape, pads, strides, dilations;
    if (auto p = node.try_get_attr("kernel_shape"))
        kernel_shape = p->get<onnx::AttributeType::INTS>();
    if (auto p = node.try_get_attr("pads"))
        pads = p->get<onnx::AttributeType::INTS>();
    if (auto p = node.try_get_attr("strides"))
        strides = p->get<onnx::AttributeType::INTS>();
    if (auto p = node.try_get_attr("dilations"))
        dilations = p->get<onnx::AttributeType::INTS>();
    return std::make_unique<MaxPool>(std::move(kernel_shape), std::move(pads),
                                     std::move(strides), std::move(dilations));
};

}// namespace lcml::onnx
