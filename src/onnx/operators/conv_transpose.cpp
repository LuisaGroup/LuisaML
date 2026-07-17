#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/memory.h>

namespace lcml::onnx {

template<typename T>
struct ConvTransposeSupported : std::bool_constant<
    IsFloatingPoint<T>::value ||
    std::is_same_v<T, FP4E2M1> ||
    std::is_same_v<T, FP8E4M3FN> ||
    std::is_same_v<T, FP8E5M2> ||
    std::is_same_v<T, FP16Quantized>> {};

class ConvTranspose : public Operator {
private:
    luisa::string auto_pad_;
    luisa::vector<int32_t> dilations_;
    int32_t group_;
    luisa::vector<int32_t> kernel_shape_;
    luisa::vector<int32_t> output_padding_;
    luisa::vector<int32_t> pads_;
    luisa::vector<int32_t> strides_;

public:
    ConvTranspose(luisa::string auto_pad, luisa::vector<int32_t> dilations, int32_t group,
                  luisa::vector<int32_t> kernel_shape, luisa::vector<int32_t> output_padding,
                  luisa::vector<int32_t> pads, luisa::vector<int32_t> strides)
        : Operator("ConvTranspose"), auto_pad_(std::move(auto_pad)),
          dilations_(std::move(dilations)), group_(group),
          kernel_shape_(std::move(kernel_shape)),
          output_padding_(std::move(output_padding)),
          pads_(std::move(pads)), strides_(std::move(strides)) {}

    void forward(luisa::span<std::reference_wrapper<ITensor>> inputs,
                 luisa::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() >= 2 && inputs.size() <= 3 && outputs.size() == 1,
                     "ConvTranspose requires 2-3 inputs and 1 output.");
#endif
        auto &X = inputs[0].get();
        auto &W = inputs[1].get();
        ITensor *B_ptr = (inputs.size() == 3) ? &inputs[2].get() : nullptr;
        auto &Y = outputs[0].get();

        auto const &x_shape = X.shape();// (N, C, iH, iW)
        auto const &w_shape = W.shape();// (C, M/group, kH, kW)
        auto const &y_shape = Y.shape();// (N, M, oH, oW)
        auto spatial_dims = x_shape.size() - 2;

#ifndef NDEBUG
        LUISA_ASSERT(spatial_dims == 2, "ConvTranspose: only 2D transposed convolution is supported.");
#endif

        luisa::vector<int32_t> dilations = dilations_;
        luisa::vector<int32_t> strides = strides_;
        luisa::vector<int32_t> pads = pads_;
        if (dilations.empty()) dilations.assign(spatial_dims, 1);
        if (strides.empty()) strides.assign(spatial_dims, 1);
        if (pads.empty()) pads.assign(spatial_dims * 2, 0);

        uint32_t N = x_shape[0];
        uint32_t C = x_shape[1];
        uint32_t M = y_shape[1];
        uint32_t group = static_cast<uint32_t>(group_);
        uint32_t C_per_group = C / group;
        uint32_t M_per_group = M / group;

        visit_type_index<NNFilteredTypeList<ConvTransposeSupported>>(X.element_type_index(), [&]<typename T>() {
            using VT = nn_storage_type_t<T>;
            using CT = std::conditional_t<
                std::is_same_v<T, FP4E2M1> || std::is_same_v<T, FP8E4M3FN> ||
                std::is_same_v<T, FP8E5M2> || std::is_same_v<T, FP16Quantized>,
                half, VT>;

            auto &x = static_cast<NNTensor<T> &>(X);
            auto &w = static_cast<NNTensor<T> &>(W);
            auto &y = static_cast<NNTensor<T> &>(Y);

            auto dequant = [&](auto &v) -> Var<CT> {
                if constexpr (std::is_same_v<T, FP4E2M1>) {
                    static auto c = fp4e2m1_to_float();
                    return c(v.bits.cast<ushort>());
                } else if constexpr (std::is_same_v<T, FP8E4M3FN>) {
                    static auto c = fp8e4m3_to_float();
                    return c(v.bits.cast<ushort>());
                } else if constexpr (std::is_same_v<T, FP8E5M2>) {
                    static auto c = fp8e5m2_to_float();
                    return c(v.bits.cast<ushort>());
                } else if constexpr (std::is_same_v<T, FP16Quantized>) {
                    return v.bits;
                } else {
                    return v;
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
                } else if constexpr (std::is_same_v<T, FP16Quantized>) {
                    y[idx].bits = val.template cast<half>();
                } else {
                    if (y.container().is_byte_buffer()) {
                        auto buf_y = y.container().get_byte_buffer();
                        auto off_y = static_cast<uint>(y.container().get_byte_offset());
                        auto byte_addr = off_y + idx * static_cast<uint>(sizeof(CT));
                        buf_y->write(byte_addr, val);
                    } else {
                        y[idx] = val;
                    }
                }
            };

            uint32_t iH = x_shape[2], iW = x_shape[3];
            uint32_t kH = w_shape[2], kW = w_shape[3];
            uint32_t oH = y_shape[2], oW = y_shape[3];
            int32_t sH = strides[0], sW = strides[1];
            int32_t dH = dilations[0], dW = dilations[1];
            int32_t pH = pads[0], pW = pads[1];

            auto x_stride_n = x.strides()[0];
            auto x_stride_c = x.strides()[1];
            auto x_stride_h = x.strides()[2];
            auto x_stride_w = x.strides()[3];
            auto w_stride_c = w.strides()[0];
            auto w_stride_m = w.strides()[1];
            auto w_stride_h = w.strides()[2];
            auto w_stride_w = w.strides()[3];
            auto y_stride_n = y.strides()[0];
            auto y_stride_m = y.strides()[1];
            auto y_stride_h = y.strides()[2];
            auto y_stride_w = y.strides()[3];

            // ByteBuffer optimization state for native float/half only.
            // Quantized types keep the original DynamicArray scalar path.
            bool x_is_bb = false;
            bool w_is_bb = false;
            bool use_vec_bb = false;
            Var<ByteBuffer> *buf_x = nullptr;
            Var<ByteBuffer> *buf_w = nullptr;
            uint off_x = 0;
            uint off_w = 0;

            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
                x_is_bb = x.container().is_byte_buffer();
                w_is_bb = w.container().is_byte_buffer();
                use_vec_bb = x_is_bb && w_is_bb && x_stride_c == 1 && w_stride_c == 1;
                if (x_is_bb) {
                    buf_x = x.container().get_byte_buffer();
                    off_x = static_cast<uint>(x.container().get_byte_offset());
                }
                if (w_is_bb) {
                    buf_w = w.container().get_byte_buffer();
                    off_w = static_cast<uint>(w.container().get_byte_offset());
                }
            }

            auto read_x = [&](auto idx) -> Var<CT> {
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
                    if (x_is_bb) {
                        return buf_x->read<CT>(off_x + idx * static_cast<uint>(sizeof(CT)));
                    }
                }
                return dequant(x[idx]);
            };

            auto read_w = [&](auto idx) -> Var<CT> {
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
                    if (w_is_bb) {
                        return buf_w->read<CT>(off_w + idx * static_cast<uint>(sizeof(CT)));
                    }
                }
                return dequant(w[idx]);
            };

            // Gather: for each output pixel, accumulate from input
            for (auto n : dynamic_range(N)) {
                for (auto m : dynamic_range(M)) {
                    auto g = m / M_per_group;
                    auto m_in_g = m % M_per_group;
                    for (auto oh : dynamic_range(oH)) {
                        for (auto ow : dynamic_range(oW)) {
                            auto sum = def(CT{0});
                            for (uint32_t kh = 0; kh < kH; ++kh) {
                                for (uint32_t kw = 0; kw < kW; ++kw) {
                                    auto ih_s = oh + pH - kh * dH;
                                    auto iw_s = ow + pW - kw * dW;
                                    $if (ih_s % sH == 0 & iw_s % sW == 0) {
                                        auto ih = ih_s / sH;
                                        auto iw = iw_s / sW;
                                        $if (ih >= 0u & ih < iH & iw >= 0u & iw < iW) {
                                            auto x_base = n * x_stride_n + ih * x_stride_h + iw * x_stride_w;
                                            auto w_base = m_in_g * w_stride_m + kh * w_stride_h + kw * w_stride_w;

                                            uint32_t c_vec_end = (C_per_group / 4u) * 4u;
                                            if (c_vec_end > 0) {
                                                if constexpr (std::is_same_v<CT, float>) {
                                                    auto sum_vec = def(make_float4(0.0f));
                                                    if (use_vec_bb) {
                                                        for (uint32_t c = 0; c < c_vec_end; c += 4u) {
                                                            auto x_idx = x_base + g * C_per_group + c;
                                                            auto w_idx = g * C_per_group + c + w_base;
                                                            auto xv = buf_x->read<float4>(off_x + x_idx * static_cast<uint>(sizeof(float)));
                                                            auto wv = buf_w->read<float4>(off_w + w_idx * static_cast<uint>(sizeof(float)));
                                                            sum_vec = luisa::compute::fma(xv, wv, sum_vec);
                                                        }
                                                    } else {
                                                        for (uint32_t c = 0; c < c_vec_end; c += 4u) {
                                                            auto xv = make_float4(
                                                                read_x(x_base + (g * C_per_group + c + 0u) * x_stride_c),
                                                                read_x(x_base + (g * C_per_group + c + 1u) * x_stride_c),
                                                                read_x(x_base + (g * C_per_group + c + 2u) * x_stride_c),
                                                                read_x(x_base + (g * C_per_group + c + 3u) * x_stride_c));
                                                            auto wv = make_float4(
                                                                read_w((g * C_per_group + c + 0u) * w_stride_c + w_base),
                                                                read_w((g * C_per_group + c + 1u) * w_stride_c + w_base),
                                                                read_w((g * C_per_group + c + 2u) * w_stride_c + w_base),
                                                                read_w((g * C_per_group + c + 3u) * w_stride_c + w_base));
                                                            sum_vec = luisa::compute::fma(xv, wv, sum_vec);
                                                        }
                                                    }
                                                    sum += sum_vec.x + sum_vec.y + sum_vec.z + sum_vec.w;
                                                } else if constexpr (std::is_same_v<CT, half>) {
                                                    auto sum_vec = def(make_half4(half(0.0f)));
                                                    if (use_vec_bb) {
                                                        for (uint32_t c = 0; c < c_vec_end; c += 4u) {
                                                            auto x_idx = x_base + g * C_per_group + c;
                                                            auto w_idx = g * C_per_group + c + w_base;
                                                            auto xv = buf_x->read<half4>(off_x + x_idx * static_cast<uint>(sizeof(half)));
                                                            auto wv = buf_w->read<half4>(off_w + w_idx * static_cast<uint>(sizeof(half)));
                                                            sum_vec = luisa::compute::fma(xv, wv, sum_vec);
                                                        }
                                                    } else {
                                                        for (uint32_t c = 0; c < c_vec_end; c += 4u) {
                                                            auto xv = make_half4(
                                                                read_x(x_base + (g * C_per_group + c + 0u) * x_stride_c),
                                                                read_x(x_base + (g * C_per_group + c + 1u) * x_stride_c),
                                                                read_x(x_base + (g * C_per_group + c + 2u) * x_stride_c),
                                                                read_x(x_base + (g * C_per_group + c + 3u) * x_stride_c));
                                                            auto wv = make_half4(
                                                                read_w((g * C_per_group + c + 0u) * w_stride_c + w_base),
                                                                read_w((g * C_per_group + c + 1u) * w_stride_c + w_base),
                                                                read_w((g * C_per_group + c + 2u) * w_stride_c + w_base),
                                                                read_w((g * C_per_group + c + 3u) * w_stride_c + w_base));
                                                            sum_vec = luisa::compute::fma(xv, wv, sum_vec);
                                                        }
                                                    }
                                                    sum += sum_vec.x + sum_vec.y + sum_vec.z + sum_vec.w;
                                                }
                                            }
                                            for (uint32_t c = c_vec_end; c < C_per_group; ++c) {
                                                auto x_idx = x_base + (g * C_per_group + c) * x_stride_c;
                                                auto w_idx = (g * C_per_group + c) * w_stride_c + w_base;
                                                sum = luisa::compute::fma(read_x(x_idx), read_w(w_idx), sum);
                                            }
                                        };
                                    };
                                }
                            }
                            if (B_ptr) {
                                auto &b = static_cast<NNTensor<T> &>(*B_ptr);
                                sum += dequant(b[m]);
                            }
                            auto y_idx = n * y_stride_n + m * y_stride_m +
                                         oh * y_stride_h + ow * y_stride_w;
                            store_y(y_idx, sum);
                        }
                    }
                }
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(ConvTranspose) {
    luisa::string auto_pad = "NOTSET";
    luisa::vector<int32_t> dilations, kernel_shape, output_padding, pads, strides;
    int32_t group = 1;
    if (auto p = node.try_get_attr("auto_pad"))
        auto_pad = p->get<onnx::AttributeType::STRING>();
    if (auto p = node.try_get_attr("dilations"))
        dilations = p->get<onnx::AttributeType::INTS>();
    if (auto p = node.try_get_attr("group"))
        group = p->get<onnx::AttributeType::INT>();
    if (auto p = node.try_get_attr("kernel_shape"))
        kernel_shape = p->get<onnx::AttributeType::INTS>();
    if (auto p = node.try_get_attr("output_padding"))
        output_padding = p->get<onnx::AttributeType::INTS>();
    if (auto p = node.try_get_attr("pads"))
        pads = p->get<onnx::AttributeType::INTS>();
    if (auto p = node.try_get_attr("strides"))
        strides = p->get<onnx::AttributeType::INTS>();
    return luisa::make_unique<ConvTranspose>(std::move(auto_pad), std::move(dilations), group,
                                           std::move(kernel_shape), std::move(output_padding),
                                           std::move(pads), std::move(strides));
};

}// namespace lcml::onnx