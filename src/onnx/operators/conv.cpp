#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

namespace lcml::onnx {

template<typename T>
struct ConvSupported : std::bool_constant<
    IsFloatingPoint<T>::value ||
    std::is_same_v<T, FP4E2M1> ||
    std::is_same_v<T, FP8E4M3FN> ||
    std::is_same_v<T, FP8E5M2> ||
    std::is_same_v<T, FP16Quantized>> {};

class Conv : public Operator {
private:
    std::string auto_pad_;
    std::vector<int> dilations_;
    int group_;
    std::vector<int> kernel_shape_;
    std::vector<int> pads_;
    std::vector<int> strides_;

public:
    Conv(std::string auto_pad, std::vector<int> dilations, int group,
         std::vector<int> kernel_shape, std::vector<int> pads, std::vector<int> strides)
        : Operator("Conv"), auto_pad_(std::move(auto_pad)),
          dilations_(std::move(dilations)), group_(group),
          kernel_shape_(std::move(kernel_shape)), pads_(std::move(pads)),
          strides_(std::move(strides)) {}

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() >= 2 && inputs.size() <= 3 && outputs.size() == 1,
                     "Conv requires 2-3 inputs and 1 output.");
#endif
        auto &X = inputs[0].get();
        auto &W = inputs[1].get();
        ITensor *B_ptr = (inputs.size() == 3) ? &inputs[2].get() : nullptr;
        auto &Y = outputs[0].get();

        auto const &x_shape = X.shape();// (N, C, D1, ..., Dn)
        auto const &w_shape = W.shape();// (M, C/group, kD1, ..., kDn)
        auto const &y_shape = Y.shape();// (N, M, oD1, ..., oDn)
        auto spatial_dims = x_shape.size() - 2;

        // Set defaults
        std::vector<int> dilations = dilations_;
        std::vector<int> strides = strides_;
        std::vector<int> pads = pads_;
        if (dilations.empty()) dilations.assign(spatial_dims, 1);
        if (strides.empty()) strides.assign(spatial_dims, 1);
        if (pads.empty()) pads.assign(spatial_dims * 2, 0);

        uint32_t N = x_shape[0];
        uint32_t C = x_shape[1];
        uint32_t M = w_shape[0];
        uint32_t C_per_group = w_shape[1];
        uint32_t group = static_cast<uint32_t>(group_);

        visit_typeid<NNFilteredTypeList<ConvSupported>>(X.element_type(), [&]<typename T>() {
            using VT = nn_storage_type_t<T>;
            using CT = std::conditional_t<
                std::is_same_v<T, FP4E2M1> || std::is_same_v<T, FP8E4M3FN> ||
                std::is_same_v<T, FP8E5M2> || std::is_same_v<T, FP16Quantized>,
                half, VT>;

            auto &x = static_cast<NNTensor<T> &>(X);
            auto &w = static_cast<NNTensor<T> &>(W);
            auto &y = static_cast<NNTensor<T> &>(Y);

#ifndef NDEBUG
            LUISA_ASSERT(spatial_dims == 1 || spatial_dims == 2, "Conv: only 1D and 2D convolution are supported.");
#endif

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
                    y[idx] = val;
                }
            };

            // For 2D conv (most common case): X=(N,C,H,W), W=(M,C/g,kH,kW), Y=(N,M,oH,oW)
            if (spatial_dims == 2) {
                uint32_t iH = x_shape[2], iW = x_shape[3];
                uint32_t kH = w_shape[2], kW = w_shape[3];
                uint32_t oH = y_shape[2], oW = y_shape[3];
                int sH = strides[0], sW = strides[1];
                int dH = dilations[0], dW = dilations[1];
                int pH = pads[0], pW = pads[1];// begin pads
                uint32_t M_per_group = M / group;

                auto x_stride_n = x.strides()[0];
                auto x_stride_c = x.strides()[1];
                auto x_stride_h = x.strides()[2];
                auto x_stride_w = x.strides()[3];
                auto w_stride_m = w.strides()[0];
                auto w_stride_c = w.strides()[1];
                auto w_stride_h = w.strides()[2];
                auto w_stride_w = w.strides()[3];
                auto y_stride_n = y.strides()[0];
                auto y_stride_m = y.strides()[1];
                auto y_stride_h = y.strides()[2];
                auto y_stride_w = y.strides()[3];

                for (auto n : dynamic_range(N)) {
                    for (auto m : dynamic_range(M)) {
                        auto g = m / M_per_group;
                        for (auto oh : dynamic_range(oH)) {
                            for (auto ow : dynamic_range(oW)) {
                                auto sum = def(CT{0});
                                for (uint32_t kh = 0; kh < kH; ++kh) {
                                    for (uint32_t kw = 0; kw < kW; ++kw) {
                                        auto ih = oh * sH + kh * dH - pH;
                                        auto iw = ow * sW + kw * dW - pW;
                                        // Bounds check
                                        $if (ih >= 0u & ih < iH & iw >= 0u & iw < iW) {
                                            auto x_base = n * x_stride_n + ih * x_stride_h + iw * x_stride_w;
                                            auto w_base = m * w_stride_m + kh * w_stride_h + kw * w_stride_w;

                                            uint32_t c_vec_end = (C_per_group / 4u) * 4u;
                                            if (c_vec_end > 0) {
                                                if constexpr (std::is_same_v<CT, float>) {
                                                    auto sum_vec = def(make_float4(0.0f));
                                                    for (uint32_t c = 0; c < c_vec_end; c += 4u) {
                                                        auto xv = make_float4(
                                                            dequant(x[x_base + (g * C_per_group + c + 0u) * x_stride_c]),
                                                            dequant(x[x_base + (g * C_per_group + c + 1u) * x_stride_c]),
                                                            dequant(x[x_base + (g * C_per_group + c + 2u) * x_stride_c]),
                                                            dequant(x[x_base + (g * C_per_group + c + 3u) * x_stride_c]));
                                                        auto wv = make_float4(
                                                            dequant(w[w_base + (c + 0u) * w_stride_c]),
                                                            dequant(w[w_base + (c + 1u) * w_stride_c]),
                                                            dequant(w[w_base + (c + 2u) * w_stride_c]),
                                                            dequant(w[w_base + (c + 3u) * w_stride_c]));
                                                        sum_vec = luisa::compute::fma(xv, wv, sum_vec);
                                                    }
                                                    sum += sum_vec.x + sum_vec.y + sum_vec.z + sum_vec.w;
                                                } else if constexpr (std::is_same_v<CT, half>) {
                                                    auto sum_vec = def(make_half4(half(0.0f)));
                                                    for (uint32_t c = 0; c < c_vec_end; c += 4u) {
                                                        auto xv = make_half4(
                                                            dequant(x[x_base + (g * C_per_group + c + 0u) * x_stride_c]),
                                                            dequant(x[x_base + (g * C_per_group + c + 1u) * x_stride_c]),
                                                            dequant(x[x_base + (g * C_per_group + c + 2u) * x_stride_c]),
                                                            dequant(x[x_base + (g * C_per_group + c + 3u) * x_stride_c]));
                                                        auto wv = make_half4(
                                                            dequant(w[w_base + (c + 0u) * w_stride_c]),
                                                            dequant(w[w_base + (c + 1u) * w_stride_c]),
                                                            dequant(w[w_base + (c + 2u) * w_stride_c]),
                                                            dequant(w[w_base + (c + 3u) * w_stride_c]));
                                                        sum_vec = luisa::compute::fma(xv, wv, sum_vec);
                                                    }
                                                    sum += sum_vec.x + sum_vec.y + sum_vec.z + sum_vec.w;
                                                }
                                            }
                                            for (uint32_t c = c_vec_end; c < C_per_group; ++c) {
                                                auto x_idx = x_base + (g * C_per_group + c) * x_stride_c;
                                                auto w_idx = w_base + c * w_stride_c;
                                                sum = luisa::compute::fma(dequant(x[x_idx]), dequant(w[w_idx]), sum);
                                            }
                                        };
                                    }
                                }
                                // Add bias
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
            } else if (spatial_dims == 1) {
                // 1D conv: X=(N,C,L), W=(M,C/g,kL), Y=(N,M,oL)
                uint32_t iL = x_shape[2];
                uint32_t kL = w_shape[2];
                uint32_t oL = y_shape[2];
                int sL = strides[0], dL = dilations[0], pL = pads[0];
                uint32_t M_per_group = M / group;

                auto x_stride_n = x.strides()[0];
                auto x_stride_c = x.strides()[1];
                auto x_stride_l = x.strides()[2];
                auto w_stride_m = w.strides()[0];
                auto w_stride_c = w.strides()[1];
                auto w_stride_l = w.strides()[2];
                auto y_stride_n = y.strides()[0];
                auto y_stride_m = y.strides()[1];
                auto y_stride_l = y.strides()[2];

                for (auto n : dynamic_range(N)) {
                    for (auto m : dynamic_range(M)) {
                        auto g = m / M_per_group;
                        for (auto ol : dynamic_range(oL)) {
                            auto sum = def(CT{0});
                            for (uint32_t kl = 0; kl < kL; ++kl) {
                                auto il = ol * sL + kl * dL - pL;
                                $if (il >= 0u & il < iL) {
                                    auto x_base = n * x_stride_n + il * x_stride_l;
                                    auto w_base = m * w_stride_m + kl * w_stride_l;
                                    for (uint32_t c = 0; c < C_per_group; ++c) {
                                        auto x_idx = x_base + (g * C_per_group + c) * x_stride_c;
                                        auto w_idx = w_base + c * w_stride_c;
                                        sum = luisa::compute::fma(dequant(x[x_idx]), dequant(w[w_idx]), sum);
                                    }
                                };
                            }
                            if (B_ptr) {
                                auto &b = static_cast<NNTensor<T> &>(*B_ptr);
                                sum += dequant(b[m]);
                            }
                            auto y_idx = n * y_stride_n + m * y_stride_m + ol * y_stride_l;
                            store_y(y_idx, sum);
                        }
                    }
                }
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(Conv) {
    std::string auto_pad = "NOTSET";
    std::vector<int> dilations, kernel_shape, pads, strides;
    int group = 1;
    if (auto p = node.try_get_attr("auto_pad"))
        auto_pad = p->get<onnx::AttributeType::STRING>();
    if (auto p = node.try_get_attr("dilations"))
        dilations = p->get<onnx::AttributeType::INTS>();
    if (auto p = node.try_get_attr("group"))
        group = p->get<onnx::AttributeType::INT>();
    if (auto p = node.try_get_attr("kernel_shape"))
        kernel_shape = p->get<onnx::AttributeType::INTS>();
    if (auto p = node.try_get_attr("pads"))
        pads = p->get<onnx::AttributeType::INTS>();
    if (auto p = node.try_get_attr("strides"))
        strides = p->get<onnx::AttributeType::INTS>();
    return std::make_unique<Conv>(std::move(auto_pad), std::move(dilations), group,
                                  std::move(kernel_shape), std::move(pads), std::move(strides));
};

}// namespace lcml::onnx
