#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"
#include "onnx/fp_quantized.h"

namespace lcml::onnx {

// AveragePool: computes average pooling over spatial dimensions.
// ONNX spec: attributes: auto_pad, ceil_mode, count_include_pad, kernel_shape, pads, strides
class AveragePool : public Operator {
private:
    std::vector<int> kernel_shape_;
    std::vector<int> pads_;
    std::vector<int> strides_;
    int count_include_pad_;

public:
    AveragePool(std::vector<int> kernel_shape, std::vector<int> pads,
                std::vector<int> strides, int count_include_pad)
        : Operator("AveragePool"), kernel_shape_(std::move(kernel_shape)),
          pads_(std::move(pads)), strides_(std::move(strides)),
          count_include_pad_(count_include_pad) {}

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 1 && outputs.size() == 1,
                     "AveragePool requires 1 input and 1 output.");
        auto &X = inputs[0].get();
        auto &Y = outputs[0].get();

        auto const &x_shape = X.shape();// (N, C, H, W)
        auto const &y_shape = Y.shape();// (N, C, oH, oW)
        auto spatial_dims = x_shape.size() - 2;

        std::vector<int> strides = strides_;
        std::vector<int> pads = pads_;
        if (strides.empty()) strides.assign(spatial_dims, 1);
        if (pads.empty()) pads.assign(spatial_dims * 2, 0);
        // ONNX pads for 2D: [top, left, bottom, right]; expand symmetric if only 2 given
        if (pads.size() == 2 && spatial_dims == 2) {
            pads.push_back(pads[0]);
            pads.push_back(pads[1]);
        }

        LUISA_ASSERT(spatial_dims == 2, "AveragePool: only 2D pooling is supported.");
#else
        auto &X = inputs[0].get();
        auto &Y = outputs[0].get();

        auto const &x_shape = X.shape();// (N, C, H, W)
        auto const &y_shape = Y.shape();
        auto spatial_dims = x_shape.size() - 2;

        std::vector<int> strides = strides_;
        std::vector<int> pads = pads_;
        if (strides.empty()) strides.assign(spatial_dims, 1);
        if (pads.empty()) pads.assign(spatial_dims * 2, 0);
        // ONNX pads for 2D: [top, left, bottom, right]; expand symmetric if only 2 given
        if (pads.size() == 2 && spatial_dims == 2) {
            pads.push_back(pads[0]);
            pads.push_back(pads[1]);
        }
#endif

        visit_typeid<NNTypeList>(X.element_type(), [&]<typename T>() {
            using VT = nn_storage_type_t<T>;
            auto &x = static_cast<NNTensor<T> &>(X);
            auto &y = static_cast<NNTensor<T> &>(Y);

            uint32_t N = x_shape[0], C = x_shape[1];
            uint32_t iH = x_shape[2], iW = x_shape[3];
            uint32_t kH = kernel_shape_[0], kW = kernel_shape_[1];
            uint32_t oH = y_shape[2], oW = y_shape[3];
            int sH = strides[0], sW = strides[1];
            int pH0 = pads[0], pW0 = pads[1];

            if constexpr (IsFloatingPoint<T>::value) {
                auto one = Var<VT>{VT{1}};
                auto inv_divisor_const = def(VT(1.0) / static_cast<VT>(kH * kW));

                // Fast path: vectorized buffer reads when input is backed by a ByteBuffer
                // and the innermost dimension is contiguous (stride == 1).
                bool use_vec = x.container().is_byte_buffer() && x.strides()[3] == 1;

                for (auto n : dynamic_range(N)) {
                    for (auto c : dynamic_range(C)) {
                        auto base_x = n * x.strides()[0] + c * x.strides()[1];
                        auto base_y = n * y.strides()[0] + c * y.strides()[1];
                        for (auto oh : dynamic_range(oH)) {
                            for (auto ow : dynamic_range(oW)) {
                                auto sum = def(VT{0});
                                auto count = def(0u);

                                if (use_vec) {
                                    auto buf_x = x.container().get_byte_buffer();
                                    auto off_x = static_cast<uint>(x.container().get_byte_offset());
                                    for (uint32_t kh = 0; kh < kH; ++kh) {
                                        auto ih = oh * sH + kh - pH0;
                                        auto iw0 = ow * sW - pW0;
                                        auto iw_last = iw0 + (kW - 1u);
                                        auto row_base = base_x + ih * x.strides()[2];
                                        $if (ih >= 0u & ih < iH & iw0 >= 0u & iw_last < iW) {
                                            // Entire kernel row is in bounds: use vectorized reads.
                                            if constexpr (std::is_same_v<VT, float>) {
                                                auto sum_vec = def(make_float4(0.0f));
                                                auto one_vec = make_float4(one);
                                                uint32_t kw_vec_end = (kW / 4u) * 4u;
                                                for (uint32_t kw = 0; kw < kw_vec_end; kw += 4u) {
                                                    auto byte_off = off_x + (row_base + iw0 + kw) * static_cast<uint>(sizeof(float));
                                                    auto v = buf_x->read<float4>(byte_off);
                                                    sum_vec = luisa::compute::fma(v, one_vec, sum_vec);
                                                }
                                                sum += sum_vec.x + sum_vec.y + sum_vec.z + sum_vec.w;
                                                for (uint32_t kw = kw_vec_end; kw < kW; ++kw) {
                                                    sum = luisa::compute::fma(x[row_base + (iw0 + kw) * x.strides()[3]], one, sum);
                                                }
                                            } else {
                                                auto sum_vec = def(make_half4(half(0.0f)));
                                                auto one_vec = make_half4(one);
                                                uint32_t kw_vec_end = (kW / 4u) * 4u;
                                                for (uint32_t kw = 0; kw < kw_vec_end; kw += 4u) {
                                                    auto byte_off = off_x + (row_base + iw0 + kw) * static_cast<uint>(sizeof(half));
                                                    auto v = buf_x->read<half4>(byte_off);
                                                    sum_vec = luisa::compute::fma(v, one_vec, sum_vec);
                                                }
                                                sum += sum_vec.x + sum_vec.y + sum_vec.z + sum_vec.w;
                                                for (uint32_t kw = kw_vec_end; kw < kW; ++kw) {
                                                    sum = luisa::compute::fma(x[row_base + (iw0 + kw) * x.strides()[3]], one, sum);
                                                }
                                            }
                                            count += kW;
                                        } $else {
                                            // Partial row: scalar fallback with per-element bounds.
                                            for (uint32_t kw = 0; kw < kW; ++kw) {
                                                auto iw = ow * sW + kw - pW0;
                                                $if (ih >= 0u & ih < iH & iw >= 0u & iw < iW) {
                                                    sum = luisa::compute::fma(x[row_base + iw * x.strides()[3]], one, sum);
                                                    count += 1u;
                                                };
                                            }
                                        };
                                    }
                                } else {
                                    // Scalar path: exact bounds to avoid inner branch.
                                    auto oh_i = oh.cast<int>();
                                    auto ow_i = ow.cast<int>();
                                    auto kh_b = max(def(0), pH0 - oh_i * sH);
                                    auto kh_e = max(def(0), min(def(static_cast<int>(kH)), static_cast<int>(iH) + pH0 - oh_i * sH));
                                    for (auto kh : dynamic_range(kh_b.cast<uint>(), kh_e.cast<uint>())) {
                                        auto ih = oh * sH + kh - pH0;
                                        auto row_base = base_x + ih * x.strides()[2];
                                        auto kw_b = max(def(0), pW0 - ow_i * sW);
                                        auto kw_e = max(def(0), min(def(static_cast<int>(kW)), static_cast<int>(iW) + pW0 - ow_i * sW));
                                        for (auto kw : dynamic_range(kw_b.cast<uint>(), kw_e.cast<uint>())) {
                                            auto iw = ow * sW + kw - pW0;
                                            sum = luisa::compute::fma(x[row_base + iw * x.strides()[3]], one, sum);
                                            count += 1u;
                                        }
                                    }
                                }

                                auto divisor = count_include_pad_ ? def(kH * kW) : count;
                                auto result = count_include_pad_ ? sum * inv_divisor_const : sum / divisor.cast<VT>();
                                y[base_y + oh * y.strides()[2] + ow * y.strides()[3]] = result;
                            }
                        }
                    }
                }
            } else if constexpr (std::is_same_v<T, FP8E4M3FN> || std::is_same_v<T, FP8E5M2> || std::is_same_v<T, FP4E2M1>) {
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
                auto inv_divisor_const_f = 1.0f / static_cast<float>(kH * kW);
                auto one_f = def(1.0f);

                for (auto n : dynamic_range(N)) {
                    for (auto c : dynamic_range(C)) {
                        auto base_x = n * x.strides()[0] + c * x.strides()[1];
                        auto base_y = n * y.strides()[0] + c * y.strides()[1];
                        for (auto oh : dynamic_range(oH)) {
                            auto oh_i = oh.cast<int>();
                            auto kh_b = max(def(0), pH0 - oh_i * sH);
                            auto kh_e = max(def(0), min(def(static_cast<int>(kH)), static_cast<int>(iH) + pH0 - oh_i * sH));
                            for (auto ow : dynamic_range(oW)) {
                                auto ow_i = ow.cast<int>();
                                auto kw_b = max(def(0), pW0 - ow_i * sW);
                                auto kw_e = max(def(0), min(def(static_cast<int>(kW)), static_cast<int>(iW) + pW0 - ow_i * sW));

                                auto sum = def(0.0f);
                                auto count = def(0u);
                                for (auto kh : dynamic_range(kh_b.cast<uint>(), kh_e.cast<uint>())) {
                                    auto ih = oh * sH + kh - pH0;
                                    auto row_base = base_x + ih * x.strides()[2];
                                    for (auto kw : dynamic_range(kw_b.cast<uint>(), kw_e.cast<uint>())) {
                                        auto iw = ow * sW + kw - pW0;
                                        auto bits = x[row_base + iw * x.strides()[3]].bits.cast<ushort>();
                                        auto val = deq(bits);
                                        sum = luisa::compute::fma(val.cast<float>(), one_f, sum);
                                        count += 1u;
                                    }
                                }
                                auto result = count_include_pad_ ? sum * inv_divisor_const_f : sum / count.cast<float>();
                                auto out_bits = q(cast<half>(result));
                                y[base_y + oh * y.strides()[2] + ow * y.strides()[3]].bits = out_bits.cast<uint16_t>();
                            }
                        }
                    }
                }
            } else {
                LUISA_ASSERT(false, "AveragePool: unsupported element type");
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(AveragePool) {
    std::vector<int> kernel_shape, pads, strides;
    int count_include_pad = 0;
    if (auto p = node.try_get_attr("kernel_shape"))
        kernel_shape = p->get<onnx::AttributeType::INTS>();
    if (auto p = node.try_get_attr("pads"))
        pads = p->get<onnx::AttributeType::INTS>();
    if (auto p = node.try_get_attr("strides"))
        strides = p->get<onnx::AttributeType::INTS>();
    if (auto p = node.try_get_attr("count_include_pad"))
        count_include_pad = p->get<onnx::AttributeType::INT>();
    return std::make_unique<AveragePool>(std::move(kernel_shape), std::move(pads),
                                         std::move(strides), count_include_pad);
};

}// namespace lcml::onnx
