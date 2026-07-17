#include <luisa/core/stl/memory.h>
#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

namespace lcml::onnx {

template<typename T>
struct SeluSupported : std::bool_constant<
    IsFloatingPoint<T>::value ||
    std::is_same_v<T, FP4E2M1> ||
    std::is_same_v<T, FP8E4M3FN> ||
    std::is_same_v<T, FP8E5M2> ||
    std::is_same_v<T, FP16Quantized>> {};

// Selu: f(x) = gamma * (alpha * exp(x) - alpha) if x <= 0, gamma * x if x > 0
// ONNX spec: alpha (default 1.67326319217681884765625), gamma (default 1.05070102214813232421875)
class Selu : public Operator {
private:
    float alpha_;
    float gamma_;

public:
    Selu(float alpha, float gamma) : Operator("Selu"), alpha_(alpha), gamma_(gamma) {}

    /// Element-wise activation: safe for in-place.
    bool can_operate_inplace() const override { return true; }

    void forward(luisa::span<std::reference_wrapper<ITensor>> inputs,
                 luisa::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 1 && outputs.size() == 1, "Selu requires 1 input and 1 output.");
        auto &input = inputs[0].get();
        auto &output = outputs[0].get();
        LUISA_ASSERT(input.element_type_index() == output.element_type_index(), "Selu: input and output must have the same element type.");
        LUISA_ASSERT(input.size() == output.size(), "Selu: input and output must have the same size.");
#else
        auto &input = inputs[0].get();
        auto &output = outputs[0].get();
#endif

        visit_type_index<NNFilteredTypeList<SeluSupported>>(input.element_type_index(), [&]<typename T>() {
            auto &in = static_cast<NNTensor<T> &>(input);
            auto &out = static_cast<NNTensor<T> &>(output);
            if constexpr (std::is_same_v<T, FP8E4M3FN> || std::is_same_v<T, FP8E5M2>) {
                auto deq = [&]() {
                    if constexpr (std::is_same_v<T, FP8E4M3FN>) return fp8e4m3_to_float();
                    else return fp8e5m2_to_float();
                }();
                auto q = [&]() {
                    if constexpr (std::is_same_v<T, FP8E4M3FN>) return fp8e4m3_from_float();
                    else return fp8e5m2_from_float();
                }();
                auto scale = half(gamma_ * alpha_);
                auto gamma = half(gamma_);
                auto zero = half(0.0f);
                for (auto i : dynamic_range(in.size())) {
                    auto x = deq(in[i].bits.cast<ushort>());
                    Var<half> y{half(0.0f)};
                    $if (x <= zero) {
                        y = fma(scale, exp(x), -scale);
                    } $else {
                        y = gamma * x;
                    };
                    out[i].bits = q(y).cast<uint16_t>();
                }
            } else if constexpr (std::is_same_v<T, FP4E2M1>) {
                auto deq = fp4e2m1_to_float();
                auto q = fp4e2m1_from_float();
                auto scale = half(gamma_ * alpha_);
                auto gamma = half(gamma_);
                auto zero = half(0.0f);
                for (auto i : dynamic_range(in.size())) {
                    auto x = deq(in[i].bits.cast<ushort>());
                    Var<half> y{half(0.0f)};
                    $if (x <= zero) {
                        y = fma(scale, exp(x), -scale);
                    } $else {
                        y = gamma * x;
                    };
                    out[i].bits = q(y).cast<uint16_t>();
                }
            } else if constexpr (std::is_same_v<T, FP16Quantized>) {
                auto scale = half(gamma_ * alpha_);
                auto gamma = half(gamma_);
                auto zero = half(0.0f);
                auto selu_half = [&](auto x) {
                    auto result = Var<half>{half(0.0f)};
                    $if (x <= zero) {
                        result = fma(scale, exp(x), -scale);
                    } $else {
                        result = gamma * x;
                    };
                    return result;
                };
                if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                    auto buf_in = in.container().get_byte_buffer();
                    auto buf_o = out.container().get_byte_buffer();
                    auto off_in = static_cast<uint>(in.container().get_byte_offset());
                    auto off_o = static_cast<uint>(out.container().get_byte_offset());
                    auto n = out.size();
                    auto vec_n = static_cast<uint>(n / 4u);
                    auto rem = static_cast<uint>(n % 4u);
                    for (auto i : dynamic_range(vec_n)) {
                        auto byte_idx = i * static_cast<uint>(sizeof(half4));
                        auto x = buf_in->read<half4>(off_in + byte_idx);
                        auto vr = detail::vectorized_unary_scalar(x, selu_half);
                        buf_o->write(off_o + byte_idx, vr);
                    }
                    for (auto i : dynamic_range(rem)) {
                        auto idx = vec_n * 4u + i;
                        auto byte_idx = idx * static_cast<uint>(sizeof(half));
                        auto v = buf_in->read<half>(off_in + byte_idx);
                        buf_o->write(off_o + byte_idx, selu_half(v));
                    }
                    return;
                }
                for (auto i : dynamic_range(in.size())) {
                    auto x = in[i].bits;
                    out[i].bits = selu_half(x);
                }
            } else {
                using VT = nn_storage_type_t<T>;
                auto alpha = Var<VT>{VT(alpha_)};
                auto gamma = Var<VT>{VT(gamma_)};
                auto scale = Var<VT>{VT(gamma_ * alpha_)};
                auto zero = Var<VT>{VT{0}};
                auto selu_scalar = [&](auto x) {
                    auto result = Var<VT>{VT{0}};
                    $if (x <= zero) {
                        result = fma(scale, exp(x), -scale);
                    } $else {
                        result = gamma * x;
                    };
                    return result;
                };
                if constexpr (detail::VecDispatch<VT>::supported) {
                    if (detail::all_byte_buffer(in, out)) {
                        using VecT = typename detail::VecDispatch<VT>::VecT;
                        auto buf_in = in.container().get_byte_buffer();
                        auto buf_o = out.container().get_byte_buffer();
                        auto off_in = static_cast<uint>(in.container().get_byte_offset());
                        auto off_o = static_cast<uint>(out.container().get_byte_offset());
                        auto n = out.size();
                        auto vec_n = static_cast<uint>(n / detail::VecDispatch<VT>::N);
                        auto rem = static_cast<uint>(n % detail::VecDispatch<VT>::N);
                        for (auto i : dynamic_range(vec_n)) {
                            auto byte_idx = i * static_cast<uint>(sizeof(VecT));
                            auto x = buf_in->read<VecT>(off_in + byte_idx);
                            auto vr = detail::vectorized_unary_scalar(x, selu_scalar);
                            buf_o->write(off_o + byte_idx, vr);
                        }
                        for (auto i : dynamic_range(rem)) {
                            auto idx = vec_n * detail::VecDispatch<VT>::N + i;
                            auto byte_idx = idx * static_cast<uint>(sizeof(VT));
                            auto v = buf_in->read<VT>(off_in + byte_idx);
                            buf_o->write(off_o + byte_idx, selu_scalar(v));
                        }
                        return;
                    }
                }
                for (auto i : dynamic_range(in.size())) {
                    out[i] = selu_scalar(in[i]);
                }
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(Selu) {
    float alpha = 1.67326319217681884765625f;
    float gamma = 1.05070102214813232421875f;
    if (auto p = node.try_get_attr("alpha"))
        alpha = p->get<onnx::AttributeType::FLOAT>();
    if (auto p = node.try_get_attr("gamma"))
        gamma = p->get<onnx::AttributeType::FLOAT>();
    return luisa::make_unique<Selu>(alpha, gamma);
};

}// namespace lcml::onnx
