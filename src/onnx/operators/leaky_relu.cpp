#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

namespace lcml::onnx {

template<typename T>
struct LeakyReluSupported : std::bool_constant<
    IsFloatingPoint<T>::value ||
    std::is_same_v<T, FP4E2M1> ||
    std::is_same_v<T, FP8E4M3FN> ||
    std::is_same_v<T, FP8E5M2> ||
    std::is_same_v<T, FP16Quantized>> {};

// LeakyRelu: f(x) = x if x >= 0, alpha * x if x < 0
// ONNX spec: attribute alpha (float, default 0.01)
class LeakyRelu : public Operator {
private:
    float alpha_;

public:
    LeakyRelu(float alpha) : Operator("LeakyRelu"), alpha_(alpha) {}

    /// Element-wise activation: safe for in-place.
    bool can_operate_inplace() const override { return true; }

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 1 && outputs.size() == 1, "LeakyRelu requires 1 input and 1 output.");
        auto &input = inputs[0].get();
        auto &output = outputs[0].get();
        LUISA_ASSERT(input.element_type() == output.element_type(), "LeakyRelu: input and output must have the same element type.");
        LUISA_ASSERT(input.size() == output.size(), "LeakyRelu: input and output must have the same size.");
#else
        auto &input = inputs[0].get();
        auto &output = outputs[0].get();
#endif

        visit_typeid<NNFilteredTypeList<LeakyReluSupported>>(input.element_type(), [&]<typename T>() {
            auto &in = static_cast<NNTensor<T> &>(input);
            auto &out = static_cast<NNTensor<T> &>(output);
            if constexpr (std::is_same_v<T, FP4E2M1> || std::is_same_v<T, FP8E4M3FN> || std::is_same_v<T, FP8E5M2>) {
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
                auto alpha_m1 = half(alpha_ - 1.0f);
                auto zero = half(0.0f);
                for (auto i : dynamic_range(in.size())) {
                    auto x = deq(in[i].bits.cast<ushort>());
                    auto y = fma(min(x, zero), alpha_m1, x);
                    out[i].bits = q(y).cast<uint16_t>();
                }
            } else if constexpr (std::is_same_v<T, FP16Quantized>) {
                auto alpha_m1 = half(alpha_ - 1.0f);
                auto zero = half(0.0f);
                for (auto i : dynamic_range(in.size())) {
                    auto x = in[i].bits;
                    auto y = fma(min(x, zero), alpha_m1, x);
                    out[i].bits = y;
                }
            } else {
                using VT = nn_storage_type_t<T>;
                auto alpha_m1 = Var<VT>{VT(alpha_ - 1.0f)};
                auto zero = Var<VT>{VT{0}};
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
                        auto vec_alpha_m1 = detail::VecDispatch<VT>::broadcast(alpha_m1);
                        auto vec_zero = detail::VecDispatch<VT>::broadcast(zero);
                        for (auto i : dynamic_range(vec_n)) {
                            auto byte_idx = i * static_cast<uint>(sizeof(VecT));
                            auto x = buf_in->read<VecT>(off_in + byte_idx);
                            auto vr = fma(min(x, vec_zero), vec_alpha_m1, x);
                            buf_o->write(off_o + byte_idx, vr);
                        }
                        for (auto i : dynamic_range(rem)) {
                            auto byte_idx = (vec_n * detail::VecDispatch<VT>::N + i) * static_cast<uint>(sizeof(VT));
                            auto x = buf_in->read<VT>(off_in + byte_idx);
                            buf_o->write(off_o + byte_idx, fma(min(x, zero), alpha_m1, x));
                        }
                        return;
                    }
                }
                for (auto i : dynamic_range(in.size())) {
                    auto x = in[i];
                    out[i] = fma(min(x, zero), alpha_m1, x);
                }
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(LeakyRelu) {
    float alpha = 0.01f;
    if (auto p = node.try_get_attr("alpha"))
        alpha = p->get<onnx::AttributeType::FLOAT>();
    return std::make_unique<LeakyRelu>(alpha);
};

}// namespace lcml::onnx
