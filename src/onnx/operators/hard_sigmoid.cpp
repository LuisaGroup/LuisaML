#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

#include <luisa/core/stl/memory.h>

namespace lcml::onnx {

// HardSigmoid: max(0, min(1, alpha * x + beta))
// ONNX spec: alpha (default 0.2), beta (default 0.5)
class HardSigmoid : public Operator {
private:
    float alpha_;
    float beta_;

public:
    HardSigmoid(float alpha, float beta) : Operator("HardSigmoid"), alpha_(alpha), beta_(beta) {}

    /// Element-wise activation: safe for in-place.
    bool can_operate_inplace() const override { return true; }

    void forward(luisa::span<std::reference_wrapper<ITensor>> inputs,
                 luisa::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 1 && outputs.size() == 1, "HardSigmoid requires 1 input and 1 output.");
#endif
        auto &input = inputs[0].get();
        auto &output = outputs[0].get();
#ifndef NDEBUG
        LUISA_ASSERT(input.element_type_index() == output.element_type_index(), "HardSigmoid: input and output must have the same element type.");
        LUISA_ASSERT(input.size() == output.size(), "HardSigmoid: input and output must have the same size.");
        LUISA_ASSERT(input.element_type_index() == refl::type_index_of<float>() || input.element_type_index() == refl::type_index_of<half>(),
                     "HardSigmoid: unsupported element type");
#endif

        if (input.size() == 0) return;

        visit_type_index<float, half>(input.element_type_index(), [&]<typename T>() {
            using VT = nn_storage_type_t<T>;
            auto &in = static_cast<NNTensor<T> &>(input);
            auto &out = static_cast<NNTensor<T> &>(output);
            auto alpha = Var<VT>{VT(alpha_)};
            auto beta = Var<VT>{VT(beta_)};
            auto zero = Var<VT>{VT{0}};
            auto one = Var<VT>{VT{1}};

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
                    auto vec_alpha = detail::VecDispatch<VT>::broadcast(alpha);
                    auto vec_beta = detail::VecDispatch<VT>::broadcast(beta);
                    auto vec_zero = detail::VecDispatch<VT>::broadcast(zero);
                    auto vec_one = detail::VecDispatch<VT>::broadcast(one);
                    for (auto i : dynamic_range(vec_n)) {
                        auto byte_idx = i * static_cast<uint>(sizeof(VecT));
                        auto x = buf_in->read<VecT>(off_in + byte_idx);
                        auto vr = clamp(fma(vec_alpha, x, vec_beta), vec_zero, vec_one);
                        buf_o->write(off_o + byte_idx, vr);
                    }
                    for (auto i : dynamic_range(rem)) {
                        auto byte_idx = (vec_n * detail::VecDispatch<VT>::N + i) * static_cast<uint>(sizeof(VT));
                        auto x = buf_in->read<VT>(off_in + byte_idx);
                        auto vr = clamp(fma(alpha, x, beta), zero, one);
                        buf_o->write(off_o + byte_idx, vr);
                    }
                    return;
                }
            }
            for (auto i : dynamic_range(in.size())) {
                out[i] = clamp(fma(alpha, in[i], beta), zero, one);
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(HardSigmoid) {
    float alpha = 0.2f;
    float beta = 0.5f;
    if (auto p = node.try_get_attr("alpha"))
        alpha = p->get<onnx::AttributeType::FLOAT>();
    if (auto p = node.try_get_attr("beta"))
        beta = p->get<onnx::AttributeType::FLOAT>();
    return luisa::make_unique<HardSigmoid>(alpha, beta);
};

}// namespace lcml::onnx
