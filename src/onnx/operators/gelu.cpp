#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>

namespace lcml::onnx {

// Gelu: Gaussian Error Linear Unit
// ONNX spec: attribute approximate ("none" | "tanh")
//   none:  x * 0.5 * (1 + erf(x / sqrt(2)))
//   tanh:  x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
class Gelu : public Operator {
private:
    luisa::string approximate_;

public:
    Gelu(luisa::string approximate) : Operator("Gelu"), approximate_(std::move(approximate)) {}

    /// Element-wise activation: safe for in-place.
    bool can_operate_inplace() const override { return true; }

    void forward(luisa::span<std::reference_wrapper<ITensor>> inputs,
                 luisa::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 1 && outputs.size() == 1,
                     "Gelu requires 1 input and 1 output.");
        auto &input = inputs[0].get();
        auto &output = outputs[0].get();
        LUISA_ASSERT(input.element_type_index() == output.element_type_index(),
                     "Gelu: input and output must have the same element type.");
        LUISA_ASSERT(input.size() == output.size(),
                     "Gelu: input and output must have the same size.");
        LUISA_ASSERT(approximate_ == "none" || approximate_ == "tanh",
                     "Gelu: unsupported approximate attribute: {}", approximate_);
#else
        auto &input = inputs[0].get();
        auto &output = outputs[0].get();
#endif

        bool use_tanh = (approximate_ == "tanh");

        visit_type_index<NNFilteredTypeList<IsFloatingPoint>>(input.element_type_index(), [&]<typename T>() {
            using VT = nn_storage_type_t<T>;
            auto &in = static_cast<NNTensor<T> &>(input);
            auto &out = static_cast<NNTensor<T> &>(output);

            if (use_tanh) {
                // tanh approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
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
                        auto sqrt_2_over_pi = detail::VecDispatch<VT>::broadcast(Var<VT>{VT{0.79788456f}});
                        auto c = detail::VecDispatch<VT>::broadcast(Var<VT>{VT{0.044715f}});
                        auto half = detail::VecDispatch<VT>::broadcast(Var<VT>{VT{0.5f}});
                        auto one = detail::VecDispatch<VT>::broadcast(Var<VT>{VT{1.0f}});
                        for (auto i : dynamic_range(vec_n)) {
                            auto byte_idx = i * static_cast<uint>(sizeof(VecT));
                            auto x = buf_in->read<VecT>(off_in + byte_idx);
                            auto x2 = x * x;
                            auto inner = sqrt_2_over_pi * fma(c * x2, x, x);
                            auto vr = x * half * (one + tanh(inner));
                            buf_o->write(off_o + byte_idx, vr);
                        }
                        for (auto i : dynamic_range(rem)) {
                            auto byte_idx = (vec_n * detail::VecDispatch<VT>::N + i) * static_cast<uint>(sizeof(VT));
                            auto x = buf_in->read<VT>(off_in + byte_idx);
                            auto x2 = x * x;
                            auto inner = Var<VT>{VT{0.79788456f}} * fma(Var<VT>{VT{0.044715f}} * x2, x, x);
                            auto vr = x * Var<VT>{VT{0.5f}} * (Var<VT>{VT{1.0f}} + tanh(inner));
                            buf_o->write(off_o + byte_idx, vr);
                        }
                        return;
                    }
                }
                auto sqrt_2_over_pi = Var<VT>{VT{0.79788456f}};
                auto c = Var<VT>{VT{0.044715f}};
                for (auto i : dynamic_range(in.size())) {
                    auto x = in[i];
                    auto x2 = x * x;
                    auto inner = sqrt_2_over_pi * fma(c * x2, x, x);
                    out[i] = x * Var<VT>{VT{0.5f}} * (Var<VT>{VT{1.0f}} + tanh(inner));
                }
            } else {
                // exact: x * 0.5 * (1 + erf(x / sqrt(2)))
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
                        auto inv_sqrt2 = detail::VecDispatch<VT>::broadcast(Var<VT>{VT{0.70710678f}});
                        auto half = detail::VecDispatch<VT>::broadcast(Var<VT>{VT{0.5f}});
                        auto one = detail::VecDispatch<VT>::broadcast(Var<VT>{VT{1.0f}});
                        for (auto i : dynamic_range(vec_n)) {
                            auto byte_idx = i * static_cast<uint>(sizeof(VecT));
                            auto x = buf_in->read<VecT>(off_in + byte_idx);
                            auto vr = x * half * (one + erf(x * inv_sqrt2));
                            buf_o->write(off_o + byte_idx, vr);
                        }
                        for (auto i : dynamic_range(rem)) {
                            auto byte_idx = (vec_n * detail::VecDispatch<VT>::N + i) * static_cast<uint>(sizeof(VT));
                            auto x = buf_in->read<VT>(off_in + byte_idx);
                            auto vr = x * Var<VT>{VT{0.5f}} * (Var<VT>{VT{1.0f}} + erf(x * Var<VT>{VT{0.70710678f}}));
                            buf_o->write(off_o + byte_idx, vr);
                        }
                        return;
                    }
                }
                auto inv_sqrt2 = Var<VT>{VT{0.70710678f}};
                for (auto i : dynamic_range(in.size())) {
                    auto x = in[i];
                    out[i] = x * Var<VT>{VT{0.5f}} * (Var<VT>{VT{1.0f}} + erf(x * inv_sqrt2));
                }
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(Gelu) {
    luisa::string approximate = "none";
    if (auto p = node.try_get_attr("approximate"))
        approximate = p->get<onnx::AttributeType::STRING>();
    return luisa::make_unique<Gelu>(std::move(approximate));
};

}// namespace lcml::onnx
