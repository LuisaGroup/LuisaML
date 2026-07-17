#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"
#include "onnx/fp_quantized.h"
#include <luisa/dsl/sugar.h>
#include <luisa/core/stl/memory.h>

namespace lcml::onnx {

class Cast : public Operator {
private:
    onnx::DataType to_;
    int32_t saturate_;

    template<typename T1, typename T2>
    void apply(T1 &input, T2 &output) const {
        using SrcT = typename T1::value_type;
        using DstT = typename T2::value_type;
        auto const &shape = input.shape();
        auto const &out_shape = output.shape();

#ifndef NDEBUG
        LUISA_ASSERT(shape == out_shape,
                     "Cast: input and output shapes must match");
#endif

        if constexpr (std::is_same_v<SrcT, DstT>) {
            if constexpr (detail::VecDispatch<SrcT>::supported) {
                if (detail::all_byte_buffer(input, output)) {
                    detail::vectorized_unary<SrcT>(
                        input, output,
                        [&](auto v) { return v; });
                    return;
                }
            }
            for (auto i : dynamic_range(input.size())) {
                output[i] = input[i];
            }
        } else if constexpr (std::is_same_v<SrcT, FP4E2M1>) {
            auto fp4_to = fp4e2m1_to_float();
            for (auto i : dynamic_range(input.size())) {
                output[i] = fp4_to(input[i].bits.cast<ushort>()).template cast<DstT>();
            }
        } else if constexpr (std::is_same_v<SrcT, FP8E4M3FN>) {
            auto fp8_to = fp8e4m3_to_float();
            for (auto i : dynamic_range(input.size())) {
                output[i] = fp8_to(input[i].bits.cast<ushort>()).template cast<DstT>();
            }
        } else if constexpr (std::is_same_v<SrcT, FP8E5M2>) {
            auto fp8_to = fp8e5m2_to_float();
            for (auto i : dynamic_range(input.size())) {
                output[i] = fp8_to(input[i].bits.cast<ushort>()).template cast<DstT>();
            }
        } else if constexpr (std::is_same_v<SrcT, FP16Quantized>) {
            for (auto i : dynamic_range(input.size())) {
                output[i] = input[i].bits.template cast<DstT>();
            }
        } else if constexpr (std::is_same_v<DstT, FP4E2M1>) {
            auto fp4_from = fp4e2m1_from_float();
            for (auto i : dynamic_range(input.size())) {
                output[i].bits = fp4_from(input[i].template cast<half>()).cast<uint16_t>();
            }
        } else if constexpr (std::is_same_v<DstT, FP8E4M3FN>) {
            auto fp8_from = fp8e4m3_from_float();
            for (auto i : dynamic_range(input.size())) {
                output[i].bits = fp8_from(input[i].template cast<half>()).cast<uint16_t>();
            }
        } else if constexpr (std::is_same_v<DstT, FP8E5M2>) {
            auto fp8_from = fp8e5m2_from_float();
            for (auto i : dynamic_range(input.size())) {
                output[i].bits = fp8_from(input[i].template cast<half>()).cast<uint16_t>();
            }
        } else if constexpr (std::is_same_v<DstT, FP16Quantized>) {
            for (auto i : dynamic_range(input.size())) {
                output[i].bits = input[i].template cast<half>();
            }
        } else {
            if constexpr (detail::VecDispatch<SrcT>::supported && detail::VecDispatch<DstT>::supported) {
                if (detail::all_byte_buffer(input, output)) {
                    detail::vectorized_cast<SrcT, DstT>(
                        input, output,
                        [&](auto v) { return cast<DstT>(v); });
                    return;
                }
            }
            if (saturate_ == 1 && luisa::is_floating_point_v<SrcT> &&
                luisa::is_integral_v<DstT> && !std::is_same_v<DstT, bool>) {
                auto min_val = static_cast<SrcT>(std::numeric_limits<DstT>::min());
                auto max_val = static_cast<SrcT>(std::numeric_limits<DstT>::max());
                for (auto i : dynamic_range(input.size())) {
                    $if (input[i] >= max_val) {
                        output[i] = std::numeric_limits<DstT>::max();
                    } $elif (input[i] <= min_val) {
                        output[i] = std::numeric_limits<DstT>::min();
                    } $else {
                        output[i] = input[i].template cast<DstT>();
                    };
                }
            } else {
                for (auto i : dynamic_range(input.size())) {
                    output[i] = input[i].template cast<DstT>();
                }
            }
        }
    }

    // Map onnx::DataType to C++ type and invoke the apply function
    template<typename SrcT>
    void dispatch_dst(NNTensor<SrcT> &input, ITensor &output) const {
        visit_onnx_dtype(to_, [&]<typename DstT>() {
            apply(input, static_cast<NNTensor<DstT> &>(output));
        });
    }

public:
    Cast(onnx::DataType to, int32_t saturate)
        : Operator("Cast"), to_(to), saturate_(saturate) {}

    bool can_operate_inplace() const override { return true; }

    void forward(luisa::span<std::reference_wrapper<ITensor>> inputs,
                 luisa::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 1 && outputs.size() == 1,
                     "Cast requires exactly 1 input and 1 output.");
#endif
        auto &input = inputs[0].get();
        auto &output = outputs[0].get();

#ifndef NDEBUG
        LUISA_ASSERT(input.ndim() != 0,
                     "Cast: input tensor must not be empty.");
#endif

        visit_type_index<NNTypeList>(input.element_type_index(), [&]<typename SrcT>() {
            auto &in = static_cast<NNTensor<SrcT> &>(input);
            // Fast path: same type and shared storage -> zero-copy bypass
            if (input.element_type_index() == output.element_type_index()) {
                auto &out = static_cast<NNTensor<SrcT> &>(output);
                if (in.container().shares_storage_with(out.container())) {
                    return;
                }
            }
            dispatch_dst<SrcT>(in, output);
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(Cast) {
    int32_t to = 0;
    int32_t saturate = 1;

    auto p_to = node.try_get_attr("to");
    LUISA_ASSERT(p_to != nullptr, "Cast: 'to' attribute is required.");
    to = p_to->get<onnx::AttributeType::INT>();

    if (auto p = node.try_get_attr("saturate")) {
        saturate = p->get<onnx::AttributeType::INT>();
    }

    return luisa::make_unique<Cast>(static_cast<onnx::DataType>(to), saturate);
};

}// namespace lcml::onnx