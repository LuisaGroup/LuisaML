
#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

namespace lcml::onnx {

// Identity: returns the input tensor as-is.
// ONNX spec: output = input (no transformation).
class Identity : public Operator {
public:
    Identity() : Operator("Identity") {}

    bool is_output_view([[maybe_unused]] size_t output_index,
                        [[maybe_unused]] onnx::Node const &node) const override { return true; }

    bool can_operate_inplace() const override { return true; }

    bool need_outline() const override { return false; }

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 1 && outputs.size() == 1, "Identity requires 1 input and 1 output.");
#endif
        auto &input = inputs[0].get();
        auto &output = outputs[0].get();
#ifndef NDEBUG
        LUISA_ASSERT(input.size() == output.size(), "Identity: input and output must have the same total size.");
        LUISA_ASSERT(input.element_type() == output.element_type(), "Identity: input and output must have the same element type.");
#endif

        visit_typeid<NNTypeList>(input.element_type(), [&]<typename T>() {
            auto &in = static_cast<NNTensor<T> &>(input);
            auto &out = static_cast<NNTensor<T> &>(output);
            // If already sharing storage (inplace allocation), skip assignment
            if (in.container().shares_storage_with(out.container())) return;
            // Vectorized copy fallback
            using ST = nn_storage_type_t<T>;
            if constexpr (IsNativeArithmetic<T>::value && detail::VecDispatch<ST>::supported) {
                if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                    detail::vectorized_unary<ST>(in, out, [](auto v) { return v; });
                } else {
                    for (auto i : dynamic_range(static_cast<uint>(out.size()))) {
                        out[i] = in[i];
                    }
                }
            } else if constexpr (std::is_same_v<T, FP4E2M1>) {
                if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                    auto buf_in = in.container().get_byte_buffer();
                    auto buf_out = out.container().get_byte_buffer();
                    auto off_in = static_cast<uint>(in.container().get_byte_offset());
                    auto off_out = static_cast<uint>(out.container().get_byte_offset());
                    auto n = static_cast<uint>(out.size());
                    auto vec4_count = n / 32u;
                    auto rem = n % 32u;
                    for (auto i : dynamic_range(vec4_count)) {
                        auto v = buf_in->read<uint4>(off_in + i * 16u);
                        buf_out->write(off_out + i * 16u, v);
                    }
                    if (rem > 0) {
                        auto rem_word_count = (rem + 7u) / 8u;
                        for (auto i : dynamic_range(rem_word_count)) {
                            auto v = buf_in->read<uint>(off_in + vec4_count * 16u + i * 4u);
                            buf_out->write(off_out + vec4_count * 16u + i * 4u, v);
                        }
                    }
                } else {
                    for (auto i : dynamic_range(static_cast<uint>(out.size()))) {
                        out[i] = in[i];
                    }
                }
            } else if constexpr (std::is_same_v<T, FP8E4M3FN> || std::is_same_v<T, FP8E5M2>) {
                if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                    auto buf_in = in.container().get_byte_buffer();
                    auto buf_out = out.container().get_byte_buffer();
                    auto off_in = static_cast<uint>(in.container().get_byte_offset());
                    auto off_out = static_cast<uint>(out.container().get_byte_offset());
                    auto n = static_cast<uint>(out.size());
                    auto vec4_count = n / 16u;
                    auto rem = n % 16u;
                    for (auto i : dynamic_range(vec4_count)) {
                        auto v = buf_in->read<uint4>(off_in + i * 16u);
                        buf_out->write(off_out + i * 16u, v);
                    }
                    if (rem > 0) {
                        auto rem_word_count = (rem + 3u) / 4u;
                        for (auto i : dynamic_range(rem_word_count)) {
                            auto v = buf_in->read<uint>(off_in + vec4_count * 16u + i * 4u);
                            buf_out->write(off_out + vec4_count * 16u + i * 4u, v);
                        }
                    }
                } else {
                    for (auto i : dynamic_range(static_cast<uint>(out.size()))) {
                        out[i] = in[i];
                    }
                }
            } else if constexpr (std::is_same_v<T, FP16Quantized>) {
                if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                    auto buf_in = in.container().get_byte_buffer();
                    auto buf_out = out.container().get_byte_buffer();
                    auto off_in = static_cast<uint>(in.container().get_byte_offset());
                    auto off_out = static_cast<uint>(out.container().get_byte_offset());
                    auto n = static_cast<uint>(out.size());
                    auto vec4_count = n / 4u;
                    auto rem = n % 4u;
                    for (auto i : dynamic_range(vec4_count)) {
                        auto v = buf_in->read<half4>(off_in + i * static_cast<uint>(sizeof(half4)));
                        buf_out->write(off_out + i * static_cast<uint>(sizeof(half4)), v);
                    }
                    for (auto i : dynamic_range(rem)) {
                        auto idx = vec4_count * 4u + i;
                        auto v = buf_in->read<half>(off_in + idx * static_cast<uint>(sizeof(half)));
                        buf_out->write(off_out + idx * static_cast<uint>(sizeof(half)), v);
                    }
                } else {
                    for (auto i : dynamic_range(static_cast<uint>(out.size()))) {
                        out[i] = in[i];
                    }
                }
            } else {
                if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                    auto buf_in = in.container().get_byte_buffer();
                    auto buf_out = out.container().get_byte_buffer();
                    auto off_in = static_cast<uint>(in.container().get_byte_offset());
                    auto off_out = static_cast<uint>(out.container().get_byte_offset());
                    auto n = static_cast<uint>(out.size());
                    for (auto i : dynamic_range(n)) {
                        auto byte_idx = i * static_cast<uint>(sizeof(ST));
                        auto v = buf_in->read<ST>(off_in + byte_idx);
                        buf_out->write(off_out + byte_idx, v);
                    }
                } else {
                    for (auto i : dynamic_range(static_cast<uint>(out.size()))) {
                        out[i] = in[i];
                    }
                }
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(Identity) {
    return std::make_unique<Identity>();
};

}// namespace lcml::onnx
