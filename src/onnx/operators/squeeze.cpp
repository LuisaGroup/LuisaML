#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

namespace lcml::onnx {

namespace detail {

template<typename In, typename Out>
inline void squeeze_copy(In &in, Out &out) {
    // If already sharing storage (inplace allocation), skip assignment
    if (in.container().shares_storage_with(out.container())) return;

    using ST = typename In::value_type;
    static_assert(std::is_same_v<ST, typename Out::value_type>, "Mismatched tensor value types");

    // Fast path 1: native vectorized copy for types with VecDispatch (float, half)
    // This also covers double (storage=float) and FP16Quantized (storage=half)
    if constexpr (detail::VecDispatch<ST>::supported) {
        if (detail::all_byte_buffer(in, out)) {
            detail::vectorized_unary<ST>(in, out, [](auto v) { return v; });
            return;
        }
    }

    // Fast path 2: packed FP4 copy via uint4 (32 elements = 16 bytes)
    if constexpr (std::is_same_v<ST, FP4E2M1>) {
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
            return;
        }
    }
    // Fast path 3: packed FP8 copy via uint4 (16 elements = 16 bytes)
    else if constexpr (std::is_same_v<ST, FP8E4M3FN> || std::is_same_v<ST, FP8E5M2>) {
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
            return;
        }
    }
    // Fast path 4: generic byte-vectorized copy for dense types (int, uint, short,
    // ushort, bool, etc.) backed by ByteBuffer. Copies in 16-byte uint4 chunks
    // with per-element scalar remainder to stay within buffer bounds.
    else {
        if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
            auto buf_in = in.container().get_byte_buffer();
            auto buf_out = out.container().get_byte_buffer();
            auto off_in = static_cast<uint>(in.container().get_byte_offset());
            auto off_out = static_cast<uint>(out.container().get_byte_offset());
            auto n = static_cast<uint>(out.size());
            auto elem_size = static_cast<uint>(sizeof(ST));
            auto total_bytes = n * elem_size;
            auto vec4_count = total_bytes / 16u;
            auto rem_bytes = total_bytes % 16u;
            auto rem_elems = rem_bytes / elem_size;
            for (auto i : dynamic_range(vec4_count)) {
                auto v = buf_in->read<uint4>(off_in + i * 16u);
                buf_out->write(off_out + i * 16u, v);
            }
            for (auto i : dynamic_range(rem_elems)) {
                auto idx = vec4_count * (16u / elem_size) + i;
                auto byte_idx = idx * elem_size;
                auto v = buf_in->read<ST>(off_in + byte_idx);
                buf_out->write(off_out + byte_idx, v);
            }
            return;
        }
    }

    // Scalar fallback for non-ByteBuffer or any remaining edge cases
    for (auto i : dynamic_range(static_cast<uint>(out.size()))) {
        out[i] = in[i];
    }
}

}// namespace detail

// Squeeze: removes dimensions of size 1 from the shape.
// ONNX spec (opset 13+): axes is a second input tensor (not an attribute).
//   If axes not provided, remove all dims of size 1.
class Squeeze : public Operator {
public:
    Squeeze() : Operator("Squeeze") {}

    bool is_output_view([[maybe_unused]] size_t output_index,
                        [[maybe_unused]] onnx::Node const &node) const override { return true; }

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() >= 1 && inputs.size() <= 2 && outputs.size() == 1,
                     "Squeeze requires 1-2 inputs and 1 output.");
#endif
        auto &input = inputs[0].get();
        auto &output = outputs[0].get();
#ifndef NDEBUG
        LUISA_ASSERT(input.size() == output.size(),
                     "Squeeze: input and output must have the same total size.");
        LUISA_ASSERT(input.element_type() == output.element_type(),
                     "Squeeze: input and output must have the same element type.");
#endif

        visit_typeid<NNTypeList>(input.element_type(), [&]<typename T>() {
            auto &in = static_cast<NNTensor<T> &>(input);
            auto &out = static_cast<NNTensor<T> &>(output);
            detail::squeeze_copy(in, out);
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(Squeeze) {
    return std::make_unique<Squeeze>();
};

// Unsqueeze: inserts a dimension of size 1 at the specified axes.
// ONNX spec (opset 13+): axes is a second input tensor.
class Unsqueeze : public Operator {
public:
    Unsqueeze() : Operator("Unsqueeze") {}

    bool is_output_view([[maybe_unused]] size_t output_index,
                        [[maybe_unused]] onnx::Node const &node) const override { return true; }

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 2 && outputs.size() == 1,
                     "Unsqueeze requires 2 inputs and 1 output.");
#endif
        auto &input = inputs[0].get();
        auto &output = outputs[0].get();
#ifndef NDEBUG
        LUISA_ASSERT(input.size() == output.size(),
                     "Unsqueeze: input and output must have the same total size.");
        LUISA_ASSERT(input.element_type() == output.element_type(),
                     "Unsqueeze: input and output must have the same element type.");
#endif

        visit_typeid<NNTypeList>(input.element_type(), [&]<typename T>() {
            auto &in = static_cast<NNTensor<T> &>(input);
            auto &out = static_cast<NNTensor<T> &>(output);
            detail::squeeze_copy(in, out);
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(Unsqueeze) {
    return std::make_unique<Unsqueeze>();
};

}// namespace lcml::onnx
