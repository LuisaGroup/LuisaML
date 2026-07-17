#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"
#include <luisa/core/stl/memory.h>

namespace lcml::onnx {

// Flatten: flattens the input tensor into a 2D matrix.
// ONNX spec: attribute axis (default 1). Output shape: (product of dims 0..axis-1, product of dims axis..end)
class Flatten : public Operator {
private:
    int64_t axis_;

public:
    Flatten(int64_t axis) : Operator("Flatten"), axis_(axis) {}

    bool is_output_view([[maybe_unused]] size_t output_index,
                        [[maybe_unused]] onnx::Node const &node) const override { return true; }

    void forward(luisa::span<std::reference_wrapper<ITensor>> inputs,
                 luisa::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 1 && outputs.size() == 1,
                     "Flatten requires 1 input and 1 output.");
#endif
        auto &input = inputs[0].get();
        auto &output = outputs[0].get();
#ifndef NDEBUG
        LUISA_ASSERT(input.size() == output.size(),
                     "Flatten: input and output must have the same total size.");
        LUISA_ASSERT(input.element_type_index() == output.element_type_index(),
                     "Flatten: input and output must have the same element type.");
#endif

        visit_type_index<NNTypeList>(input.element_type_index(), [&]<typename T>() {
            auto &in = static_cast<NNTensor<T> &>(input);
            auto &out = static_cast<NNTensor<T> &>(output);
            // If already sharing storage (inplace allocation), skip assignment
            if (in.container().shares_storage_with(out.container())) return;
            using ST = nn_storage_type_t<T>;
            if constexpr (std::is_same_v<T, FP4E2M1>) {
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
            } else if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                // Generic raw byte vectorized copy for all other types.
                // Uses uint4 (16 bytes) chunks plus uint remainder words.
                // This is optimal for float/half/int32_t/uint/etc. and avoids the
                // ByteBuffer write-back issue in the scalar fallback path.
                auto buf_in = in.container().get_byte_buffer();
                auto buf_out = out.container().get_byte_buffer();
                auto off_in = static_cast<uint>(in.container().get_byte_offset());
                auto off_out = static_cast<uint>(out.container().get_byte_offset());
                auto n = static_cast<uint>(out.size());
                auto total_bytes = n * static_cast<uint>(sizeof(ST));
                auto vec4_count = total_bytes / 16u;
                auto rem_bytes = total_bytes % 16u;
                for (auto i : dynamic_range(vec4_count)) {
                    auto v = buf_in->read<uint4>(off_in + i * 16u);
                    buf_out->write(off_out + i * 16u, v);
                }
                if (rem_bytes > 0) {
                    auto rem_word_count = (rem_bytes + 3u) / 4u;
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
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(Flatten) {
    int64_t axis = 1;
    if (auto p = node.try_get_attr("axis"))
        axis = p->get<onnx::AttributeType::INT>();
    return luisa::make_unique<Flatten>(axis);
};

}// namespace lcml::onnx