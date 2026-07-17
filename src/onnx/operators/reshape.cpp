#include <luisa/core/stl/memory.h>
#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

namespace lcml::onnx {

// Reshape: gives a new shape to the data without changing its values.
// ONNX spec: input[0] = data, input[1] = shape (1-D int64 tensor)
// allowzero attribute (default 0): if 0, dim=0 means copy from input; if 1, dim=0 means actual 0
class Reshape : public Operator {
private:
    int32_t allowzero_;

public:
    Reshape(int32_t allowzero) : Operator("Reshape"), allowzero_(allowzero) {}

    bool is_output_view([[maybe_unused]] size_t output_index,
                        [[maybe_unused]] onnx::Node const &node) const override { return true; }

    bool can_operate_inplace() const override { return true; }

    bool need_outline() const override { return false; }

    void forward(luisa::span<std::reference_wrapper<ITensor>> inputs,
                 luisa::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() >= 1 && inputs.size() <= 2 && outputs.size() == 1, "Reshape requires 1 or 2 inputs and 1 output.");
#endif
        auto &data = inputs[0].get();
        auto &output = outputs[0].get();
#ifndef NDEBUG
        LUISA_ASSERT(data.size() == output.size(), "Reshape: input and output must have the same total size.");
        LUISA_ASSERT(data.element_type_index() == output.element_type_index(), "Reshape: input and output must have the same element type.");
#endif

        visit_type_index<NNTypeList>(data.element_type_index(), [&]<typename T>() {
            auto &in = static_cast<NNTensor<T> &>(data);
            auto &out = static_cast<NNTensor<T> &>(output);
            // If already sharing storage (inplace allocation), skip assignment
            if (in.container().shares_storage_with(out.container())) return;

            using ST = nn_storage_type_t<T>;
            if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                auto buf_in = in.container().get_byte_buffer();
                auto buf_out = out.container().get_byte_buffer();
                auto off_in = static_cast<uint>(in.container().get_byte_offset());
                auto off_out = static_cast<uint>(out.container().get_byte_offset());
                auto n = static_cast<uint>(out.size());

                if constexpr (std::is_same_v<T, FP4E2M1>) {
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
                } else if constexpr (std::is_same_v<T, FP8E4M3FN> || std::is_same_v<T, FP8E5M2>) {
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
                    // Generic raw-byte vectorized copy for all other types.
                    // sizeof(ST) is guaranteed to divide 16 (valid values are 1, 2, 4).
                    auto byte_size = n * static_cast<uint>(sizeof(ST));
                    auto vec4_count = byte_size / 16u;
                    auto rem_bytes = byte_size % 16u;
                    for (auto i : dynamic_range(vec4_count)) {
                        auto v = buf_in->read<uint4>(off_in + i * 16u);
                        buf_out->write(off_out + i * 16u, v);
                    }
                    auto rem_off = vec4_count * 16u;
                    auto word_count = rem_bytes / 4u;
                    for (auto i : dynamic_range(word_count)) {
                        auto v = buf_in->read<uint>(off_in + rem_off + i * 4u);
                        buf_out->write(off_out + rem_off + i * 4u, v);
                    }
                    rem_off += word_count * 4u;
                    rem_bytes -= word_count * 4u;
                    if (rem_bytes > 0u) {
                        auto rem_start = rem_off / static_cast<uint>(sizeof(ST));
                        auto rem_elem = rem_bytes / static_cast<uint>(sizeof(ST));
                        for (auto i : dynamic_range(rem_elem)) {
                            auto idx = rem_start + i;
                            auto v = buf_in->read<ST>(off_in + idx * static_cast<uint>(sizeof(ST)));
                            buf_out->write(off_out + idx * static_cast<uint>(sizeof(ST)), v);
                        }
                    }
                }
                return;
            }

            // Non-byte-buffer fallback: scalar copy via DynamicArray indexing
            for (auto i : dynamic_range(static_cast<uint>(out.size()))) {
                out[i] = in[i];
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(Reshape) {
    int32_t allowzero = 0;
    if (auto p = node.try_get_attr("allowzero"))
        allowzero = p->get<onnx::AttributeType::INT>();
    return luisa::make_unique<Reshape>(allowzero);
};

}// namespace lcml::onnx
