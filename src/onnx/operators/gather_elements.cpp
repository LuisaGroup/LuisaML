#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/memory.h>

namespace lcml::onnx {

// GatherElements: gathers values along an axis determined by index values.
// ONNX spec: input[0]=data, input[1]=indices (same rank as data); attribute axis (default 0)
class GatherElements : public Operator {
private:
    int64_t axis_;

public:
    GatherElements(int64_t axis) : Operator("GatherElements"), axis_(axis) {}

    void forward(luisa::span<std::reference_wrapper<ITensor>> inputs,
                 luisa::span<std::reference_wrapper<ITensor>> outputs) override {
        LUISA_ASSERT(inputs.size() == 2 && outputs.size() == 1,
                     "GatherElements requires 2 inputs and 1 output.");
        auto &data = inputs[0].get();
        auto &indices = inputs[1].get();
        auto &output = outputs[0].get();

        auto ndim = data.ndim();
        int64_t axis = axis_ < 0 ? axis_ + ndim : axis_;

        LUISA_ASSERT(data.element_type_index() == output.element_type_index(),
                     "GatherElements: data and output must have the same element type.");
        LUISA_ASSERT(indices.element_type_index() == refl::type_index_of<int32_t>() || indices.element_type_index() == refl::type_index_of<slong>(),
                     "GatherElements: indices must be int32_t or int64 type.");

#ifndef NDEBUG
        LUISA_ASSERT(data.ndim() == indices.ndim(),
                     "GatherElements: data and indices must have the same rank.");
        for (uint32_t d = 0; d < ndim; ++d) {
            if (d != static_cast<uint32_t>(axis)) {
                LUISA_ASSERT(data.shape()[d] == indices.shape()[d],
                             "GatherElements: non-axis dimensions must match.");
            }
        }
#endif

        visit_type_index<NNTypeList>(data.element_type_index(), [&]<typename T>() {
            auto &in = static_cast<NNTensor<T> &>(data);
            auto &idx_tensor = static_cast<NNTensor<int32_t> &>(indices);
            auto &out = static_cast<NNTensor<T> &>(output);
            using ST = nn_storage_type_t<T>;

            // Fast path: constant indices -> precompute gather positions at C++ time
            if (indices.is_constant() && indices.size() <= 128) {
                auto &idx_const = static_cast<NNConstTensor<int32_t> const &>(indices);
                auto const &cpu_idx = idx_const.const_data();

                // Precompute data offsets for every output element
                luisa::vector<uint32_t> data_offsets;
                data_offsets.reserve(out.size());
                for (uint32_t linear_out = 0; linear_out < out.size(); ++linear_out) {
                    uint32_t data_linear = 0;
                    uint32_t remaining = linear_out;
                    for (uint32_t d = 0; d < ndim; ++d) {
                        uint32_t coord = remaining / out.strides()[d];
                        remaining = remaining % out.strides()[d];
                        if (d == static_cast<uint32_t>(axis)) {
                            int32_t idx_val = static_cast<int32_t>(cpu_idx[linear_out]);
                            if (idx_val < 0) idx_val += static_cast<int32_t>(data.shape()[d]);
                            data_linear += static_cast<uint32_t>(idx_val) * in.strides()[d];
                        } else {
                            data_linear += coord * in.strides()[d];
                        }
                    }
                    data_offsets.push_back(data_linear);
                }

                // Emit vectorized reads/writes for contiguous runs
                if constexpr (detail::VecDispatch<ST>::supported) {
                    if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                        using VecT = typename detail::VecDispatch<ST>::VecT;
                        constexpr uint32_t N = detail::VecDispatch<ST>::N;
                        auto buf_in = in.container().get_byte_buffer();
                        auto buf_o = out.container().get_byte_buffer();
                        auto off_in_base = static_cast<uint>(in.container().get_byte_offset());
                        auto off_o_base = static_cast<uint>(out.container().get_byte_offset());
                        auto elem_size = static_cast<uint>(sizeof(ST));

                        uint32_t i = 0;
                        while (i < out.size()) {
                            uint32_t run_start = i;
                            uint32_t run_len = 1;
                            ++i;
                            while (i < out.size() &&
                                   data_offsets[i] == data_offsets[i - 1] + 1 &&
                                   (i - run_start) < 1024u) {
                                ++run_len;
                                ++i;
                            }

                            uint32_t vec_count = run_len / N;
                            uint32_t rem = run_len % N;

                            auto data_byte_off = off_in_base + data_offsets[run_start] * elem_size;
                            auto out_byte_off = off_o_base + run_start * elem_size;

                            for (uint32_t v = 0; v < vec_count; ++v) {
                                auto vb = v * static_cast<uint>(sizeof(VecT));
                                auto val = buf_in->read<VecT>(data_byte_off + vb);
                                buf_o->write(out_byte_off + vb, val);
                            }
                            for (uint32_t r = 0; r < rem; ++r) {
                                auto idx = run_start + vec_count * N + r;
                                auto val = buf_in->read<ST>(off_in_base + data_offsets[idx] * elem_size);
                                buf_o->write(off_o_base + idx * elem_size, val);
                            }
                        }
                        return;
                    }
                }

                // ByteBuffer scalar fast path
                if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                    auto buf_in = in.container().get_byte_buffer();
                    auto buf_o = out.container().get_byte_buffer();
                    auto off_in = static_cast<uint>(in.container().get_byte_offset());
                    auto off_o = static_cast<uint>(out.container().get_byte_offset());
                    auto elem_size = static_cast<uint>(sizeof(ST));
                    for (uint32_t linear_out = 0; linear_out < out.size(); ++linear_out) {
                        auto val = buf_in->read<ST>(off_in + data_offsets[linear_out] * elem_size);
                        buf_o->write(off_o + linear_out * elem_size, val);
                    }
                    return;
                }

                // Scalar fallback
                for (uint32_t linear_out = 0; linear_out < out.size(); ++linear_out) {
                    out[linear_out] = in[data_offsets[linear_out]];
                }
                return;
            }

            // General path
            bool use_bb_scalar = in.container().is_byte_buffer() && out.container().is_byte_buffer();
            if (use_bb_scalar) {
                auto buf_in = in.container().get_byte_buffer();
                auto buf_o = out.container().get_byte_buffer();
                auto off_in = static_cast<uint>(in.container().get_byte_offset());
                auto off_o = static_cast<uint>(out.container().get_byte_offset());
                auto elem_size = static_cast<uint>(sizeof(ST));
                for (auto linear_out : dynamic_range(out.size())) {
                    auto data_linear = def(0u);
                    // Hoisted index load with negative-index normalization
                    auto idx_signed = idx_tensor[linear_out];
                    auto dim_size = static_cast<int32_t>(in.shape()[axis]);
                    auto idx_norm = select(idx_signed + dim_size, idx_signed, idx_signed < 0);

                    for_each_dim(linear_out, out.strides(), ndim, [&](uint32_t d, auto coord) {
                        if (d == static_cast<uint32_t>(axis)) {
                            data_linear += idx_norm.cast<uint>() * in.strides()[d];
                        } else {
                            data_linear += coord * in.strides()[d];
                        }
                    }, out.size());
                    auto val = buf_in->read<ST>(off_in + data_linear * elem_size);
                    buf_o->write(off_o + linear_out * elem_size, val);
                }
                return;
            }

            for (auto linear_out : dynamic_range(out.size())) {
                auto data_linear = def(0u);
                // Hoisted index load with negative-index normalization
                auto idx_signed = idx_tensor[linear_out];
                auto dim_size = static_cast<int32_t>(in.shape()[axis]);
                auto idx_norm = select(idx_signed + dim_size, idx_signed, idx_signed < 0);

                for_each_dim(linear_out, out.strides(), ndim, [&](uint32_t d, auto coord) {
                    if (d == static_cast<uint32_t>(axis)) {
                        data_linear += idx_norm.cast<uint>() * in.strides()[d];
                    } else {
                        data_linear += coord * in.strides()[d];
                    }
                }, out.size());
                out[linear_out] = in[data_linear];
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(GatherElements) {
    int64_t axis = 0;
    if (auto p = node.try_get_attr("axis"))
        axis = p->get<onnx::AttributeType::INT>();
    return luisa::make_unique<GatherElements>(axis);
};

}// namespace lcml::onnx