#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

namespace lcml::onnx {

// Gather: takes slices of data along the given axis, at positions specified by indices.
// ONNX spec: input[0]=data, input[1]=indices; attribute axis (default 0)
class Gather : public Operator {
private:
    int64_t axis_;

public:
    Gather(int64_t axis) : Operator("Gather"), axis_(axis) {}

    /// Aggressive compile-time view detection for Gather.
    /// Gather produces a view when axis==0 and indices are constant and
    /// either a single scalar or consecutive [S, S+1, ..., S+N-1].
    bool is_output_view([[maybe_unused]] size_t output_index,
                        [[maybe_unused]] onnx::Node const &node) const override {
        // Only axis==0 fast path produces views
        auto const &inputs = node.get_inputs();
        if (inputs.size() != 2) return false;
        auto const &data_var = inputs[0].get();
        auto const &idx_var = inputs[1].get();
        auto ndim = data_var.get_shape().size();
        int64_t axis = axis_ < 0 ? axis_ + static_cast<int64_t>(ndim) : axis_;
        if (axis != 0) return false;

        // Indices must be constant with raw_data available
        if (!idx_var.is_constant() || idx_var.get_raw_data().empty()) return false;

        // Parse constant int indices
        auto const &raw = idx_var.get_raw_data();
        size_t num_indices = 1;
        for (auto d : idx_var.get_shape()) num_indices *= d;
        if (num_indices == 0) return false;

        std::vector<int> indices(num_indices);
        if (idx_var.get_dtype() == onnx::DataType::INT64) {
            auto const *src = reinterpret_cast<int64_t const *>(raw.data());
            for (size_t i = 0; i < num_indices; ++i) indices[i] = static_cast<int>(src[i]);
        } else {
            auto const *src = reinterpret_cast<int32_t const *>(raw.data());
            for (size_t i = 0; i < num_indices; ++i) indices[i] = src[i];
        }

        int dim0 = static_cast<int>(data_var.get_shape()[0]);

        // Single scalar index -> always a view
        if (num_indices == 1) return true;

        // Check consecutive indices -> contiguous sub-region -> view
        int start_val = indices[0];
        if (start_val < 0) start_val += dim0;
        if (start_val < 0 || start_val >= dim0) return false;
        for (size_t i = 1; i < num_indices; ++i) {
            int v = indices[i];
            if (v < 0) v += dim0;
            if (v != start_val + static_cast<int>(i)) return false;
        }
        return true;
    }

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
        LUISA_ASSERT(inputs.size() == 2 && outputs.size() == 1,
                     "Gather requires 2 inputs and 1 output.");
        auto &data = inputs[0].get();
        auto &indices = inputs[1].get();
        auto &output = outputs[0].get();

        auto const &data_shape = data.shape();
        auto ndim = data.ndim();
        int64_t axis = axis_ < 0 ? axis_ + ndim : axis_;

        LUISA_ASSERT(data.element_type() == output.element_type(),
                     "Gather: data and output must have the same element type.");
        LUISA_ASSERT(indices.element_type() == typeid(int) || indices.element_type() == typeid(slong),
                     "Gather: indices must be int or int64 type.");

        visit_typeid<NNTypeList>(data.element_type(), [&]<typename T>() {
            auto &in = static_cast<NNTensor<T> &>(data);
            auto &idx_tensor = static_cast<NNTensor<int> &>(indices);
            auto &out = static_cast<NNTensor<T> &>(output);
            auto const &out_shape = out.shape();
            auto out_ndim = out.ndim();
            using ST = nn_storage_type_t<T>;

            // Helper for vectorized contiguous copy with an element offset on the input side.
            auto vectorized_copy_offset = [&](uint32_t off) {
                if constexpr (detail::VecDispatch<ST>::supported) {
                    if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                        using VecT = typename detail::VecDispatch<ST>::VecT;
                        constexpr uint32_t N = detail::VecDispatch<ST>::N;
                        auto buf_in = in.container().get_byte_buffer();
                        auto buf_o = out.container().get_byte_buffer();
                        auto off_in = static_cast<uint>(in.container().get_byte_offset()) +
                                      off * static_cast<uint>(sizeof(ST));
                        auto off_o = static_cast<uint>(out.container().get_byte_offset());
                        auto n = out.size();
                        auto vec_count = static_cast<uint>(n / N);
                        auto rem = static_cast<uint>(n % N);
                        for (auto i : dynamic_range(vec_count)) {
                            auto byte_idx = i * static_cast<uint>(sizeof(VecT));
                            auto v = buf_in->read<VecT>(off_in + byte_idx);
                            buf_o->write(off_o + byte_idx, v);
                        }
                        for (auto i : dynamic_range(rem)) {
                            auto idx = vec_count * N + i;
                            out[idx] = in[idx + off];
                        }
                        return;
                    }
                }
                // Scalar fallback for non-vectorizable types or non-ByteBuffer storage.
                for (auto i : dynamic_range(static_cast<uint32_t>(out.size()))) {
                    out[i] = in[i + off];
                }
            };

            // Fast path: axis == 0, single constant scalar index -> explicit copy
            if (axis == 0 && indices.size() == 1 && indices.is_constant()) {
                auto &idx_const = static_cast<NNConstTensor<int> const &>(indices);
                int index_val = static_cast<int>(idx_const.const_data()[0]);
                if (index_val < 0) index_val += static_cast<int>(in.shape()[0]);
                size_t offset = static_cast<size_t>(index_val) * in.strides()[0];
                // If already sharing storage (inplace), skip
                if (!in.container().shares_storage_with(out.container())) {
                    vectorized_copy_offset(static_cast<uint32_t>(offset));
                }
                return;
            }

            // Fast path: axis == 0, constant consecutive indices [S, S+1, ..., S+N-1] -> View
            if (axis == 0 && indices.is_constant()) {
                auto &idx_const = static_cast<NNConstTensor<int> const &>(indices);
                auto const &cpu_idx = idx_const.const_data();
                auto num_indices = indices.size();
                if (num_indices > 0) {
                    int start_val = static_cast<int>(cpu_idx[0]);
                    if (start_val < 0) start_val += static_cast<int>(in.shape()[0]);
                    bool consecutive = start_val >= 0;
                    for (uint32_t i = 1; consecutive && i < num_indices; ++i) {
                        int v = static_cast<int>(cpu_idx[i]);
                        if (v < 0) v += static_cast<int>(in.shape()[0]);
                        if (v != start_val + static_cast<int>(i)) consecutive = false;
                    }
                    if (consecutive) {
                        size_t offset = static_cast<size_t>(start_val) * in.strides()[0];
                        // Explicit element-wise copy (no view)
                        if (!in.container().shares_storage_with(out.container())) {
                            vectorized_copy_offset(static_cast<uint32_t>(offset));
                        }
                        return;
                    }
                }
            }

            auto idx_ndim = indices.ndim();
            auto axis_u = static_cast<uint32_t>(axis);

            // Fast path: constant indices (any axis, small) -> precompute offsets and vectorize
            if (indices.is_constant() && indices.size() <= 128) {
                auto &idx_const = static_cast<NNConstTensor<int> const &>(indices);
                auto const &cpu_idx = idx_const.const_data();

                std::vector<uint32_t> data_offsets;
                data_offsets.reserve(out.size());
                for (uint32_t linear_out = 0; linear_out < out.size(); ++linear_out) {
                    uint32_t idx_linear = 0;
                    uint32_t final_in = 0;
                    uint32_t remaining = linear_out;
                    for (uint32_t d = 0; d < out_ndim; ++d) {
                        uint32_t coord = remaining / out.strides()[d];
                        remaining = remaining % out.strides()[d];
                        if (d < axis_u) {
                            final_in += coord * in.strides()[d];
                        } else if (d >= axis_u && d < axis_u + idx_ndim) {
                            auto idx_d = d - axis_u;
                            idx_linear += coord * indices.strides()[idx_d];
                        } else {
                            auto data_dim = d - idx_ndim + 1;
                            final_in += coord * in.strides()[data_dim];
                        }
                    }
                    int gathered_idx = static_cast<int>(cpu_idx[idx_linear]);
                    auto dim_size = static_cast<int>(in.shape()[axis]);
                    if (gathered_idx < 0) gathered_idx += dim_size;
                    final_in += static_cast<uint32_t>(gathered_idx) * in.strides()[axis];
                    data_offsets.push_back(final_in);
                }

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
                                out[idx] = in[data_offsets[idx]];
                            }
                        }
                        return;
                    }
                }

                // Scalar fallback for constant indices
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
                for (auto linear_out : dynamic_range(out.size())) {
                    auto idx_linear = def(0u);
                    auto final_in = def(0u);

                    for_each_dim(linear_out, out.strides(), out_ndim, [&](uint32_t d, auto coord) {
                        if (d < axis_u) {
                            final_in += coord * in.strides()[d];
                        } else if (d >= axis_u && d < axis_u + idx_ndim) {
                            auto idx_d = d - axis_u;
                            idx_linear += coord * indices.strides()[idx_d];
                        } else {
                            auto data_dim = d - idx_ndim + 1;
                            final_in += coord * in.strides()[data_dim];
                        }
                    }, out.size());

                    auto gathered_idx_signed = idx_tensor[idx_linear];
                    auto dim_size = static_cast<int>(in.shape()[axis]);
                    gathered_idx_signed = select(gathered_idx_signed + dim_size,
                                                 gathered_idx_signed,
                                                 gathered_idx_signed < 0);
                    auto gathered_idx = gathered_idx_signed.cast<uint>();

                    final_in += gathered_idx * in.strides()[axis];

                    auto val = buf_in->read<ST>(off_in + final_in * static_cast<uint>(sizeof(ST)));
                    buf_o->write(off_o + linear_out * static_cast<uint>(sizeof(ST)), val);
                }
                return;
            }
            for (auto linear_out : dynamic_range(out.size())) {
                auto idx_linear = def(0u);
                auto final_in = def(0u);

                for_each_dim(linear_out, out.strides(), out_ndim, [&](uint32_t d, auto coord) {
                    if (d < axis_u) {
                        final_in += coord * in.strides()[d];
                    } else if (d >= axis_u && d < axis_u + idx_ndim) {
                        auto idx_d = d - axis_u;
                        idx_linear += coord * indices.strides()[idx_d];
                    } else {
                        auto data_dim = d - idx_ndim + 1;
                        final_in += coord * in.strides()[data_dim];
                    }
                }, out.size());

                auto gathered_idx_signed = idx_tensor[idx_linear];
                auto dim_size = static_cast<int>(in.shape()[axis]);
                gathered_idx_signed = select(gathered_idx_signed + dim_size,
                                             gathered_idx_signed,
                                             gathered_idx_signed < 0);
                auto gathered_idx = gathered_idx_signed.cast<uint>();

                final_in += gathered_idx * in.strides()[axis];

                out[linear_out] = in[final_in];
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(Gather) {
    int64_t axis = 0;
    if (auto p = node.try_get_attr("axis"))
        axis = p->get<onnx::AttributeType::INT>();
    return std::make_unique<Gather>(axis);
};

}// namespace lcml::onnx
