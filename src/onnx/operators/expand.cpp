#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

namespace lcml::onnx {

// Expand: broadcasts input tensor to output shape (NumPy broadcasting).
// ONNX spec: input[0] = data, input[1] = shape tensor
class Expand : public Operator {
public:
    Expand() : Operator("Expand") {}

    /// When input and output shapes are identical, Expand is a no-op view.
    bool is_output_view([[maybe_unused]] size_t output_index,
                        [[maybe_unused]] onnx::Node const &node) const override {
        auto const &inputs = node.get_inputs();
        if (inputs.size() < 1) return false;
        auto const &data_var = inputs[0].get();
        auto const &out_vars = node.get_outputs();
        if (out_vars.empty()) return false;
        auto const &out_var = out_vars[0].get();
        // Same-shape check: if input and output have identical shape, it's a view
        return data_var.get_shape() == out_var.get_shape();
    }

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
        LUISA_ASSERT(inputs.size() == 2 && outputs.size() == 1,
                     "Expand requires 2 inputs and 1 output.");
        auto &data = inputs[0].get();
        auto &output = outputs[0].get();

        LUISA_ASSERT(data.element_type() == output.element_type(),
                     "Expand: input and output must have the same element type.");

        auto const &in_shape = data.shape();
        auto const &out_shape = output.shape();
        auto out_ndim = out_shape.size();

        // Fast path: same-shape → output is already a view of input (set up by allocator)
        if (in_shape == out_shape) {
            visit_typeid<NNTypeList>(data.element_type(), [&]<typename T>() {
                auto &in = static_cast<NNTensor<T> &>(data);
                auto &out = static_cast<NNTensor<T> &>(output);
                // If not already sharing storage, copy
                if (!in.container().shares_storage_with(out.container())) {
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
                    } else {
                        for (auto i : dynamic_range(static_cast<uint>(out.size()))) {
                            out[i] = in[i];
                        }
                    }
                }
            });
            return;
        }

        // Precompute padded input shape and strides for broadcasting
        std::vector<uint32_t> padded(out_ndim, 1u);
        for (size_t i = 0; i < in_shape.size(); ++i)
            padded[out_ndim - in_shape.size() + i] = in_shape[i];

        std::vector<uint32_t> in_stride(out_ndim, 0u);
        {
            uint32_t acc = 1;
            for (int i = static_cast<int>(out_ndim) - 1; i >= 0; --i) {
                in_stride[i] = (padded[i] == 1) ? 0u : acc;
                acc *= padded[i];
            }
        }

        visit_typeid<NNTypeList>(data.element_type(), [&]<typename T>() {
            auto &in = static_cast<NNTensor<T> &>(data);
            auto &out = static_cast<NNTensor<T> &>(output);

#ifndef NDEBUG
            if constexpr (std::is_same_v<T, FP4E2M1> || std::is_same_v<T, FP8E4M3FN> || std::is_same_v<T, FP8E5M2>) {
                LUISA_ASSERT(!in.container().is_byte_buffer() && !out.container().is_byte_buffer(),
                             "Expand broadcast for FP4/FP8 ByteBuffer is not yet supported");
            }
#endif

            using ST = nn_storage_type_t<T>;
            if constexpr (IsNativeArithmetic<T>::value && detail::VecDispatch<ST>::supported) {
                bool use_vec = in.container().is_byte_buffer() && out.container().is_byte_buffer();
                if (use_vec && out_ndim > 0) {
                    use_vec = out.strides()[out_ndim - 1] == 1;
                }

                if (use_vec) {
                    detail::vectorized_broadcast_unary<ST>(
                        in, out, out_shape, static_cast<uint32_t>(out_ndim), in_stride,
                        [](auto v) { return v; });
                } else {
                    for (auto linear : dynamic_range(static_cast<uint>(out.size()))) {
                        auto idx_in = def(0u);
                        for_each_dim(linear, out.strides(), static_cast<uint32_t>(out_ndim), [&](size_t d, auto coord) { idx_in += coord * in_stride[d]; }, static_cast<uint32_t>(out.size()));
                        out[linear] = in[idx_in];
                    }
                }
            } else {
                for (auto linear : dynamic_range(static_cast<uint>(out.size()))) {
                    auto idx_in = def(0u);
                    for_each_dim(linear, out.strides(), static_cast<uint32_t>(out_ndim), [&](size_t d, auto coord) { idx_in += coord * in_stride[d]; }, static_cast<uint32_t>(out.size()));
                    out[linear] = in[idx_in];
                }
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(Expand) {
    return std::make_unique<Expand>();
};

}// namespace lcml::onnx
