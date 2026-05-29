#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

namespace lcml::onnx {

// Pad: pads a tensor with a constant, edge, or reflect value.
// ONNX spec (opset 11+): inputs: data, pads, [constant_value]; attribute mode (default "constant")
class Pad : public Operator {
private:
    std::string mode_;

public:
    Pad(std::string mode) : Operator("Pad"), mode_(std::move(mode)) {}

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() >= 2 && outputs.size() == 1, "Pad requires >=2 inputs and 1 output.");
        auto &data = inputs[0].get();
        auto &output = outputs[0].get();
        auto ndim = data.ndim();

        LUISA_ASSERT(data.element_type() == output.element_type(), "Pad: data and output must have the same element type.");
#else
        auto &data = inputs[0].get();
        auto &output = outputs[0].get();
        auto ndim = data.ndim();
#endif

        // Read pad values from the pads tensor (may be int32 or int64).
        // Both NNTensor<slong> and NNTensor<int> have storage type int
        // (nn_storage_type_t<slong> = int), so we can safely use NNTensor<int>.
        // pads tensor layout: [dim0_begin, dim1_begin, ..., dimN_begin, dim0_end, dim1_end, ..., dimN_end]
        auto &pads_t = static_cast<NNTensor<int> &>(inputs[1].get());

        // Fast path: all-zero pads -> vectorized copy (avoids coordinate decomposition)
        bool all_zero_pads = false;
        if (inputs[1].get().is_constant()) {
            auto &pads_const = static_cast<NNConstTensor<int> const &>(inputs[1].get());
            all_zero_pads = true;
            for (size_t i = 0; i < pads_const.const_data().size(); ++i) {
                if (pads_const.const_data()[i] != 0) {
                    all_zero_pads = false;
                    break;
                }
            }
        }

        visit_typeid<NNTypeList>(data.element_type(), [&]<typename T>() {
            using VT = nn_storage_type_t<T>;
            auto &in = static_cast<NNTensor<T> &>(data);
            auto &out = static_cast<NNTensor<T> &>(output);
            bool strides_equal = (in.strides() == out.strides());

            // Helper: vectorized contiguous copy with element offset on the input side.
            auto vectorized_copy_offset = [&](Var<uint> off) {
                if constexpr (detail::VecDispatch<VT>::supported) {
                    if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                        using VecT = typename detail::VecDispatch<VT>::VecT;
                        constexpr uint32_t N = detail::VecDispatch<VT>::N;
                        auto buf_in = in.container().get_byte_buffer();
                        auto buf_o = out.container().get_byte_buffer();
                        auto off_in = static_cast<uint>(in.container().get_byte_offset()) +
                                      off * static_cast<uint>(sizeof(VT));
                        auto off_o = static_cast<uint>(out.container().get_byte_offset());
                        auto n = in.size();
                        auto vec_count = static_cast<uint>(n / N);
                        auto rem = static_cast<uint>(n % N);
                        for (auto i : dynamic_range(vec_count)) {
                            auto byte_idx = i * static_cast<uint>(sizeof(VecT));
                            auto v = buf_in->read<VecT>(off_in + byte_idx);
                            buf_o->write(off_o + byte_idx, v);
                        }
                        for (auto i : dynamic_range(rem)) {
                            auto idx = vec_count * N + i;
                            auto v = buf_in->read<VT>(off_in + idx * static_cast<uint>(sizeof(VT)));
                            buf_o->write(off_o + idx * static_cast<uint>(sizeof(VT)), v);
                        }
                        return;
                    }
                } else if constexpr (std::is_same_v<T, FP4E2M1>) {
                    if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                        auto buf_in = in.container().get_byte_buffer();
                        auto buf_out = out.container().get_byte_buffer();
                        auto off_in = static_cast<uint>(in.container().get_byte_offset()) +
                                      off / 8u * 4u;
                        auto off_out = static_cast<uint>(out.container().get_byte_offset());
                        auto n = static_cast<uint>(in.size());
                        auto word_count = (n + 7u) / 8u;
                        for (auto i : dynamic_range(word_count)) {
                            auto v = buf_in->read<uint>(off_in + i * 4u);
                            buf_out->write(off_out + i * 4u, v);
                        }
                        return;
                    }
                } else if constexpr (std::is_same_v<T, FP8E4M3FN> || std::is_same_v<T, FP8E5M2>) {
                    if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                        auto buf_in = in.container().get_byte_buffer();
                        auto buf_out = out.container().get_byte_buffer();
                        auto off_in = static_cast<uint>(in.container().get_byte_offset()) +
                                      off / 4u * 4u;
                        auto off_out = static_cast<uint>(out.container().get_byte_offset());
                        auto n = static_cast<uint>(in.size());
                        auto word_count = (n + 3u) / 4u;
                        for (auto i : dynamic_range(word_count)) {
                            auto v = buf_in->read<uint>(off_in + i * 4u);
                            buf_out->write(off_out + i * 4u, v);
                        }
                        return;
                    }
                }
                // Scalar fallback for non-vectorizable types or non-ByteBuffer storage.
                for (auto i : dynamic_range(static_cast<uint>(in.size()))) {
                    out[i] = in[i + off];
                }
            };

            // Zero-pad fast path: vectorized direct copy
            if (all_zero_pads) {
                if (!in.container().shares_storage_with(out.container())) {
                    vectorized_copy_offset(def(0u));
                }
                return;
            }

            // Read constant_value if provided (for constant mode)
            Var<VT> const_val;
            // Var<T> default-constructs to zero for all types (including FP quantized)
            if (inputs.size() >= 3 && inputs[2].get().size() > 0) {
                auto &cv = static_cast<NNTensor<T> &>(inputs[2].get());
                const_val = cv[0u];
            }

            auto const &in_shape = in.shape();
            auto const &out_shape = out.shape();

            if (mode_ == "constant") {
                // Initialize output with constant value
                if constexpr (detail::VecDispatch<VT>::supported) {
                    if (out.container().is_byte_buffer()) {
                        detail::vectorized_scalar_b<VT>(
                            out, const_val, out,
                            [](auto, auto b) { return b; });
                    } else {
                        for (auto i : dynamic_range(out.size())) {
                            out[i] = const_val;
                        }
                    }
                } else {
                    for (auto i : dynamic_range(out.size())) {
                        out[i] = const_val;
                    }
                }
                // Copy data into the padded region
                if (strides_equal) {
                    // When strides match, out_linear = in_linear + base_offset
                    auto base_offset = def(0u);
                    for (uint32_t d = 0; d < ndim; ++d) {
                        base_offset += pads_t[d].cast<uint>() * out.strides()[d];
                    }
                    // vectorized_copy_offset copies in[i+off] -> out[i],
                    // but Pad needs in[i] -> out[i+off]. Use direct loop.
                    for (auto i : dynamic_range(static_cast<uint>(in.size()))) {
                        out[i + base_offset] = in[i];
                    }
                } else {
                    if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                        auto buf_in = in.container().get_byte_buffer();
                        auto buf_out = out.container().get_byte_buffer();
                        auto off_in = static_cast<uint>(in.container().get_byte_offset());
                        auto off_out = static_cast<uint>(out.container().get_byte_offset());
                        for (auto linear_in : dynamic_range(in.size())) {
                            auto out_linear = def(0u);
                            for_each_dim(linear_in, in.strides(), ndim, [&](uint32_t d, auto coord) {
                                auto pad_val = pads_t[d].cast<uint>();
                                out_linear += (coord + pad_val) * out.strides()[d]; }, in.size());
                            auto v = buf_in->read<VT>(off_in + linear_in * static_cast<uint>(sizeof(VT)));
                            buf_out->write(off_out + out_linear * static_cast<uint>(sizeof(VT)), v);
                        }
                    } else {
                        for (auto linear_in : dynamic_range(in.size())) {
                            auto out_linear = def(0u);
                            for_each_dim(linear_in, in.strides(), ndim, [&](uint32_t d, auto coord) {
                                auto pad_val = pads_t[d].cast<uint>();
                                out_linear += (coord + pad_val) * out.strides()[d]; }, in.size());
                            out[out_linear] = in[linear_in];
                        }
                    }
                }
            } else if (mode_ == "edge") {
                // Edge padding: clamp source coordinates to [0, dim_size - 1]
                if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                    auto buf_in = in.container().get_byte_buffer();
                    auto buf_out = out.container().get_byte_buffer();
                    auto off_in = static_cast<uint>(in.container().get_byte_offset());
                    auto off_out = static_cast<uint>(out.container().get_byte_offset());
                    for (auto linear_out : dynamic_range(out.size())) {
                        auto in_linear = def(0u);
                        for_each_dim(linear_out, out.strides(), ndim, [&](uint32_t d, auto coord) {
                            // src_coord = coord - pad_begin[d], then clamp to [0, in_shape[d]-1]
                            auto pad_val = pads_t[d].cast<int>();
                            auto src = coord.cast<int>() - pad_val;
                            auto dim_size = def(static_cast<int>(in_shape[d]));
                            auto clamped = max(min(src, dim_size - 1), def(0));
                            in_linear += clamped.cast<uint>() * in.strides()[d]; }, out.size());
                        auto v = buf_in->read<VT>(off_in + in_linear * static_cast<uint>(sizeof(VT)));
                        buf_out->write(off_out + linear_out * static_cast<uint>(sizeof(VT)), v);
                    }
                } else {
                    for (auto linear_out : dynamic_range(out.size())) {
                        auto in_linear = def(0u);
                        for_each_dim(linear_out, out.strides(), ndim, [&](uint32_t d, auto coord) {
                            // src_coord = coord - pad_begin[d], then clamp to [0, in_shape[d]-1]
                            auto pad_val = pads_t[d].cast<int>();
                            auto src = coord.cast<int>() - pad_val;
                            auto dim_size = def(static_cast<int>(in_shape[d]));
                            auto clamped = max(min(src, dim_size - 1), def(0));
                            in_linear += clamped.cast<uint>() * in.strides()[d]; }, out.size());
                        out[linear_out] = in[in_linear];
                    }
                }
            } else if (mode_ == "reflect") {
                // Reflect padding: reflect source coordinates into [0, dim_size - 1]
                if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                    auto buf_in = in.container().get_byte_buffer();
                    auto buf_out = out.container().get_byte_buffer();
                    auto off_in = static_cast<uint>(in.container().get_byte_offset());
                    auto off_out = static_cast<uint>(out.container().get_byte_offset());
                    for (auto linear_out : dynamic_range(out.size())) {
                        auto in_linear = def(0u);
                        for_each_dim(linear_out, out.strides(), ndim, [&](uint32_t d, auto coord) {
                            auto pad_val = pads_t[d].cast<int>();
                            auto src = coord.cast<int>() - pad_val;
                            auto dim = def(static_cast<int>(in_shape[d]));
                            // Reflect: for dim_size > 1, fold src into [0, dim_size-1]
                            // Using periodic reflection: period = 2*(dim_size-1)
                            if (in_shape[d] > 1) {
                                auto period = def(static_cast<int>(2 * (in_shape[d] - 1)));
                                // Make positive: src = src % period, then handle negative
                                auto mod_src = src % period;
                                // If negative, add period
                                mod_src = select(mod_src + period, mod_src, mod_src >= 0);
                                // Fold: if mod_src >= dim_size, reflect back
                                auto reflected = select(period - mod_src, mod_src, mod_src < dim);
                                in_linear += reflected.cast<uint>() * in.strides()[d];
                            } else {
                                // dim_size == 1: always index 0
                                // in_linear += 0
                            } }, out.size());
                        auto v = buf_in->read<VT>(off_in + in_linear * static_cast<uint>(sizeof(VT)));
                        buf_out->write(off_out + linear_out * static_cast<uint>(sizeof(VT)), v);
                    }
                } else {
                    for (auto linear_out : dynamic_range(out.size())) {
                        auto in_linear = def(0u);
                        for_each_dim(linear_out, out.strides(), ndim, [&](uint32_t d, auto coord) {
                            auto pad_val = pads_t[d].cast<int>();
                            auto src = coord.cast<int>() - pad_val;
                            auto dim = def(static_cast<int>(in_shape[d]));
                            // Reflect: for dim_size > 1, fold src into [0, dim_size-1]
                            // Using periodic reflection: period = 2*(dim_size-1)
                            if (in_shape[d] > 1) {
                                auto period = def(static_cast<int>(2 * (in_shape[d] - 1)));
                                // Make positive: src = src % period, then handle negative
                                auto mod_src = src % period;
                                // If negative, add period
                                mod_src = select(mod_src + period, mod_src, mod_src >= 0);
                                // Fold: if mod_src >= dim_size, reflect back
                                auto reflected = select(period - mod_src, mod_src, mod_src < dim);
                                in_linear += reflected.cast<uint>() * in.strides()[d];
                            } else {
                                // dim_size == 1: always index 0
                                // in_linear += 0
                            } }, out.size());
                        out[linear_out] = in[in_linear];
                    }
                }
            } else {
                LUISA_ASSERT(false, "Pad: mode '{}' is not supported.", mode_);
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(Pad) {
    std::string mode = "constant";
    if (auto p = node.try_get_attr("mode"))
        mode = p->get<onnx::AttributeType::STRING>();
    return std::make_unique<Pad>(std::move(mode));
};

}// namespace lcml::onnx
