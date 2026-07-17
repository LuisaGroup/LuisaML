#include <luisa/core/stl/memory.h>
#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

namespace lcml::onnx {

// Tile: constructs a tensor by tiling the input tensor.
// ONNX spec: input[0]=data, input[1]=repeats (1-D int64 tensor, one value per dimension)
class Tile : public Operator {
public:
    Tile() : Operator("Tile") {}

    void forward(luisa::span<std::reference_wrapper<ITensor>> inputs,
                 luisa::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 2 && outputs.size() == 1, "Tile requires 2 inputs and 1 output.");
#endif
        auto &data = inputs[0].get();
        auto &output = outputs[0].get();
        auto ndim = data.ndim();

#ifndef NDEBUG
        LUISA_ASSERT(data.element_type_index() == output.element_type_index(), "Tile: input and output must have the same element type.");
        LUISA_ASSERT(ndim == output.ndim(), "Tile: input and output must have the same rank.");
        for (uint32_t d = 0; d < ndim; ++d) {
            LUISA_ASSERT(data.shape()[d] > 0, "Tile: input dimension must be > 0.");
            LUISA_ASSERT(output.shape()[d] % data.shape()[d] == 0, "Tile: output shape must be a multiple of input shape.");
        }
#endif

        // Check if input has standard contiguous row-major strides.
        bool in_is_contiguous = true;
        if (ndim > 0) {
            uint32_t expected_stride = 1u;
            for (int32_t d = static_cast<int32_t>(ndim) - 1; d >= 0; --d) {
                if (data.strides()[d] != expected_stride) {
                    in_is_contiguous = false;
                    break;
                }
                expected_stride *= data.shape()[d];
            }
        }

        visit_type_index<NNTypeList>(data.element_type_index(), [&]<typename T>() {
            auto &in = static_cast<NNTensor<T> &>(data);
            auto &out = static_cast<NNTensor<T> &>(output);
            auto const &in_shape = in.shape();
            auto in_size = static_cast<uint>(in.size());
            auto out_size = static_cast<uint>(out.size());

            bool both_byte_buffer = in.container().is_byte_buffer() && out.container().is_byte_buffer();

            // Fast path: contiguous input and output backed by byte buffers.
            // We can copy whole input blocks repeatedly, eliminating per-dim modulo.
            if (in_is_contiguous && both_byte_buffer) {
                using ST = nn_storage_type_t<T>;
                auto buf_in = in.container().get_byte_buffer();
                auto buf_out = out.container().get_byte_buffer();
                auto off_in = static_cast<uint>(in.container().get_byte_offset());
                auto off_out = static_cast<uint>(out.container().get_byte_offset());

                // Vectorized copy with float4 / half4 when block size is aligned.
                if constexpr (IsNativeArithmetic<T>::value && detail::VecDispatch<ST>::supported) {
                    using VecT = typename detail::VecDispatch<ST>::VecT;
                    constexpr auto vec_n = detail::VecDispatch<ST>::N;
                    if (in_size % vec_n == 0u) {
                        auto block_count = out_size / in_size;
                        auto rem_count = out_size % in_size;
                        auto in_vec_count = in_size / vec_n;
                        auto block_bytes = in_size * static_cast<uint>(sizeof(ST));
                        auto vec_bytes = static_cast<uint>(sizeof(VecT));

                        for (auto block : dynamic_range(block_count)) {
                            auto block_off_out = off_out + block * block_bytes;
                            for (auto i : dynamic_range(in_vec_count)) {
                                auto byte_idx = i * vec_bytes;
                                auto v = buf_in->read<VecT>(off_in + byte_idx);
                                buf_out->write(block_off_out + byte_idx, v);
                            }
                        }
                        if (rem_count > 0) {
                            auto rem_base_out = block_count * in_size;
                            for (auto i : dynamic_range(rem_count)) {
                                auto v = buf_in->read<ST>(off_in + i * static_cast<uint>(sizeof(ST)));
                                buf_out->write(off_out + (rem_base_out + i) * static_cast<uint>(sizeof(ST)), v);
                            }
                        }
                        return;
                    }
                }

                // Scalar block copy for contiguous byte-buffer tensors.
                // Use direct ByteBuffer read/write instead of DynamicArray operator[].
                auto block_count = out_size / in_size;
                auto rem_count = out_size % in_size;
                auto st_bytes = static_cast<uint>(sizeof(ST));
                for (auto block : dynamic_range(block_count)) {
                    auto base_out = block * in_size;
                    for (auto i : dynamic_range(in_size)) {
                        auto v = buf_in->read<ST>(off_in + i * st_bytes);
                        buf_out->write(off_out + (base_out + i) * st_bytes, v);
                    }
                }
                if (rem_count > 0) {
                    auto base_out = block_count * in_size;
                    for (auto i : dynamic_range(rem_count)) {
                        auto v = buf_in->read<ST>(off_in + i * st_bytes);
                        buf_out->write(off_out + (base_out + i) * st_bytes, v);
                    }
                }
                return;
            }

            // Medium path: both are ByteBuffer-backed but not contiguous.
            // Still avoid DynamicArray operator[] overhead by using direct ByteBuffer access.
            if constexpr (IsNativeArithmetic<T>::value) {
                if (both_byte_buffer) {
                    using ST = nn_storage_type_t<T>;
                    auto buf_in = in.container().get_byte_buffer();
                    auto buf_out = out.container().get_byte_buffer();
                    auto off_in = static_cast<uint>(in.container().get_byte_offset());
                    auto off_out = static_cast<uint>(out.container().get_byte_offset());
                    auto st_bytes = static_cast<uint>(sizeof(ST));

                    for (auto linear_out : dynamic_range(out.size())) {
                        auto in_linear = def(0u);
                        for_each_dim(linear_out, out.strides(), ndim, [&](uint32_t d, auto coord) {
                            auto in_coord = coord % in_shape[d];
                            in_linear += in_coord * in.strides()[d];
                        }, out.size());
                        auto v = buf_in->read<ST>(off_in + in_linear * st_bytes);
                        buf_out->write(off_out + linear_out * st_bytes, v);
                    }
                    return;
                }
            }

            // Generic fallback: per-dim coordinate decomposition with modulo.
            for (auto linear_out : dynamic_range(out.size())) {
                auto in_linear = def(0u);
                for_each_dim(linear_out, out.strides(), ndim, [&](uint32_t d, auto coord) {
                    auto in_coord = coord % in_shape[d];
                    in_linear += in_coord * in.strides()[d];
                }, out.size());
                out[linear_out] = in[in_linear];
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(Tile) {
    return luisa::make_unique<Tile>();
};

}// namespace lcml::onnx
