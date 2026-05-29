#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

namespace lcml::onnx {

// GatherND: gathers slices from data into an output tensor of shape determined by indices.
// ONNX spec: input[0]=data, input[1]=indices; attribute batch_dims (default 0)
class GatherND : public Operator {
private:
    int batch_dims_;

public:
    GatherND(int batch_dims) : Operator("GatherND"), batch_dims_(batch_dims) {}

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
        LUISA_ASSERT(inputs.size() == 2 && outputs.size() == 1,
                     "GatherND requires 2 inputs and 1 output.");
        auto &data = inputs[0].get();
        auto &indices = inputs[1].get();
        auto &output = outputs[0].get();

        auto const &data_shape = data.shape();
        auto const &idx_shape = indices.shape();
        auto data_ndim = data.ndim();
        auto idx_ndim = indices.ndim();
        auto last_idx_dim = idx_shape[idx_ndim - 1];// number of index dimensions

        LUISA_ASSERT(data.element_type() == output.element_type(),
                     "GatherND: data and output must have the same element type.");
        LUISA_ASSERT(indices.element_type() == typeid(int) || indices.element_type() == typeid(slong),
                     "GatherND: indices must be int or int64 type.");

        LUISA_ASSERT(batch_dims_ >= 0 && batch_dims_ < static_cast<int>(data_ndim),
                     "GatherND: batch_dims must be in [0, data_ndim).");
        LUISA_ASSERT(batch_dims_ < static_cast<int>(idx_ndim),
                     "GatherND: batch_dims must be < indices.ndim.");
        LUISA_ASSERT(last_idx_dim <= data_ndim - batch_dims_,
                     "GatherND: last_idx_dim must be <= data_ndim - batch_dims.");

        visit_typeid<NNTypeList>(data.element_type(), [&]<typename T>() {
            auto &in = static_cast<NNTensor<T> &>(data);
            auto &idx_tensor = static_cast<NNTensor<int> &>(indices);
            auto &out = static_cast<NNTensor<T> &>(output);
            auto const &out_shape = out.shape();
            auto out_ndim = out.ndim();
            using ST = nn_storage_type_t<T>;

            // Compute slice size = product of trailing data dimensions after indexed dims
            uint32_t slice_size = 1u;
            for (uint32_t d = batch_dims_ + last_idx_dim; d < data_ndim; ++d) {
                slice_size *= data_shape[d];
            }
            auto num_slices = static_cast<uint32_t>(out.size()) / slice_size;
            auto slice_start = static_cast<uint32_t>(idx_ndim - 1);

            auto vectorized_slice_copy = [&](Var<uint> data_base, Var<uint> out_base, uint32_t n) {
                if constexpr (detail::VecDispatch<ST>::supported) {
                    if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                        using VecT = typename detail::VecDispatch<ST>::VecT;
                        constexpr uint32_t N = detail::VecDispatch<ST>::N;
                        auto buf_in = in.container().get_byte_buffer();
                        auto buf_o = out.container().get_byte_buffer();
                        auto off_in = static_cast<uint>(in.container().get_byte_offset()) +
                                      data_base * static_cast<uint>(sizeof(ST));
                        auto off_o = static_cast<uint>(out.container().get_byte_offset()) +
                                     out_base * static_cast<uint>(sizeof(ST));
                        auto vec_count = n / N;
                        auto rem = n % N;
                        for (auto i : dynamic_range(vec_count)) {
                            auto byte_idx = i * static_cast<uint>(sizeof(VecT));
                            auto v = buf_in->read<VecT>(off_in + byte_idx);
                            buf_o->write(off_o + byte_idx, v);
                        }
                        for (auto i : dynamic_range(rem)) {
                            auto idx = vec_count * N + i;
                            auto byte_idx = idx * static_cast<uint>(sizeof(ST));
                            auto v = buf_in->read<ST>(off_in + byte_idx);
                            buf_o->write(off_o + byte_idx, v);
                        }
                        return;
                    }
                }
                for (auto i : dynamic_range(n)) {
                    out[out_base + i] = in[data_base + i];
                }
            };

            if (in.container().shares_storage_with(out.container())) {
                // Shared storage: fall back to element-wise to minimize
                // overwrite risk (order is still unpredictable, same as original).
                bool use_bb_scalar = in.container().is_byte_buffer() && out.container().is_byte_buffer();
                if (use_bb_scalar) {
                    auto buf_in = in.container().get_byte_buffer();
                    auto buf_o = out.container().get_byte_buffer();
                    auto off_in = static_cast<uint>(in.container().get_byte_offset());
                    auto off_o = static_cast<uint>(out.container().get_byte_offset());
                    for (auto linear_out : dynamic_range(out.size())) {
                        DynamicArray<RemoveVarT<decltype(linear_out)>> out_coords(out_ndim);
                        for_each_dim(linear_out, out.strides(), out_ndim, [&](uint32_t d, auto coord) { out_coords[d] = coord; }, out.size());

                        auto idx_base = def(0u);
                        for (uint32_t d = 0; d < idx_ndim - 1; ++d) {
                            idx_base += out_coords[d] * idx_tensor.strides()[d];
                        }

                        auto data_linear = def(0u);
                        for (int d = 0; d < batch_dims_; ++d) {
                            data_linear += out_coords[d] * in.strides()[d];
                        }
                        for (uint32_t k = 0; k < last_idx_dim; ++k) {
                            auto idx_signed = idx_tensor[idx_base + k];
                            auto dim_size = static_cast<int>(in.shape()[batch_dims_ + k]);
                            idx_signed = ite(idx_signed < 0, idx_signed + dim_size, idx_signed);
                            auto idx_val = idx_signed.cast<uint>();
                            data_linear += idx_val * in.strides()[batch_dims_ + k];
                        }
                        for (uint32_t d = slice_start; d < out_ndim; ++d) {
                            auto data_dim = batch_dims_ + last_idx_dim + (d - slice_start);
                            data_linear += out_coords[d] * in.strides()[data_dim];
                        }

                        auto val = buf_in->read<ST>(off_in + data_linear * static_cast<uint>(sizeof(ST)));
                        buf_o->write(off_o + linear_out * static_cast<uint>(sizeof(ST)), val);
                    }
                } else {
                    for (auto linear_out : dynamic_range(out.size())) {
                        DynamicArray<RemoveVarT<decltype(linear_out)>> out_coords(out_ndim);
                        for_each_dim(linear_out, out.strides(), out_ndim, [&](uint32_t d, auto coord) { out_coords[d] = coord; }, out.size());

                        auto idx_base = def(0u);
                        for (uint32_t d = 0; d < idx_ndim - 1; ++d) {
                            idx_base += out_coords[d] * idx_tensor.strides()[d];
                        }

                        auto data_linear = def(0u);
                        for (int d = 0; d < batch_dims_; ++d) {
                            data_linear += out_coords[d] * in.strides()[d];
                        }
                        for (uint32_t k = 0; k < last_idx_dim; ++k) {
                            auto idx_signed = idx_tensor[idx_base + k];
                            auto dim_size = static_cast<int>(in.shape()[batch_dims_ + k]);
                            idx_signed = ite(idx_signed < 0, idx_signed + dim_size, idx_signed);
                            auto idx_val = idx_signed.cast<uint>();
                            data_linear += idx_val * in.strides()[batch_dims_ + k];
                        }
                        for (uint32_t d = slice_start; d < out_ndim; ++d) {
                            auto data_dim = batch_dims_ + last_idx_dim + (d - slice_start);
                            data_linear += out_coords[d] * in.strides()[data_dim];
                        }

                        out[linear_out] = in[data_linear];
                    }
                }
                return;
            }

            for (auto slice_idx : dynamic_range(num_slices)) {
                auto out_base = slice_idx * slice_size;

                // Decompose the slice-start linear index to obtain coordinates
                DynamicArray<uint> out_coords(out_ndim);
                for_each_dim(out_base, out.strides(), out_ndim, [&](uint32_t d, auto coord) { out_coords[d] = coord; }, out.size());

                // Build index into indices tensor
                auto idx_base = def(0u);
                for (uint32_t d = 0; d < idx_ndim - 1; ++d) {
                    idx_base += out_coords[d] * idx_tensor.strides()[d];
                }

                // Compute data base address for this slice
                auto data_base = def(0u);
                for (int d = 0; d < batch_dims_; ++d) {
                    data_base += out_coords[d] * in.strides()[d];
                }
                for (uint32_t k = 0; k < last_idx_dim; ++k) {
                    auto idx_signed = idx_tensor[idx_base + k];
                    auto dim_size = static_cast<int>(in.shape()[batch_dims_ + k]);
                    idx_signed = ite(idx_signed < 0, idx_signed + dim_size, idx_signed);
                    auto idx_val = idx_signed.cast<uint>();
                    data_base += idx_val * in.strides()[batch_dims_ + k];
                }

                vectorized_slice_copy(data_base, out_base, slice_size);
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(GatherND) {
    int batch_dims = 0;
    if (auto p = node.try_get_attr("batch_dims"))
        batch_dims = p->get<onnx::AttributeType::INT>();
    return std::make_unique<GatherND>(batch_dims);
};

}// namespace lcml::onnx
