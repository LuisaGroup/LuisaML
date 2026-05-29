#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

namespace lcml::onnx {

// ReverseSequence: reverses elements of each batch along a specified axis,
// up to the length specified by the sequence_lens input.
// ONNX spec: input[0]=data, input[1]=sequence_lens; attributes: batch_axis (default 1), time_axis (default 0)
class ReverseSequence : public Operator {
private:
    int64_t batch_axis_;
    int64_t time_axis_;

public:
    ReverseSequence(int64_t batch_axis, int64_t time_axis)
        : Operator("ReverseSequence"), batch_axis_(batch_axis), time_axis_(time_axis) {}

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 2 && outputs.size() == 1, "ReverseSequence requires 2 inputs and 1 output.");
        auto &data = inputs[0].get();
        auto &output = outputs[0].get();
        auto ndim = data.ndim();

        LUISA_ASSERT(data.element_type() == output.element_type(), "ReverseSequence: data and output must have the same element type.");
        LUISA_ASSERT(inputs[1].get().element_type() == typeid(int) || inputs[1].get().element_type() == typeid(slong),
                     "ReverseSequence: sequence_lens must be int or int64 type.");
#else
        auto &data = inputs[0].get();
        auto &output = outputs[0].get();
        auto ndim = data.ndim();
#endif

        visit_typeid<NNTypeList>(data.element_type(), [&]<typename T>() {
            auto &in = static_cast<NNTensor<T> &>(data);
            auto &seq_lens = static_cast<NNTensor<int> &>(inputs[1].get());
            auto &out = static_cast<NNTensor<T> &>(output);

            auto batch_axis = static_cast<uint32_t>(batch_axis_ < 0 ? batch_axis_ + ndim : batch_axis_);
            auto time_axis = static_cast<uint32_t>(time_axis_ < 0 ? time_axis_ + ndim : time_axis_);
            auto time_stride = out.strides()[time_axis];
            auto batch_stride = out.strides()[batch_axis];
            auto time_size = out.shape()[time_axis];
            auto batch_size = out.shape()[batch_axis];

#ifndef NDEBUG
            LUISA_ASSERT(batch_axis != time_axis, "ReverseSequence: batch_axis and time_axis must be different.");
            LUISA_ASSERT(batch_axis < ndim && time_axis < ndim, "ReverseSequence: axis out of range.");
            LUISA_ASSERT(seq_lens.size() == batch_size,
                         "ReverseSequence: sequence_lens length must equal batch size.");
#endif

            using ST = nn_storage_type_t<T>;

            // When one stride is a multiple of the other, the tensor can be partitioned into
            // uniform blocks where both batch_coord and time_coord are constant.
            bool aligned_blocks = (batch_stride % time_stride == 0u) || (time_stride % batch_stride == 0u);
            auto block_size = std::min(batch_stride, time_stride);

            if constexpr (detail::VecDispatch<ST>::supported) {
                if (aligned_blocks && block_size >= 4u &&
                    in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                    using VecT = typename detail::VecDispatch<ST>::VecT;
                    constexpr uint32_t N = detail::VecDispatch<ST>::N;
                    auto buf_in = in.container().get_byte_buffer();
                    auto buf_o = out.container().get_byte_buffer();
                    auto off_in = static_cast<uint>(in.container().get_byte_offset());
                    auto off_o = static_cast<uint>(out.container().get_byte_offset());
                    auto es = static_cast<uint>(sizeof(ST));

                    auto total = static_cast<uint>(out.size());
                    auto num_blocks = total / block_size;
                    auto rem = total % block_size;

                    for (auto block_idx : dynamic_range(num_blocks)) {
                        auto block_start = block_idx * block_size;
                        auto batch_coord = (block_start / batch_stride) % batch_size;
                        auto time_coord = (block_start / time_stride) % time_size;
                        auto seq_len = seq_lens[batch_coord].cast<uint>();

                        auto delta = def(0u);
                        $if (time_coord < seq_len) {
                            auto reversed_time = seq_len - 1u - time_coord;
                            delta = (reversed_time - time_coord) * time_stride;
                        };

                        auto vec_count = block_size / N;
                        auto tail = block_size % N;
                        auto src_base = (block_start + delta) * es + off_in;
                        auto dst_base = block_start * es + off_o;
                        for (auto i : dynamic_range(vec_count)) {
                            auto byte_offset = i * N * es;
                            auto v = buf_in->read<VecT>(src_base + byte_offset);
                            buf_o->write(dst_base + byte_offset, v);
                        }
                        auto tail_start = block_start + vec_count * N;
                        auto src_tail_base = (tail_start + delta) * es + off_in;
                        auto dst_tail_base = tail_start * es + off_o;
                        for (auto i : dynamic_range(tail)) {
                            auto byte_offset = i * es;
                            auto v = buf_in->read<ST>(src_tail_base + byte_offset);
                            buf_o->write(dst_tail_base + byte_offset, v);
                        }
                    }

                    for (auto i : dynamic_range(rem)) {
                        auto linear_out = num_blocks * block_size + i;
                        auto batch_coord_r = (linear_out / batch_stride) % batch_size;
                        auto time_coord_r = (linear_out / time_stride) % time_size;
                        auto seq_len_r = seq_lens[batch_coord_r].cast<uint>();

                        auto in_linear = linear_out;
                        $if (time_coord_r < seq_len_r) {
                            auto reversed_time = seq_len_r - 1u - time_coord_r;
                            in_linear = linear_out - time_coord_r * time_stride + reversed_time * time_stride;
                        };
                        auto src_byte = in_linear * es + off_in;
                        auto dst_byte = linear_out * es + off_o;
                        auto v = buf_in->read<ST>(src_byte);
                        buf_o->write(dst_byte, v);
                    }
                    return;
                }
            }

            // Scalar ByteBuffer fallback for native types (avoids DynamicArray::operator[]
            // write-through issues on BufferData).
            if constexpr (IsNativeArithmetic<ST>::value) {
                if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                    auto buf_in = in.container().get_byte_buffer();
                    auto buf_o = out.container().get_byte_buffer();
                    auto off_in = static_cast<uint>(in.container().get_byte_offset());
                    auto off_o = static_cast<uint>(out.container().get_byte_offset());
                    auto es = static_cast<uint>(sizeof(ST));
                    for (auto linear_out : dynamic_range(out.size())) {
                        auto batch_coord = (linear_out / batch_stride) % batch_size;
                        auto time_coord = (linear_out / time_stride) % time_size;
                        auto seq_len = seq_lens[batch_coord].cast<uint>();

                        auto in_linear = linear_out;
                        $if (time_coord < seq_len) {
                            auto reversed_time = seq_len - 1u - time_coord;
                            in_linear = linear_out - time_coord * time_stride + reversed_time * time_stride;
                        };
                        auto v = buf_in->read<ST>(off_in + in_linear * es);
                        buf_o->write(off_o + linear_out * es, v);
                    }
                    return;
                }
            }

            // Pure scalar fallback with DynamicArray indexing (safe for Local/View/Scalar/Linear)
            for (auto linear_out : dynamic_range(out.size())) {
                auto batch_coord = (linear_out / batch_stride) % batch_size;
                auto time_coord = (linear_out / time_stride) % time_size;
                auto seq_len = seq_lens[batch_coord].cast<uint>();

                auto in_linear = linear_out;
                $if (time_coord < seq_len) {
                    auto reversed_time = seq_len - 1u - time_coord;
                    in_linear = linear_out - time_coord * time_stride + reversed_time * time_stride;
                };

                out[linear_out] = in[in_linear];
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(ReverseSequence) {
    int64_t batch_axis = 1;
    int64_t time_axis = 0;
    if (auto p = node.try_get_attr("batch_axis"))
        batch_axis = p->get<onnx::AttributeType::INT>();
    if (auto p = node.try_get_attr("time_axis"))
        time_axis = p->get<onnx::AttributeType::INT>();
    return std::make_unique<ReverseSequence>(batch_axis, time_axis);
};

}// namespace lcml::onnx
