#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

namespace lcml::onnx {

// Compress: selects elements along the given axis based on a boolean condition tensor.
// ONNX spec: input[0]=data, input[1]=condition (1-D bool); optional attribute axis
class Compress : public Operator {
private:
    std::optional<int64_t> axis_;

public:
    Compress(std::optional<int64_t> axis) : Operator("Compress"), axis_(axis) {}

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 2 && outputs.size() == 1,
                     "Compress requires 2 inputs and 1 output.");
        auto &data = inputs[0].get();
        auto &output = outputs[0].get();

        LUISA_ASSERT(data.element_type() == output.element_type(),
                     "Compress: data and output must have the same element type.");
        LUISA_ASSERT(inputs[1].get().element_type() == typeid(bool),
                     "Compress: condition must be bool type.");
#else
        auto &data = inputs[0].get();
        auto &output = outputs[0].get();
#endif

        // If no axis, data is flattened first
        // Output is already shaped by the runtime
        visit_typeid<NNTypeList>(data.element_type(), [&]<typename T>() {
            auto &in = static_cast<NNTensor<T> &>(data);
            auto &out = static_cast<NNTensor<T> &>(output);
            auto &cond = static_cast<NNTensor<bool> &>(inputs[1].get());
            using ST = nn_storage_type_t<T>;

            if (!axis_.has_value()) {
                // Flatten mode: iterate all elements, copy where condition is true
                // Per ONNX spec, elements beyond cond length are not selected
                auto out_idx = def(0u);
                auto n = min(cond.size(), in.size());

                // Vectorized read path: use ByteBuffer::read<half4/float4> for input.
                // Condition is read scalar-wise (bool ByteBuffer has backend alignment issues).
                // Output writes are scalar because out_idx is dynamic/sequential.
                if constexpr (detail::VecDispatch<ST>::supported) {
                    if (in.container().is_byte_buffer()) {
                        using VecT = typename detail::VecDispatch<ST>::VecT;
                        auto buf_in = in.container().get_byte_buffer();
                        auto off_in = static_cast<uint>(in.container().get_byte_offset());

                        auto vec_n = static_cast<uint>(n / detail::VecDispatch<ST>::N);
                        auto rem = static_cast<uint>(n % detail::VecDispatch<ST>::N);

                        if (out.container().is_byte_buffer()) {
                            auto buf_out = out.container().get_byte_buffer();
                            auto off_out = static_cast<uint>(out.container().get_byte_offset());
                            auto elem_size = static_cast<uint>(sizeof(ST));

                            for (auto i4 : dynamic_range(vec_n)) {
                                auto base = i4 * detail::VecDispatch<ST>::N;
                                auto v4 = buf_in->read<VecT>(off_in + base * static_cast<uint>(sizeof(ST)));

                                $if (cond[base + 0u]) {
                                    buf_out->write(off_out + out_idx * elem_size, v4.x);
                                    out_idx += 1u;
                                };
                                $if (cond[base + 1u]) {
                                    buf_out->write(off_out + out_idx * elem_size, v4.y);
                                    out_idx += 1u;
                                };
                                $if (cond[base + 2u]) {
                                    buf_out->write(off_out + out_idx * elem_size, v4.z);
                                    out_idx += 1u;
                                };
                                $if (cond[base + 3u]) {
                                    buf_out->write(off_out + out_idx * elem_size, v4.w);
                                    out_idx += 1u;
                                };
                            }
                            for (auto i : dynamic_range(rem)) {
                                auto idx = vec_n * detail::VecDispatch<ST>::N + i;
                                $if (cond[idx]) {
                                    buf_out->write(off_out + out_idx * elem_size, in[idx]);
                                    out_idx += 1u;
                                };
                            }
                        } else {
                            for (auto i4 : dynamic_range(vec_n)) {
                                auto base = i4 * detail::VecDispatch<ST>::N;
                                auto v4 = buf_in->read<VecT>(off_in + base * static_cast<uint>(sizeof(ST)));

                                $if (cond[base + 0u]) {
                                    out[out_idx] = v4.x;
                                    out_idx += 1u;
                                };
                                $if (cond[base + 1u]) {
                                    out[out_idx] = v4.y;
                                    out_idx += 1u;
                                };
                                $if (cond[base + 2u]) {
                                    out[out_idx] = v4.z;
                                    out_idx += 1u;
                                };
                                $if (cond[base + 3u]) {
                                    out[out_idx] = v4.w;
                                    out_idx += 1u;
                                };
                            }
                            for (auto i : dynamic_range(rem)) {
                                auto idx = vec_n * detail::VecDispatch<ST>::N + i;
                                $if (cond[idx]) {
                                    out[out_idx] = in[idx];
                                    out_idx += 1u;
                                };
                            }
                        }
                        return;
                    }
                }

                // Scalar fallback
                for (auto i : dynamic_range(n)) {
                    $if (cond[i]) {
                        out[out_idx] = in[i];
                        out_idx += 1u;
                    };
                }
            } else {
                // Axis mode: for each slice along axis, copy if condition is true.
                // Vectorization is limited here because:
                // 1. extract_coord is per-element and axis-dependent.
                // 2. out_idx is dynamic/sequential (scatter writes).
                // 3. cond is indexed by axis_coord, not linear_in.
                auto ndim = in.ndim();
                auto axis = static_cast<uint32_t>(axis_.value() < 0 ? axis_.value() + ndim : axis_.value());
                auto out_idx = def(0u);

                for (auto linear_in : dynamic_range(in.size())) {
                    auto axis_coord = extract_coord(linear_in, in.strides(), in.shape(), axis);
                    $if (cond[axis_coord]) {
                        out[out_idx] = in[linear_in];
                        out_idx += 1u;
                    };
                }
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(Compress) {
    std::optional<int64_t> axis;
    if (auto p = node.try_get_attr("axis"))
        axis = p->get<onnx::AttributeType::INT>();
    return std::make_unique<Compress>(axis);
};

}// namespace lcml::onnx
