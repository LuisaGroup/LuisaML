#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"
#include <luisa/core/stl/memory.h>

namespace lcml::onnx {

// Concat: concatenates tensors along a specified axis.
// ONNX spec: attribute axis (required). Negative axis supported.
class Concat : public Operator {
private:
    int64_t axis_;

public:
    Concat(int64_t axis) : Operator("Concat"), axis_(axis) {}

    void forward(luisa::span<std::reference_wrapper<ITensor>> inputs,
                 luisa::span<std::reference_wrapper<ITensor>> outputs) override {
        LUISA_ASSERT(!inputs.empty() && outputs.size() == 1,
                     "Concat requires >=1 inputs and 1 output.");
        auto &output = outputs[0].get();
        auto ndim = output.ndim();

        // Check all inputs and output have same element type
        for (size_t i = 0; i < inputs.size(); ++i) {
            LUISA_ASSERT(inputs[i].get().element_type_index() == output.element_type_index(),
                         "Concat: all inputs and output must have the same element type.");
        }

        // Resolve negative axis
        int64_t axis = axis_;
        if (axis < 0) axis += ndim;
#ifndef NDEBUG
        LUISA_ASSERT(axis >= 0 && axis < static_cast<int64_t>(ndim), "Concat axis out of bounds");
#endif

        visit_type_index<NNTypeList>(output.element_type_index(), [&]<typename T>() {
            auto &out = static_cast<NNTensor<T> &>(output);
            auto const &out_shape = out.shape();
            using ST = nn_storage_type_t<T>;

            // Fast path: axis == 0, each input is a contiguous region in output
            if (axis == 0) {
                size_t offset = 0;
                for (size_t n = 0; n < inputs.size(); ++n) {
                    auto &in = static_cast<NNTensor<T> &>(inputs[n].get());
                    // If input already shares storage with output at this offset, skip copy
                    if (!in.container().shares_storage_with(out.container())) {
                        if constexpr (IsNativeArithmetic<T>::value && detail::VecDispatch<ST>::supported) {
                            if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                                using VecT = typename detail::VecDispatch<ST>::VecT;
                                constexpr uint32_t vec_n = detail::VecDispatch<ST>::N;
                                auto buf_in = in.container().get_byte_buffer();
                                auto buf_out = out.container().get_byte_buffer();
                                auto off_in = static_cast<uint>(in.container().get_byte_offset());
                                auto off_out = static_cast<uint>(out.container().get_byte_offset())
                                             + static_cast<uint>(offset) * static_cast<uint>(sizeof(ST));
                                auto n_elems = static_cast<uint>(in.size());
                                auto vec_count = n_elems / vec_n;
                                auto rem = n_elems % vec_n;
                                for (auto i : dynamic_range(vec_count)) {
                                    auto byte_idx = i * static_cast<uint>(sizeof(VecT));
                                    auto v = buf_in->read<VecT>(off_in + byte_idx);
                                    buf_out->write(off_out + byte_idx, v);
                                }
                                for (auto i : dynamic_range(rem)) {
                                    auto idx = vec_count * vec_n + i;
                                    auto byte_idx = idx * static_cast<uint>(sizeof(ST));
                                    auto v = buf_in->read<ST>(off_in + byte_idx);
                                    buf_out->write(off_out + byte_idx, v);
                                }
                            } else {
                                for (auto i : dynamic_range(static_cast<uint>(in.size()))) {
                                    out[i + static_cast<uint>(offset)] = in[i];
                                }
                            }
                        } else {
                            for (auto i : dynamic_range(static_cast<uint>(in.size()))) {
                                out[i + static_cast<uint>(offset)] = in[i];
                            }
                        }
                    }
                    offset += in.size();
                }
                return;
            }

            // Fast path: axis == ndim - 1, inner dimension is contiguous
            if (axis == static_cast<int64_t>(ndim) - 1) {
                uint32_t axis_offset = 0;
                for (size_t n = 0; n < inputs.size(); ++n) {
                    auto &in = static_cast<NNTensor<T> &>(inputs[n].get());
                    auto const &in_shape = in.shape();
                    auto slice_size = in_shape[axis];
                    auto num_slices = static_cast<uint>(in.size() / slice_size);
                    if (!in.container().shares_storage_with(out.container())) {
                        if constexpr (IsNativeArithmetic<T>::value && detail::VecDispatch<ST>::supported) {
                            if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                                using VecT = typename detail::VecDispatch<ST>::VecT;
                                constexpr uint32_t vec_n = detail::VecDispatch<ST>::N;
                                auto buf_in = in.container().get_byte_buffer();
                                auto buf_out = out.container().get_byte_buffer();
                                auto off_in = static_cast<uint>(in.container().get_byte_offset());
                                auto off_out = static_cast<uint>(out.container().get_byte_offset());
                                for (auto s : dynamic_range(num_slices)) {
                                    auto in_base = s * slice_size;
                                    auto out_base = s * out_shape[axis] + axis_offset;
                                    auto vec_count = slice_size / vec_n;
                                    auto rem = slice_size % vec_n;
                                    auto slice_off_in = off_in + in_base * static_cast<uint>(sizeof(ST));
                                    auto slice_off_out = off_out + out_base * static_cast<uint>(sizeof(ST));
                                    for (auto i : dynamic_range(vec_count)) {
                                        auto byte_idx = i * static_cast<uint>(sizeof(VecT));
                                        auto v = buf_in->read<VecT>(slice_off_in + byte_idx);
                                        buf_out->write(slice_off_out + byte_idx, v);
                                    }
                                    for (auto i : dynamic_range(rem)) {
                                        auto idx = vec_count * vec_n + i;
                                        auto byte_idx = idx * static_cast<uint>(sizeof(ST));
                                        auto v = buf_in->read<ST>(slice_off_in + byte_idx);
                                        buf_out->write(slice_off_out + byte_idx, v);
                                    }
                                }
                            } else {
                                for (auto s : dynamic_range(num_slices)) {
                                    auto in_base = s * slice_size;
                                    auto out_base = s * out_shape[axis] + axis_offset;
                                    for (auto k : dynamic_range(slice_size)) {
                                        out[out_base + k] = in[in_base + k];
                                    }
                                }
                            }
                        } else {
                            for (auto s : dynamic_range(num_slices)) {
                                auto in_base = s * slice_size;
                                auto out_base = s * out_shape[axis] + axis_offset;
                                for (auto k : dynamic_range(slice_size)) {
                                    out[out_base + k] = in[in_base + k];
                                }
                            }
                        }
                    }
                    axis_offset += in_shape[axis];
                }
                return;
            }

            // General path: nested loops over non-axis dimensions, inner loop over axis.
            // Avoids per-element for_each_dim coordinate decomposition.
            uint32_t axis_offset = 0;
            for (size_t n = 0; n < inputs.size(); ++n) {
                auto &in = static_cast<NNTensor<T> &>(inputs[n].get());
                auto const &in_shape = in.shape();
                if (!in.container().shares_storage_with(out.container())) {
                    auto recurse = [&](auto &&self, uint32_t d,
                                       Var<uint32_t> in_base,
                                       Var<uint32_t> out_base) -> void {
                        if (d == ndim) {
                            if constexpr (IsNativeArithmetic<T>::value && detail::VecDispatch<ST>::supported) {
                                if (in.container().is_byte_buffer() && out.container().is_byte_buffer() &&
                                    in.strides()[axis] == 1 && out.strides()[axis] == 1) {
                                    using VecT = typename detail::VecDispatch<ST>::VecT;
                                    constexpr uint32_t vec_n = detail::VecDispatch<ST>::N;
                                    auto buf_in = in.container().get_byte_buffer();
                                    auto buf_out = out.container().get_byte_buffer();
                                    auto off_in = static_cast<uint>(in.container().get_byte_offset());
                                    auto off_out = static_cast<uint>(out.container().get_byte_offset());
                                    auto slice_len = in_shape[axis];
                                    auto vec_count = slice_len / vec_n;
                                    auto rem = slice_len % vec_n;
                                    auto slice_off_in = off_in + in_base * static_cast<uint>(sizeof(ST));
                                    auto slice_off_out = off_out + out_base * static_cast<uint>(sizeof(ST));
                                    for (auto i : dynamic_range(vec_count)) {
                                        auto byte_idx = i * static_cast<uint>(sizeof(VecT));
                                        auto v = buf_in->read<VecT>(slice_off_in + byte_idx);
                                        buf_out->write(slice_off_out + byte_idx, v);
                                    }
                                    for (auto i : dynamic_range(rem)) {
                                        auto idx = vec_count * vec_n + i;
                                        auto byte_idx = idx * static_cast<uint>(sizeof(ST));
                                        auto v = buf_in->read<ST>(slice_off_in + byte_idx);
                                        buf_out->write(slice_off_out + byte_idx, v);
                                    }
                                } else {
                                    for (auto k : dynamic_range(in_shape[axis])) {
                                        out[out_base + k * out.strides()[axis]] = in[in_base + k * in.strides()[axis]];
                                    }
                                }
                            } else {
                                for (auto k : dynamic_range(in_shape[axis])) {
                                    out[out_base + k * out.strides()[axis]] = in[in_base + k * in.strides()[axis]];
                                }
                            }
                        } else if (d == static_cast<uint32_t>(axis)) {
                            self(self, d + 1, in_base, out_base + axis_offset * out.strides()[axis]);
                        } else if (in_shape[d] == 1) {
                            self(self, d + 1, in_base, out_base);
                        } else {
                            for (auto c : dynamic_range(in_shape[d])) {
                                self(self, d + 1,
                                     in_base + c * in.strides()[d],
                                     out_base + c * out.strides()[d]);
                            }
                        }
                    };
                    recurse(recurse, 0u, def(0u), def(0u));
                }
                axis_offset += in_shape[axis];
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(Concat) {
    int64_t axis = 0;
    if (auto p = node.try_get_attr("axis"))
        axis = p->get<onnx::AttributeType::INT>();
    return luisa::make_unique<Concat>(axis);
};

}// namespace lcml::onnx