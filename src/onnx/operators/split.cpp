#include <luisa/core/stl/memory.h>
#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

namespace lcml::onnx {

// SplitBase: shared implementation for Split and SplitToSequence.
template<bool AsSequence>
class SplitBase : public Operator {
private:
    int64_t axis_;

public:
    SplitBase(int64_t axis) : Operator(AsSequence ? "SplitToSequence" : "Split"), axis_(axis) {}

    bool is_output_view([[maybe_unused]] size_t output_index,
                        [[maybe_unused]] onnx::Node const &node) const override {
        return axis_ == 0;
    }

    void forward(luisa::span<std::reference_wrapper<ITensor>> inputs,
                 luisa::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() >= 1 && !outputs.empty(),
                     "Split requires >=1 input and >=1 outputs.");
        auto &data = inputs[0].get();
        auto ndim = data.ndim();
        int64_t axis = axis_ < 0 ? axis_ + ndim : axis_;
        LUISA_ASSERT(axis >= 0 && axis < static_cast<int64_t>(ndim), "Split axis out of bounds");

        for (size_t i = 0; i < outputs.size(); ++i) {
            LUISA_ASSERT(outputs[i].get().element_type_index() == data.element_type_index(),
                         "Split: all outputs must have the same element type as input.");
        }
#else
        auto &data = inputs[0].get();
        auto ndim = data.ndim();
        int64_t axis = axis_ < 0 ? axis_ + ndim : axis_;
#endif

        visit_type_index<NNTypeList>(data.element_type_index(), [&]<typename T>() {
            auto &in = static_cast<NNTensor<T> &>(data);
            using ST = nn_storage_type_t<T>;

            // Fast path: axis == 0 means each output is a contiguous sub-region of input.
            if (axis == 0) {
                uint32_t offset = 0;
                for (size_t o = 0; o < outputs.size(); ++o) {
                    auto &out = static_cast<NNTensor<T> &>(outputs[o].get());
                    if (!in.container().shares_storage_with(out.container())) {
                        auto off = offset;
                        if constexpr (IsNativeArithmetic<T>::value && detail::VecDispatch<ST>::supported) {
                            if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                                using VecT = typename detail::VecDispatch<ST>::VecT;
                                constexpr uint32_t vec_n = detail::VecDispatch<ST>::N;
                                auto buf_in = in.container().get_byte_buffer();
                                auto buf_o = out.container().get_byte_buffer();
                                auto off_in = static_cast<uint>(in.container().get_byte_offset());
                                auto off_o = static_cast<uint>(out.container().get_byte_offset());
                                auto n = static_cast<uint>(out.size());
                                auto vec_count = n / vec_n;
                                auto rem = n % vec_n;
                                auto byte_off = off * static_cast<uint>(sizeof(ST));
                                for (auto i : dynamic_range(vec_count)) {
                                    auto byte_idx = i * static_cast<uint>(sizeof(VecT));
                                    auto v = buf_in->read<VecT>(off_in + byte_off + byte_idx);
                                    buf_o->write(off_o + byte_idx, v);
                                }
                                for (auto i : dynamic_range(rem)) {
                                    auto idx = vec_count * vec_n + i;
                                    auto v = buf_in->read<ST>(off_in + byte_off + idx * static_cast<uint>(sizeof(ST)));
                                    buf_o->write(off_o + idx * static_cast<uint>(sizeof(ST)), v);
                                }
                            } else if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                                auto buf_in_s = in.container().get_byte_buffer();
                                auto buf_o_s = out.container().get_byte_buffer();
                                auto off_in_s = static_cast<uint>(in.container().get_byte_offset());
                                auto off_o_s = static_cast<uint>(out.container().get_byte_offset());
                                for (auto i : dynamic_range(static_cast<uint>(out.size()))) {
                                    auto v = buf_in_s->read<ST>(off_in_s + (i + off) * static_cast<uint>(sizeof(ST)));
                                    buf_o_s->write(off_o_s + i * static_cast<uint>(sizeof(ST)), v);
                                }
                            } else {
                                for (auto i : dynamic_range(static_cast<uint>(out.size()))) {
                                    out[i] = in[i + off];
                                }
                            }
                        } else {
                            for (auto i : dynamic_range(static_cast<uint>(out.size()))) {
                                out[i] = in[i + off];
                            }
                        }
                    }
                    offset += static_cast<uint32_t>(out.shape()[0]) * in.strides()[0];
                }
                return;
            }

            // Fast path: axis == ndim - 1, inner dimension is contiguous.
            if (axis == static_cast<int64_t>(ndim) - 1) {
                uint32_t axis_offset = 0;
                for (size_t o = 0; o < outputs.size(); ++o) {
                    auto &out = static_cast<NNTensor<T> &>(outputs[o].get());
                    auto const &out_shape = out.shape();
                    auto slice_size = out_shape[axis];
                    auto num_slices = static_cast<uint>(out.size() / slice_size);
                    if (!in.container().shares_storage_with(out.container())) {
                        if constexpr (IsNativeArithmetic<T>::value && detail::VecDispatch<ST>::supported) {
                            if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                                using VecT = typename detail::VecDispatch<ST>::VecT;
                                constexpr uint32_t vec_n = detail::VecDispatch<ST>::N;
                                auto buf_in = in.container().get_byte_buffer();
                                auto buf_o = out.container().get_byte_buffer();
                                auto off_in = static_cast<uint>(in.container().get_byte_offset());
                                auto off_o = static_cast<uint>(out.container().get_byte_offset());
                                for (auto s : dynamic_range(num_slices)) {
                                    auto in_base = s * in.strides()[axis - 1] + axis_offset;
                                    auto out_base = s * slice_size;
                                    auto vec_count = slice_size / vec_n;
                                    auto rem = slice_size % vec_n;
                                    auto in_byte_base = off_in + in_base * static_cast<uint>(sizeof(ST));
                                    auto out_byte_base = off_o + out_base * static_cast<uint>(sizeof(ST));
                                    for (auto i : dynamic_range(vec_count)) {
                                        auto byte_idx = i * static_cast<uint>(sizeof(VecT));
                                        auto v = buf_in->read<VecT>(in_byte_base + byte_idx);
                                        buf_o->write(out_byte_base + byte_idx, v);
                                    }
                                    for (auto i : dynamic_range(rem)) {
                                        auto idx = vec_count * vec_n + i;
                                        auto v = buf_in->read<ST>(in_byte_base + idx * static_cast<uint>(sizeof(ST)));
                                        buf_o->write(out_byte_base + idx * static_cast<uint>(sizeof(ST)), v);
                                    }
                                }
                            } else if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                                auto buf_in_s = in.container().get_byte_buffer();
                                auto buf_o_s = out.container().get_byte_buffer();
                                auto off_in_s = static_cast<uint>(in.container().get_byte_offset());
                                auto off_o_s = static_cast<uint>(out.container().get_byte_offset());
                                for (auto s : dynamic_range(num_slices)) {
                                    auto in_base = s * in.strides()[axis - 1] + axis_offset;
                                    auto out_base = s * slice_size;
                                    for (auto k : dynamic_range(slice_size)) {
                                        auto v = buf_in_s->read<ST>(off_in_s + (in_base + k) * static_cast<uint>(sizeof(ST)));
                                        buf_o_s->write(off_o_s + (out_base + k) * static_cast<uint>(sizeof(ST)), v);
                                    }
                                }
                            } else {
                                for (auto s : dynamic_range(num_slices)) {
                                    auto in_base = s * in.strides()[axis - 1] + axis_offset;
                                    auto out_base = s * slice_size;
                                    for (auto k : dynamic_range(slice_size)) {
                                        out[out_base + k] = in[in_base + k];
                                    }
                                }
                            }
                        } else {
                            for (auto s : dynamic_range(num_slices)) {
                                auto in_base = s * in.strides()[axis - 1] + axis_offset;
                                auto out_base = s * slice_size;
                                for (auto k : dynamic_range(slice_size)) {
                                    out[out_base + k] = in[in_base + k];
                                }
                            }
                        }
                    }
                    axis_offset += slice_size;
                }
                return;
            }

            // General path: nested recursion over non-axis dimensions, inner loop over axis.
            uint32_t axis_offset = 0;
            for (size_t o = 0; o < outputs.size(); ++o) {
                auto &out = static_cast<NNTensor<T> &>(outputs[o].get());
                auto const &out_shape = out.shape();
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
                                    auto buf_o = out.container().get_byte_buffer();
                                    auto off_in = static_cast<uint>(in.container().get_byte_offset());
                                    auto off_o = static_cast<uint>(out.container().get_byte_offset());
                                    auto slice_len = out_shape[axis];
                                    auto vec_count = slice_len / vec_n;
                                    auto rem = slice_len % vec_n;
                                    auto in_byte_base = off_in + in_base * static_cast<uint>(sizeof(ST));
                                    auto out_byte_base = off_o + out_base * static_cast<uint>(sizeof(ST));
                                    for (auto i : dynamic_range(vec_count)) {
                                        auto byte_idx = i * static_cast<uint>(sizeof(VecT));
                                        auto v = buf_in->read<VecT>(in_byte_base + byte_idx);
                                        buf_o->write(out_byte_base + byte_idx, v);
                                    }
                                    for (auto i : dynamic_range(rem)) {
                                        auto idx = vec_count * vec_n + i;
                                        auto v = buf_in->read<ST>(in_byte_base + idx * static_cast<uint>(sizeof(ST)));
                                        buf_o->write(out_byte_base + idx * static_cast<uint>(sizeof(ST)), v);
                                    }
                                } else if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                                    auto buf_in_g = in.container().get_byte_buffer();
                                    auto buf_o_g = out.container().get_byte_buffer();
                                    auto off_in_g = static_cast<uint>(in.container().get_byte_offset());
                                    auto off_o_g = static_cast<uint>(out.container().get_byte_offset());
                                    for (auto k : dynamic_range(out_shape[axis])) {
                                        auto byte_in = off_in_g + (in_base + k * in.strides()[axis]) * static_cast<uint>(sizeof(ST));
                                        auto byte_out = off_o_g + (out_base + k * out.strides()[axis]) * static_cast<uint>(sizeof(ST));
                                        auto v = buf_in_g->read<ST>(byte_in);
                                        buf_o_g->write(byte_out, v);
                                    }
                                } else {
                                    for (auto k : dynamic_range(out_shape[axis])) {
                                        out[out_base + k * out.strides()[axis]] = in[in_base + k * in.strides()[axis]];
                                    }
                                }
                            } else if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                                auto buf_in_g = in.container().get_byte_buffer();
                                auto buf_o_g = out.container().get_byte_buffer();
                                auto off_in_g = static_cast<uint>(in.container().get_byte_offset());
                                auto off_o_g = static_cast<uint>(out.container().get_byte_offset());
                                for (auto k : dynamic_range(out_shape[axis])) {
                                    auto byte_in = off_in_g + (in_base + k * in.strides()[axis]) * static_cast<uint>(sizeof(ST));
                                    auto byte_out = off_o_g + (out_base + k * out.strides()[axis]) * static_cast<uint>(sizeof(ST));
                                    auto v = buf_in_g->read<ST>(byte_in);
                                    buf_o_g->write(byte_out, v);
                                }
                            } else {
                                for (auto k : dynamic_range(out_shape[axis])) {
                                    out[out_base + k * out.strides()[axis]] = in[in_base + k * in.strides()[axis]];
                                }
                            }
                        } else if (d == static_cast<uint32_t>(axis)) {
                            self(self, d + 1, in_base + axis_offset * in.strides()[axis], out_base);
                        } else if (out_shape[d] == 1) {
                            self(self, d + 1, in_base, out_base);
                        } else {
                            for (auto c : dynamic_range(out_shape[d])) {
                                self(self, d + 1,
                                     in_base + c * in.strides()[d],
                                     out_base + c * out.strides()[d]);
                            }
                        }
                    };
                    recurse(recurse, 0u, def(0u), def(0u));
                }
                axis_offset += out_shape[axis];
            }
        });
    }
};

using Split = SplitBase<false>;
using SplitToSequence = SplitBase<true>;

REGISTER_TO_DEFAULT_OPSET(Split) {
    int64_t axis = 0;
    if (auto p = node.try_get_attr("axis"))
        axis = p->get<onnx::AttributeType::INT>();
    return luisa::make_unique<Split>(axis);
};

REGISTER_TO_DEFAULT_OPSET(SplitToSequence) {
    int64_t axis = 0;
    if (auto p = node.try_get_attr("axis"))
        axis = p->get<onnx::AttributeType::INT>();
    return luisa::make_unique<SplitToSequence>(axis);
};

}// namespace lcml::onnx
