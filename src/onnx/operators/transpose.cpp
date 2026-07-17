#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/memory.h>
#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

namespace lcml::onnx {

// Transpose: permutes the dimensions of the input tensor.
// ONNX spec: attribute perm (list of ints). Default: reverse of input dims.
class Transpose : public Operator {
private:
    luisa::vector<int32_t> perm_;

public:
    Transpose(luisa::vector<int32_t> perm) : Operator("Transpose"), perm_(std::move(perm)) {}

    /// Transpose is a view when either:
    /// 1. The permutation is identity, OR
    /// 2. The output tensor is effectively 1-D (at most one dimension > 1),
    ///    which means memory layout is identical regardless of permutation.
    bool is_output_view([[maybe_unused]] size_t output_index,
                        [[maybe_unused]] onnx::Node const &node) const override {
        // Check identity permutation
        if (!perm_.empty()) {
            bool identity = true;
            for (size_t i = 0; i < perm_.size(); ++i) {
                if (perm_[i] != static_cast<int32_t>(i)) {
                    identity = false;
                    break;
                }
            }
            if (identity) return true;
        }
        // Check effectively-1D: if output has at most one dim > 1, it's a view
        if (!node.get_outputs().empty()) {
            auto const &out_shape = node.get_outputs()[0].get().get_shape();
            size_t dims_gt1 = 0;
            for (auto d : out_shape) {
                if (d > 1) ++dims_gt1;
            }
            if (dims_gt1 <= 1) return true;
        }
        return false;
    }

    void forward(luisa::span<std::reference_wrapper<ITensor>> inputs,
                 luisa::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 1 && outputs.size() == 1,
                     "Transpose requires 1 input and 1 output.");
        auto &input = inputs[0].get();
        auto &output = outputs[0].get();

        auto ndim = input.ndim();
        LUISA_ASSERT(input.element_type_index() == output.element_type_index(),
                     "Transpose: input and output must have the same element type.");
#else
        auto &input = inputs[0].get();
        auto &output = outputs[0].get();
        auto ndim = input.ndim();
#endif

        // Resolve perm: if empty, reverse
        luisa::vector<int32_t> perm = perm_;
        if (perm.empty()) {
            perm.resize(ndim);
            for (uint32_t i = 0; i < ndim; ++i) perm[i] = ndim - 1 - i;
        }

#ifndef NDEBUG
        // Validate perm correctness to prevent out-of-bounds or silent errors
        LUISA_ASSERT(perm.size() == ndim,
                     "Transpose: perm size ({}) must match tensor ndim ({}).",
                     perm.size(), ndim);
        {
            luisa::vector<bool> seen(ndim, false);
            for (auto p : perm) {
                LUISA_ASSERT(p >= 0 && static_cast<uint32_t>(p) < ndim,
                             "Transpose: perm value {} out of bounds [0, {}).",
                             p, ndim);
                LUISA_ASSERT(!seen[p],
                             "Transpose: perm contains duplicate value {}.", p);
                seen[p] = true;
            }
        }
#endif

        // Fast path: effectively 1-D tensor or identity permutation -> direct copy
        bool is_identity = is_effectively_1d(output.shape(), ndim);
        for (uint32_t i = 0; !is_identity && i < ndim; ++i) {
            if (perm[i] != static_cast<int32_t>(i)) break;
            if (i == ndim - 1) is_identity = true;
        }

        visit_type_index<NNTypeList>(input.element_type_index(), [&]<typename T>() {
            auto &in = static_cast<NNTensor<T> &>(input);
            auto &out = static_cast<NNTensor<T> &>(output);
            if (is_identity) {
                // If already sharing storage (inplace allocation), skip assignment
                if (!in.container().shares_storage_with(out.container())) {
                    using ST = nn_storage_type_t<T>;
                    if constexpr (IsNativeArithmetic<T>::value && detail::VecDispatch<ST>::supported) {
                        if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                            detail::vectorized_unary<ST>(in, out, [](auto v) { return v; });
                        } else {
                            for (auto i : dynamic_range(static_cast<uint32_t>(out.size()))) {
                                out[i] = in[i];
                            }
                        }
                    } else {
                        for (auto i : dynamic_range(static_cast<uint32_t>(out.size()))) {
                            out[i] = in[i];
                        }
                    }
                }
                return;
            }

            auto const &out_shape = out.shape();
            using ST = nn_storage_type_t<T>;
            bool inner_contig = false;
            uint32_t inner_size = 0;
            if (ndim > 0) {
                inner_size = out_shape[ndim - 1];
                inner_contig = (in.strides()[perm[ndim - 1]] == 1);
            }

            if constexpr (IsNativeArithmetic<T>::value && detail::VecDispatch<ST>::supported) {
                if (inner_contig && inner_size >= detail::VecDispatch<ST>::N &&
                    in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                    // Innermost output dimension maps to contiguous input dimension:
                    // process each row with float4/half4 vector loads.
                    using VecT = typename detail::VecDispatch<ST>::VecT;
                    auto buf_in = in.container().get_byte_buffer();
                    auto buf_o = out.container().get_byte_buffer();
                    auto off_in = static_cast<uint>(in.container().get_byte_offset());
                    auto off_o = static_cast<uint>(out.container().get_byte_offset());
                    auto elem_size = static_cast<uint>(sizeof(ST));
                    auto num_rows = static_cast<uint>(out.size() / inner_size);
                    auto vec_n = inner_size / detail::VecDispatch<ST>::N;
                    auto rem = inner_size % detail::VecDispatch<ST>::N;

                    for (auto row : dynamic_range(num_rows)) {
                        auto base_out = row * inner_size;
                        auto base_in = def(0u);
                        for_each_dim(base_out, out.strides(), ndim,
                                     [&](uint32_t d, auto coord) {
                                         base_in += coord * in.strides()[perm[d]];
                                     },
                                     out.size());
                        for (auto i : dynamic_range(vec_n)) {
                            auto off = i * detail::VecDispatch<ST>::N;
                            auto v = buf_in->read<VecT>(off_in + (base_in + off) * elem_size);
                            buf_o->write(off_o + (base_out + off) * elem_size, v);
                        }
                        auto vec_len = vec_n * detail::VecDispatch<ST>::N;
                        for (auto i : dynamic_range(rem)) {
                            auto off = vec_len + i;
                            auto v = buf_in->read<ST>(off_in + (base_in + off) * elem_size);
                            buf_o->write(off_o + (base_out + off) * elem_size, v);
                        }
                    }
                    return;
                }
            }

            // General transpose: element-wise with multi-dim index remap.
            if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                auto buf_in = in.container().get_byte_buffer();
                auto buf_o = out.container().get_byte_buffer();
                auto off_in = static_cast<uint>(in.container().get_byte_offset());
                auto off_o = static_cast<uint>(out.container().get_byte_offset());
                auto es = static_cast<uint>(sizeof(ST));
                for (auto linear_out : dynamic_range(static_cast<uint32_t>(out.size()))) {
                    auto idx_in = def(0u);
                    for_each_dim(linear_out, out.strides(), ndim,
                                 [&](uint32_t d, auto coord) {
                                     idx_in += coord * in.strides()[perm[d]];
                                 },
                                 out.size());
                    auto v = buf_in->read<ST>(off_in + idx_in * es);
                    buf_o->write(off_o + linear_out * es, v);
                }
            } else {
                for (auto linear_out : dynamic_range(static_cast<uint32_t>(out.size()))) {
                    auto idx_in = def(0u);
                    for_each_dim(linear_out, out.strides(), ndim,
                                 [&](uint32_t d, auto coord) {
                                     idx_in += coord * in.strides()[perm[d]];
                                 },
                                 out.size());
                    out[linear_out] = in[idx_in];
                }
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(Transpose) {
    luisa::vector<int32_t> perm;
    if (auto p = node.try_get_attr("perm"))
        perm = p->get<onnx::AttributeType::INTS>();
    return luisa::make_unique<Transpose>(std::move(perm));
};

}// namespace lcml::onnx
