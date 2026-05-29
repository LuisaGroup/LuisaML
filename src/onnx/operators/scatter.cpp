#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

namespace lcml::onnx {

enum class Reduction { NONE,
                       ADD,
                       MUL,
                       MIN,
                       MAX };

static Reduction parse_reduction(std::string_view s) {
    if (s == "add") return Reduction::ADD;
    if (s == "mul") return Reduction::MUL;
    if (s == "min") return Reduction::MIN;
    if (s == "max") return Reduction::MAX;
    return Reduction::NONE;
}

template<typename Tensor, typename Idx1, typename Idx2>
void apply_reduction(Tensor &out, Idx1 idx, Tensor &upd, Idx2 upd_idx, Reduction reduction) {
    using elem_t = typename Tensor::value_type;
    constexpr bool is_fp_quantized = std::is_same_v<elem_t, FP4E2M1> ||
                                      std::is_same_v<elem_t, FP8E4M3FN> ||
                                      std::is_same_v<elem_t, FP8E5M2> ||
                                      std::is_same_v<elem_t, FP16Quantized>;
    if (reduction == Reduction::NONE) {
        out[idx] = upd[upd_idx];
    } else if constexpr (std::is_same_v<elem_t, bool> || is_fp_quantized) {
        // bool and FP quantized types only support NONE reduction
        out[idx] = upd[upd_idx];
    } else {
        switch (reduction) {
            case Reduction::ADD:
                if constexpr (luisa::is_floating_point_v<elem_t>) {
                    out[idx] = fma(upd[upd_idx], elem_t{1}, out[idx]);
                } else {
                    out[idx] = out[idx] + upd[upd_idx];
                }
                break;
            case Reduction::MUL:
                out[idx] = out[idx] * upd[upd_idx];
                break;
            case Reduction::MIN:
                out[idx] = min(out[idx], upd[upd_idx]);
                break;
            case Reduction::MAX:
                out[idx] = max(out[idx], upd[upd_idx]);
                break;
            default:
                break;
        }
    }
}

// ByteBuffer-aware reduction: uses explicit buf->read/write to avoid DynamicArray::operator[]
// which does not generate correct store instructions for ByteBuffer-backed tensors.
template<typename ST, typename Idx1>
void apply_reduction_buf(Var<ByteBuffer> *buf_out, uint off_out, Idx1 idx,
                         Var<ST> upd_val, Reduction reduction) {
    using elem_t = ST;
    constexpr bool is_fp_quantized = std::is_same_v<elem_t, FP4E2M1> ||
                                      std::is_same_v<elem_t, FP8E4M3FN> ||
                                      std::is_same_v<elem_t, FP8E5M2> ||
                                      std::is_same_v<elem_t, FP16Quantized>;
    auto read_out = [&]() -> Var<elem_t> {
        return buf_out->read<elem_t>(off_out + idx * static_cast<uint>(sizeof(elem_t)));
    };
    auto write_out = [&](auto val) {
        buf_out->write(off_out + idx * static_cast<uint>(sizeof(elem_t)), val);
    };
    if (reduction == Reduction::NONE) {
        write_out(upd_val);
    } else if constexpr (std::is_same_v<elem_t, bool> || is_fp_quantized) {
        write_out(upd_val);
    } else {
        switch (reduction) {
            case Reduction::ADD:
                if constexpr (luisa::is_floating_point_v<elem_t>) {
                    write_out(fma(upd_val, elem_t{1}, read_out()));
                } else {
                    write_out(read_out() + upd_val);
                }
                break;
            case Reduction::MUL:
                write_out(read_out() * upd_val);
                break;
            case Reduction::MIN:
                write_out(min(read_out(), upd_val));
                break;
            case Reduction::MAX:
                write_out(max(read_out(), upd_val));
                break;
            default:
                break;
        }
    }
}

// ScatterElements: writes values from updates into data at positions specified by indices.
// ONNX spec: input[0]=data, input[1]=indices, input[2]=updates; attribute axis (default 0), reduction (default "none")
class ScatterElements : public Operator {
private:
    int64_t axis_;
    Reduction reduction_;

public:
    ScatterElements(int64_t axis, Reduction reduction, std::string name = "ScatterElements")
        : Operator(std::move(name)), axis_(axis), reduction_(reduction) {}

    bool can_operate_inplace() const override { return true; }

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
        LUISA_ASSERT(inputs.size() == 3 && outputs.size() == 1, "ScatterElements requires 3 inputs and 1 output.");
        auto &data = inputs[0].get();
        auto &indices = inputs[1].get();
        auto &updates = inputs[2].get();
        auto &output = outputs[0].get();
        auto ndim = data.ndim();
        int64_t axis = axis_ < 0 ? axis_ + ndim : axis_;

        LUISA_ASSERT(data.element_type() == updates.element_type() && data.element_type() == output.element_type(),
                     "ScatterElements: data, updates and output must have the same element type.");
        LUISA_ASSERT(indices.element_type() == typeid(int) || indices.element_type() == typeid(slong),
                     "ScatterElements: indices must be int or int64 type.");

        visit_typeid<NNTypeList>(data.element_type(), [&]<typename T>() {
            auto &in = static_cast<NNTensor<T> &>(data);
            auto &idx = static_cast<NNTensor<int> &>(indices);
            auto &upd = static_cast<NNTensor<T> &>(updates);
            auto &out = static_cast<NNTensor<T> &>(output);
            using ST = typename NNTensor<T>::value_type;

            // Copy data to output first — but skip if they share the same
            // underlying storage (in-place reuse from memory pool).
            if (!in.container().shares_storage_with(out.container())) {
                if constexpr (detail::VecDispatch<ST>::supported) {
                    if (detail::all_byte_buffer(in, out)) {
                        detail::vectorized_unary<ST>(in, out, [](auto v) { return v; });
                    } else {
                        for (auto i : dynamic_range(in.size())) {
                            out[i] = in[i];
                        }
                    }
                } else if (detail::all_byte_buffer(in, out)) {
                    auto buf_in = in.container().get_byte_buffer();
                    auto buf_out = out.container().get_byte_buffer();
                    auto off_in = static_cast<uint>(in.container().get_byte_offset());
                    auto off_out = static_cast<uint>(out.container().get_byte_offset());
                    for (auto i : dynamic_range(in.size())) {
                        auto byte_idx = i * static_cast<uint>(sizeof(ST));
                        auto v = buf_in->read<ST>(off_in + byte_idx);
                        buf_out->write(off_out + byte_idx, v);
                    }
                } else {
                    for (auto i : dynamic_range(in.size())) {
                        out[i] = in[i];
                    }
                }
            }

            // Fast path: constant indices -> precompute scatter positions at C++ time
            // Only unroll for small index tensors to avoid shader code bloat
            auto try_scatter_elements_fast_path = [&]() -> bool {
                if (!indices.is_constant()) return false;
                if (indices.size() > 128) return false;// avoid excessive unrolling

                auto &idx_const = static_cast<NNConstTensor<int> const &>(indices);
                auto const &cpu_idx = idx_const.const_data();
                auto const &idx_shape = indices.shape();

                // Precompute output positions for each index element
                // For each linear_idx in [0, idx.size()):
                //   decompose into coords, replace axis coord with cpu_idx[linear_idx]
                //   compute out_linear = sum(coord_d * out.strides[d])
                for (uint32_t linear_idx = 0; linear_idx < indices.size(); ++linear_idx) {
                    // Decompose linear_idx into multi-dim coords
                    uint32_t out_linear = 0;
                    uint32_t remaining = linear_idx;
                    for (uint32_t d = 0; d < ndim; ++d) {
                        uint32_t coord = remaining / idx.strides()[d];
                        remaining = remaining % idx.strides()[d];
                        if (d == static_cast<uint32_t>(axis)) {
                            int idx_val = static_cast<int>(cpu_idx[linear_idx]);
                            if (idx_val < 0) idx_val += static_cast<int>(data.shape()[d]);
                            out_linear += static_cast<uint32_t>(idx_val) * out.strides()[d];
                        } else {
                            out_linear += coord * out.strides()[d];
                        }
                    }
                    // Emit direct write with compile-time constant position
                    if (out.container().is_byte_buffer()) {
                        apply_reduction_buf<ST>(
                            out.container().get_byte_buffer(),
                            static_cast<uint>(out.container().get_byte_offset()),
                            out_linear,
                            upd[linear_idx],
                            reduction_);
                    } else {
                        apply_reduction(out, out_linear, upd, linear_idx, reduction_);
                    }
                }
                return true;
            };

            if (try_scatter_elements_fast_path()) return;

            // Apply updates at indexed positions
            for (auto linear_idx : dynamic_range(idx.size())) {
                auto index_val = idx[linear_idx].cast<uint>();
                auto out_linear = def(0u);
                for_each_dim(linear_idx, idx.strides(), ndim, [&](uint32_t d, auto coord) {
                                 if (d == static_cast<uint32_t>(axis)) {
                                     out_linear += index_val * out.strides()[d];
                                 } else {
                                     out_linear += coord * out.strides()[d];
                                 } }, idx.size());
                if (out.container().is_byte_buffer()) {
                    apply_reduction_buf<ST>(
                        out.container().get_byte_buffer(),
                        static_cast<uint>(out.container().get_byte_offset()),
                        out_linear,
                        upd[linear_idx],
                        reduction_);
                } else {
                    apply_reduction(out, out_linear, upd, linear_idx, reduction_);
                }
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(ScatterElements) {
    int64_t axis = 0;
    Reduction reduction = Reduction::NONE;
    if (auto p = node.try_get_attr("axis"))
        axis = p->get<onnx::AttributeType::INT>();
    if (auto p = node.try_get_attr("reduction"))
        reduction = parse_reduction(p->get<onnx::AttributeType::STRING>());
    return std::make_unique<ScatterElements>(axis, reduction);
};

// Scatter (deprecated alias for ScatterElements)
REGISTER_TO_DEFAULT_OPSET(Scatter) {
    int64_t axis = 0;
    if (auto p = node.try_get_attr("axis"))
        axis = p->get<onnx::AttributeType::INT>();
    return std::make_unique<ScatterElements>(axis, Reduction::NONE, "Scatter");
};

// ScatterND: scatter updates into data at positions specified by N-d indices.
// ONNX spec: input[0]=data, input[1]=indices, input[2]=updates; attribute reduction (default "none")
class ScatterND : public Operator {
private:
    Reduction reduction_;

public:
    ScatterND(Reduction reduction) : Operator("ScatterND"), reduction_(reduction) {}

    bool can_operate_inplace() const override { return true; }

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
        LUISA_ASSERT(inputs.size() == 3 && outputs.size() == 1, "ScatterND requires 3 inputs and 1 output.");
        auto &data = inputs[0].get();
        auto &indices = inputs[1].get();
        auto &updates = inputs[2].get();
        auto &output = outputs[0].get();

        auto const &idx_shape = indices.shape();
        auto idx_ndim = indices.ndim();
        auto last_idx_dim = idx_shape[idx_ndim - 1];
        auto data_ndim = data.ndim();

        LUISA_ASSERT(data.element_type() == updates.element_type() && data.element_type() == output.element_type(),
                     "ScatterND: data, updates and output must have the same element type.");
        LUISA_ASSERT(indices.element_type() == typeid(int) || indices.element_type() == typeid(slong),
                     "ScatterND: indices must be int or int64 type.");

        visit_typeid<NNTypeList>(data.element_type(), [&]<typename T>() {
            auto &in = static_cast<NNTensor<T> &>(data);
            auto &idx = static_cast<NNTensor<int> &>(indices);
            auto &upd = static_cast<NNTensor<T> &>(updates);
            auto &out = static_cast<NNTensor<T> &>(output);
            using ST = typename NNTensor<T>::value_type;

            // Copy data to output — but skip if they share the same
            // underlying storage (in-place reuse from memory pool).
            if (!in.container().shares_storage_with(out.container())) {
                if constexpr (detail::VecDispatch<ST>::supported) {
                    if (detail::all_byte_buffer(in, out)) {
                        detail::vectorized_unary<ST>(in, out, [](auto v) { return v; });
                    } else {
                        for (auto i : dynamic_range(in.size())) {
                            out[i] = in[i];
                        }
                    }
                } else if (detail::all_byte_buffer(in, out)) {
                    auto buf_in = in.container().get_byte_buffer();
                    auto buf_out = out.container().get_byte_buffer();
                    auto off_in = static_cast<uint>(in.container().get_byte_offset());
                    auto off_out = static_cast<uint>(out.container().get_byte_offset());
                    for (auto i : dynamic_range(in.size())) {
                        auto byte_idx = i * static_cast<uint>(sizeof(ST));
                        auto v = buf_in->read<ST>(off_in + byte_idx);
                        buf_out->write(off_out + byte_idx, v);
                    }
                } else {
                    for (auto i : dynamic_range(in.size())) {
                        out[i] = in[i];
                    }
                }
            }

            // Fast path: check if indices are constant consecutive integers
            // e.g. [[0], [1], [2], ...] or [[S], [S+1], [S+2], ...]
            // with reduction=NONE -> linear copy with offset
            auto try_scatter_nd_fast_path = [&]() -> bool {
                if (reduction_ != Reduction::NONE) return false;
                if (!indices.is_constant()) return false;
                // last_idx_dim must be 1 for simple row-level scatter
                if (last_idx_dim != 1) return false;

                auto &idx_const = static_cast<NNConstTensor<int> const &>(indices);
                auto const &cpu_idx = idx_const.const_data();
                auto num_indices = upd.shape()[0];// number of scatter rows

                // Verify indices are [S, S+1, S+2, ..., S+N-1] for some S >= 0
                if (num_indices == 0) return false;
                int start_val = cpu_idx[0];
                if (start_val < 0) return false;
                for (uint32_t i = 1; i < num_indices; ++i) {
                    if (cpu_idx[i] != start_val + static_cast<int>(i)) {
                        return false;
                    }
                }

                // Indices are consecutive starting from start_val
                // Each row has row_size elements = data.strides()[0]
                auto row_size = static_cast<uint32_t>(out.strides()[0]);
                auto out_offset = static_cast<uint32_t>(start_val) * row_size;

                // Try vectorized linear copy with offset
                if constexpr (detail::VecDispatch<ST>::supported) {
                    if (detail::all_byte_buffer(upd, out)) {
                        auto buf_upd = upd.container().get_byte_buffer();
                        auto buf_out = out.container().get_byte_buffer();
                        auto off_upd = static_cast<uint>(upd.container().get_byte_offset());
                        auto off_out = static_cast<uint>(out.container().get_byte_offset()) + out_offset * static_cast<uint>(sizeof(ST));
                        auto n = upd.size();
                        auto vec_n = static_cast<uint>(n / detail::VecDispatch<ST>::N);
                        auto rem = static_cast<uint>(n % detail::VecDispatch<ST>::N);
                        using VecT = typename detail::VecDispatch<ST>::VecT;
                        for (auto i : dynamic_range(vec_n)) {
                            auto byte_idx = i * static_cast<uint>(sizeof(VecT));
                            auto v = buf_upd->read<VecT>(off_upd + byte_idx);
                            buf_out->write(off_out + byte_idx, v);
                        }
                        for (auto i : dynamic_range(rem)) {
                            auto idx_elem = vec_n * detail::VecDispatch<ST>::N + i;
                            auto byte_idx = idx_elem * static_cast<uint>(sizeof(ST));
                            auto v = buf_upd->read<ST>(off_upd + byte_idx);
                            buf_out->write(off_out + byte_idx, v);
                        }
                        return true;
                    }
                }

                // Scalar ByteBuffer fallback
                if (detail::all_byte_buffer(upd, out)) {
                    auto buf_upd = upd.container().get_byte_buffer();
                    auto buf_out = out.container().get_byte_buffer();
                    auto off_upd = static_cast<uint>(upd.container().get_byte_offset());
                    auto off_out = static_cast<uint>(out.container().get_byte_offset()) + out_offset * static_cast<uint>(sizeof(ST));
                    for (auto i : dynamic_range(upd.size())) {
                        auto byte_idx = i * static_cast<uint>(sizeof(ST));
                        auto v = buf_upd->read<ST>(off_upd + byte_idx);
                        buf_out->write(off_out + byte_idx, v);
                    }
                    return true;
                }

                // Local/View fallback
                for (auto i : dynamic_range(upd.size())) {
                    out[i + out_offset] = upd[i];
                }
                return true;
            };

            if (try_scatter_nd_fast_path()) return;

            // Apply updates
            for (auto linear_upd : dynamic_range(upd.size())) {
                auto upd_ndim = upd.ndim();

                // First (idx_ndim - 1) coords map to the outer indices dims
                auto idx_base = def(0u);
                auto out_linear = def(0u);
                auto remaining = linear_upd;
                for (uint32_t d = 0; d < idx_ndim - 1; ++d) {
                    auto coord = remaining / upd.strides()[d];
                    remaining = remaining % upd.strides()[d];
                    idx_base += coord * idx.strides()[d];
                }

                // Build output linear index from the index values
                for (uint32_t k = 0; k < last_idx_dim; ++k) {
                    out_linear += idx[idx_base + k].cast<uint>() * out.strides()[k];
                }

                // Remaining coords map to data slice dims
                for (uint32_t d = idx_ndim - 1; d < upd_ndim; ++d) {
                    auto coord = remaining / upd.strides()[d];
                    remaining = remaining % upd.strides()[d];
                    auto data_dim = last_idx_dim + (d - (idx_ndim - 1));
                    out_linear += coord * out.strides()[data_dim];
                }

                if (out.container().is_byte_buffer()) {
                    apply_reduction_buf<ST>(
                        out.container().get_byte_buffer(),
                        static_cast<uint>(out.container().get_byte_offset()),
                        out_linear,
                        upd[linear_upd],
                        reduction_);
                } else {
                    apply_reduction(out, out_linear, upd, linear_upd, reduction_);
                }
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(ScatterND) {
    Reduction reduction = Reduction::NONE;
    if (auto p = node.try_get_attr("reduction"))
        reduction = parse_reduction(p->get<onnx::AttributeType::STRING>());
    return std::make_unique<ScatterND>(reduction);
};

}// namespace lcml::onnx
