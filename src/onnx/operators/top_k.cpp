#include <luisa/core/stl/memory.h>
#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

namespace lcml::onnx {

// Trait for TopK-supported types: native floating point + quantized FP4/FP8
template<typename T>
struct IsTopKSupported : std::bool_constant<
    luisa::is_floating_point_v<T> ||
    std::is_same_v<T, FP4E2M1> ||
    std::is_same_v<T, FP8E4M3FN> ||
    std::is_same_v<T, FP8E5M2>> {};

// TopK: returns the top K largest/smallest values and their indices along a given axis.
// ONNX spec: inputs: X, K (scalar int64); attributes: axis (default -1), largest (default 1), sorted (default 1)
// Output: Values, Indices
class TopK : public Operator {
private:
    int64_t axis_;
    int32_t largest_;
    int32_t sorted_;

public:
    TopK(int64_t axis, int32_t largest, int32_t sorted)
        : Operator("TopK"), axis_(axis), largest_(largest), sorted_(sorted) {}

    void forward(luisa::span<std::reference_wrapper<ITensor>> inputs,
                 luisa::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 2 && outputs.size() == 2,
                     "TopK requires 2 inputs and 2 outputs.");
#endif
        auto &X = inputs[0].get();
        auto &Values = outputs[0].get();
        auto &Indices = outputs[1].get();

        auto const &x_shape = X.shape();
        auto ndim = x_shape.size();
        auto axis = static_cast<uint32_t>(axis_ < 0 ? axis_ + ndim : axis_);
        auto axis_size = x_shape[axis];

        auto const &v_shape = Values.shape();
        uint32_t K = v_shape[axis];

#ifndef NDEBUG
        LUISA_ASSERT(X.element_type_index() == Values.element_type_index(),
                     "TopK: X and Values must have the same element type.");
        LUISA_ASSERT(Indices.element_type_index() == refl::type_index_of<int32_t>() || Indices.element_type_index() == refl::type_index_of<slong>(),
                     "TopK: Indices must be int or int64 type.");
        LUISA_ASSERT(K <= axis_size, "TopK: K must be <= axis_size.");
        LUISA_ASSERT(axis_size > 0, "TopK: axis_size must be > 0.");
#endif

        uint32_t ax_stride = 1;
        for (size_t d = axis + 1; d < ndim; ++d) ax_stride *= x_shape[d];

        visit_type_index<NNFilteredTypeList<IsTopKSupported>>(X.element_type_index(), [&]<typename T>() {
            using VT = nn_storage_type_t<T>;
            auto &x = static_cast<NNTensor<T> &>(X);
            auto &vals = static_cast<NNTensor<T> &>(Values);
            auto &idxs = static_cast<NNTensor<int32_t> &>(Indices);

            // Performance: use half-precision compare for float inputs,
            // and dequantize FP4/FP8 to half. Precision loss is acceptable per rules.
            using CT = std::conditional_t<
                std::is_same_v<VT, float> ||
                std::is_same_v<T, FP4E2M1> ||
                std::is_same_v<T, FP8E4M3FN> ||
                std::is_same_v<T, FP8E5M2>,
                half, VT>;

            auto get_cmp = [&](Var<VT> val) -> Var<CT> {
                if constexpr (std::is_same_v<VT, float>) {
                    return cast<half>(val);
                } else if constexpr (std::is_same_v<T, FP4E2M1>) {
                    static auto c = fp4e2m1_to_float();
                    return c(val.bits);
                } else if constexpr (std::is_same_v<T, FP8E4M3FN>) {
                    static auto c = fp8e4m3_to_float();
                    return c(val.bits);
                } else if constexpr (std::is_same_v<T, FP8E5M2>) {
                    static auto c = fp8e5m2_to_float();
                    return c(val.bits);
                } else {
                    return val;
                }
            };

            // --- Buffer read optimization setup ---
            // For native float/half, use direct ByteBuffer reads to bypass DynamicArray::operator[] overhead.
            // When ax_stride==1, use vectorized float4/half4 reads for the load phase.
            bool x_is_bb = false;
            Var<ByteBuffer> *buf_x = nullptr;
            uint off_x = 0;
            bool can_vec = false;
            if constexpr (std::is_same_v<VT, float> || std::is_same_v<VT, half>) {
                x_is_bb = x.container().is_byte_buffer();
                if (x_is_bb) {
                    buf_x = x.container().get_byte_buffer();
                    off_x = static_cast<uint>(x.container().get_byte_offset());
                    can_vec = (ax_stride == 1);
                }
            }

            auto read_x_scalar = [&](auto idx) -> Var<VT> {
                if constexpr (std::is_same_v<VT, float> || std::is_same_v<VT, half>) {
                    if (x_is_bb) {
                        return buf_x->read<VT>(off_x + idx * static_cast<uint>(sizeof(VT)));
                    }
                }
                return x[idx];
            };

            for (auto out_linear : dynamic_range(vals.size())) {
                auto base = def(0u);
                auto k_coord = def(0u);
                for_each_dim(out_linear, vals.strides(), ndim, [&](uint32_t d, auto coord) {
                    if (d == axis) {
                        k_coord = coord;
                    } else {
                        base += coord * x.strides()[d];
                    }
                }, vals.size());

                // Allocate local arrays to cache the slice.
                // This reduces global memory traffic from O(N^2) to O(N) per output element.
                DynamicArray<VT> local_vals(axis_size);
                DynamicArray<CT> local_cmps(axis_size);

                // Vectorized load: read 4 elements at once when contiguous along axis.
                if constexpr (std::is_same_v<VT, float> || std::is_same_v<VT, half>) {
                    if (can_vec) {
                        using VecT = typename detail::VecDispatch<VT>::VecT;
                        uint32_t vec_end = (axis_size / 4u) * 4u;
                        for (uint32_t i = 0; i < vec_end; i += 4u) {
                            auto byte_off = off_x + (base + i) * static_cast<uint>(sizeof(VT));
                            auto v = buf_x->read<VecT>(byte_off);
                            local_vals[i + 0] = v.x;
                            local_vals[i + 1] = v.y;
                            local_vals[i + 2] = v.z;
                            local_vals[i + 3] = v.w;
                            local_cmps[i + 0] = get_cmp(v.x);
                            local_cmps[i + 1] = get_cmp(v.y);
                            local_cmps[i + 2] = get_cmp(v.z);
                            local_cmps[i + 3] = get_cmp(v.w);
                        }
                        for (uint32_t i = vec_end; i < axis_size; ++i) {
                            auto v = read_x_scalar(base + i);
                            local_vals[i] = v;
                            local_cmps[i] = get_cmp(v);
                        }
                    } else {
                        for (auto i : dynamic_range(axis_size)) {
                            auto v = read_x_scalar(base + i * ax_stride);
                            local_vals[i] = v;
                            local_cmps[i] = get_cmp(v);
                        }
                    }
                } else {
                    for (auto i : dynamic_range(axis_size)) {
                        auto v = read_x_scalar(base + i * ax_stride);
                        local_vals[i] = v;
                        local_cmps[i] = get_cmp(v);
                    }
                }

                // Initialize with first element so output is always valid
                auto result_val = def(local_vals[0]);
                auto result_idx = def(0);

                // Branch-free ranking from local memory.
                // Each candidate counts how many elements should precede it in sorted order.
                for (auto candidate : dynamic_range(axis_size)) {
                    auto cand_cmp = local_cmps[candidate];
                    auto cand_is_nan = luisa::compute::isnan(cand_cmp);
                    auto cand_val = local_vals[candidate];
                    auto count = def(0u);

                    for (auto other : dynamic_range(axis_size)) {
                        auto other_cmp = local_cmps[other];
                        auto other_is_nan = luisa::compute::isnan(other_cmp);

                        if (largest_) {
                            // NaN is treated as the largest value (ONNX semantics)
                            auto inc = select(0u, 1u,
                                (other_is_nan & !cand_is_nan) |
                                (other_is_nan & cand_is_nan & (other < candidate)) |
                                (!other_is_nan & !cand_is_nan &
                                 ((other_cmp > cand_cmp) |
                                  (other_cmp == cand_cmp & other < candidate)))
                            );
                            count += inc;
                        } else {
                            // NaN is treated as largest (worst when searching smallest)
                            auto inc = select(0u, 1u,
                                (!other_is_nan & cand_is_nan) |
                                (other_is_nan & cand_is_nan & (other < candidate)) |
                                (!other_is_nan & !cand_is_nan &
                                 ((other_cmp < cand_cmp) |
                                  (other_cmp == cand_cmp & other < candidate)))
                            );
                            count += inc;
                        }
                    }

                    $if (count == k_coord) {
                        result_val = cand_val;
                        result_idx = candidate.cast<int>();
                    };
                }

                vals[out_linear] = result_val;
                idxs[out_linear] = result_idx;
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(TopK) {
    int64_t axis = -1;
    int32_t largest = 1, sorted = 1;
    if (auto p = node.try_get_attr("axis")) axis = p->get<onnx::AttributeType::INT>();
    if (auto p = node.try_get_attr("largest")) largest = p->get<onnx::AttributeType::INT>();
    if (auto p = node.try_get_attr("sorted")) sorted = p->get<onnx::AttributeType::INT>();
    return luisa::make_unique<TopK>(axis, largest, sorted);
};

}// namespace lcml::onnx
