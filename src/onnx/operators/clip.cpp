#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

namespace lcml::onnx {

// Clip: clamp(x, min, max)
// ONNX spec (opset 11+): min/max are optional inputs (not attributes)
// If min input is not provided, no lower bound; if max not provided, no upper bound.
class Clip : public Operator {
public:
    Clip() : Operator("Clip") {}

    /// Element-wise clamp: output[i] = clamp(input[i], min, max). Safe for in-place.
    bool can_operate_inplace() const override { return true; }

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() >= 1 && inputs.size() <= 3 && outputs.size() == 1,
                     "Clip requires 1-3 inputs and 1 output.");
#endif
        auto &input = inputs[0].get();
        auto &output = outputs[0].get();
#ifndef NDEBUG
        LUISA_ASSERT(input.element_type() == output.element_type(),
                     "Clip: input and output must have the same element type.");
        LUISA_ASSERT(input.size() == output.size(),
                     "Clip: input and output must have the same size.");
#endif

        ITensor *min_tensor = (inputs.size() >= 2) ? &inputs[1].get() : nullptr;
        ITensor *max_tensor = (inputs.size() >= 3) ? &inputs[2].get() : nullptr;

        // Handle empty optional inputs (size==0 means not provided)
        if (min_tensor && min_tensor->size() == 0) min_tensor = nullptr;
        if (max_tensor && max_tensor->size() == 0) max_tensor = nullptr;

        visit_typeid<NNFilteredTypeList<IsFloatingPoint>>(input.element_type(), [&]<typename T>() {
            auto &in = static_cast<NNTensor<T> &>(input);
            auto &out = static_cast<NNTensor<T> &>(output);
            using ST = typename NNTensor<T>::value_type;

            bool min_scalar = min_tensor && min_tensor->size() == 1;
            bool max_scalar = max_tensor && max_tensor->size() == 1;

            auto scalar_loop = [&](auto op) {
                for (auto i : dynamic_range(in.size())) {
                    out[i] = op(i);
                }
            };

            if (min_scalar && max_scalar) {
                // Both bounds are scalar: single clamp instruction
                auto lo = static_cast<NNTensor<T> &>(*min_tensor)[0u];
                auto hi = static_cast<NNTensor<T> &>(*max_tensor)[0u];
                if constexpr (detail::VecDispatch<ST>::supported) {
                    if (detail::all_byte_buffer(in, out)) {
                        using VecT = typename detail::VecDispatch<ST>::VecT;
                        auto buf_in = in.container().get_byte_buffer();
                        auto buf_o = out.container().get_byte_buffer();
                        auto off_in = static_cast<uint>(in.container().get_byte_offset());
                        auto off_o = static_cast<uint>(out.container().get_byte_offset());
                        auto n = out.size();
                        auto vec_n = static_cast<uint>(n / detail::VecDispatch<ST>::N);
                        auto rem = static_cast<uint>(n % detail::VecDispatch<ST>::N);
                        auto vec_lo = detail::VecDispatch<ST>::broadcast(lo);
                        auto vec_hi = detail::VecDispatch<ST>::broadcast(hi);
                        for (auto i : dynamic_range(vec_n)) {
                            auto byte_idx = i * static_cast<uint>(sizeof(VecT));
                            auto v = buf_in->read<VecT>(off_in + byte_idx);
                            auto vr = clamp(v, vec_lo, vec_hi);
                            buf_o->write(off_o + byte_idx, vr);
                        }
                        for (auto i : dynamic_range(rem)) {
                            auto idx = vec_n * detail::VecDispatch<ST>::N + i;
                            out[idx] = clamp(in[idx], lo, hi);
                        }
                        return;
                    }
                }
                scalar_loop([&](auto i) { return clamp(in[i], lo, hi); });
            } else if (min_scalar && !max_tensor) {
                // Only lower bound (scalar)
                auto lo = static_cast<NNTensor<T> &>(*min_tensor)[0u];
                if constexpr (detail::VecDispatch<ST>::supported) {
                    if (detail::all_byte_buffer(in, out)) {
                        detail::vectorized_scalar_b<ST>(
                            in, lo, out,
                            [](auto a, auto b) { return max(a, b); });
                        return;
                    }
                }
                scalar_loop([&](auto i) { return max(in[i], lo); });
            } else if (max_scalar && !min_tensor) {
                // Only upper bound (scalar)
                auto hi = static_cast<NNTensor<T> &>(*max_tensor)[0u];
                if constexpr (detail::VecDispatch<ST>::supported) {
                    if (detail::all_byte_buffer(in, out)) {
                        detail::vectorized_scalar_b<ST>(
                            in, hi, out,
                            [](auto a, auto b) { return min(a, b); });
                        return;
                    }
                }
                scalar_loop([&](auto i) { return min(in[i], hi); });
            } else {
                // General case: at least one bound is a non-scalar tensor.
                // Hoist all uniform conditions out of the loop to eliminate warp divergence.
                if (min_tensor && max_tensor) {
                    auto &min_t = static_cast<NNTensor<T> &>(*min_tensor);
                    auto &max_t = static_cast<NNTensor<T> &>(*max_tensor);
                    if (min_scalar) {
                        // Scalar min + tensor max
                        auto lo = min_t[0u];
                        if constexpr (detail::VecDispatch<ST>::supported) {
                            if (detail::all_byte_buffer(in, out)) {
                                detail::vectorized_scalar_b<ST>(
                                    in, lo, out,
                                    [](auto a, auto b) { return max(a, b); });
                                if (detail::all_byte_buffer(out, max_t, out)) {
                                    detail::vectorized_same_shape<ST>(
                                        out, max_t, out,
                                        [](auto a, auto b) { return min(a, b); });
                                } else {
                                    scalar_loop([&](auto i) { return min(out[i], max_t[i]); });
                                }
                                return;
                            }
                        }
                        scalar_loop([&](auto i) { return min(max(in[i], lo), max_t[i]); });
                    } else if (max_scalar) {
                        // Tensor min + scalar max
                        auto hi = max_t[0u];
                        if constexpr (detail::VecDispatch<ST>::supported) {
                            if (detail::all_byte_buffer(in, out)) {
                                detail::vectorized_same_shape<ST>(
                                    in, min_t, out,
                                    [](auto a, auto b) { return max(a, b); });
                                detail::vectorized_scalar_b<ST>(
                                    out, hi, out,
                                    [](auto a, auto b) { return min(a, b); });
                                return;
                            }
                        }
                        scalar_loop([&](auto i) { return min(max(in[i], min_t[i]), hi); });
                    } else {
                        // Both non-scalar tensors
                        if constexpr (detail::VecDispatch<ST>::supported) {
                            if (detail::all_byte_buffer(in, min_t, out)) {
                                detail::vectorized_same_shape<ST>(
                                    in, min_t, out,
                                    [](auto a, auto b) { return max(a, b); });
                                if (detail::all_byte_buffer(out, max_t, out)) {
                                    detail::vectorized_same_shape<ST>(
                                        out, max_t, out,
                                        [](auto a, auto b) { return min(a, b); });
                                } else {
                                    scalar_loop([&](auto i) { return min(out[i], max_t[i]); });
                                }
                                return;
                            }
                        }
                        scalar_loop([&](auto i) { return min(max(in[i], min_t[i]), max_t[i]); });
                    }
                } else if (min_tensor) {
                    // Only lower bound (tensor)
                    auto &min_t = static_cast<NNTensor<T> &>(*min_tensor);
                    if constexpr (detail::VecDispatch<ST>::supported) {
                        if (detail::all_byte_buffer(in, min_t, out)) {
                            detail::vectorized_same_shape<ST>(
                                in, min_t, out,
                                [](auto a, auto b) { return max(a, b); });
                            return;
                        }
                    }
                    scalar_loop([&](auto i) { return max(in[i], min_t[i]); });
                } else if (max_tensor) {
                    // Only upper bound (tensor)
                    auto &max_t = static_cast<NNTensor<T> &>(*max_tensor);
                    if constexpr (detail::VecDispatch<ST>::supported) {
                        if (detail::all_byte_buffer(in, max_t, out)) {
                            detail::vectorized_same_shape<ST>(
                                in, max_t, out,
                                [](auto a, auto b) { return min(a, b); });
                            return;
                        }
                    }
                    scalar_loop([&](auto i) { return min(in[i], max_t[i]); });
                } else {
                    // No bounds: identity copy
                    if constexpr (detail::VecDispatch<ST>::supported) {
                        if (detail::all_byte_buffer(in, out)) {
                            detail::vectorized_unary<ST>(
                                in, out,
                                [&](auto v) { return v; });
                            return;
                        }
                    }
                    scalar_loop([&](auto i) { return in[i]; });
                }
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(Clip) {
    return std::make_unique<Clip>();
};

}// namespace lcml::onnx
