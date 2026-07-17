#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

#include <luisa/core/stl/memory.h>

namespace lcml::onnx {

// Constant: produces a constant tensor (value from attribute).
// In this DSL context, the output is pre-filled by the runtime.
class Constant : public Operator {
public:
    Constant() : Operator("Constant") {}

    void forward(luisa::span<std::reference_wrapper<ITensor>> inputs,
                 luisa::span<std::reference_wrapper<ITensor>> outputs) override {
        // Constant op: output is already set by the runtime from the attribute.
        // Nothing to do in forward pass.
    }
};

REGISTER_TO_DEFAULT_OPSET(Constant) {
    return luisa::make_unique<Constant>();
};

// ConstantOfShape: generates a tensor of a given shape filled with a constant value.
// ONNX spec: input[0] = shape (1-D int64); attribute: value (scalar tensor, default 0)
class ConstantOfShape : public Operator {
private:
    float value_;

public:
    ConstantOfShape(float value) : Operator("ConstantOfShape"), value_(value) {}

    void forward(luisa::span<std::reference_wrapper<ITensor>> inputs,
                 luisa::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(outputs.size() == 1, "ConstantOfShape requires 1 output.");
#endif
        auto &Y = outputs[0].get();

        visit_type_index<NNTypeList>(Y.element_type_index(), [&]<typename T>() {
            using VT = nn_storage_type_t<T>;
            auto &y = static_cast<NNTensor<T> &>(Y);
            if constexpr (std::is_same_v<VT, FP4E2M1>) {
                auto enc = fp4e2m1_from_float();
                Var<VT> val;
                val.bits = enc(def(static_cast<half>(value_))).cast<uint16_t>();
                for (auto i : dynamic_range(y.size())) { y[i] = val; }
            } else if constexpr (std::is_same_v<VT, FP8E4M3FN>) {
                auto enc = fp8e4m3_from_float();
                Var<VT> val;
                val.bits = enc(def(static_cast<half>(value_))).cast<uint16_t>();
                for (auto i : dynamic_range(y.size())) { y[i] = val; }
            } else if constexpr (std::is_same_v<VT, FP8E5M2>) {
                auto enc = fp8e5m2_from_float();
                Var<VT> val;
                val.bits = enc(def(static_cast<half>(value_))).cast<uint16_t>();
                for (auto i : dynamic_range(y.size())) { y[i] = val; }
            } else if constexpr (std::is_same_v<VT, FP16Quantized>) {
                Var<VT> val;
                val.bits = def(static_cast<half>(value_));
                for (auto i : dynamic_range(y.size())) { y[i] = val; }
            } else {
                auto val = Var<VT>{VT(value_)};
                if constexpr (detail::VecDispatch<VT>::supported) {
                    if (y.container().is_byte_buffer()) {
                        detail::vectorized_scalar_b<VT>(
                            y, val, y,
                            [&](auto, auto b) { return b; });
                    } else {
                        for (auto i : dynamic_range(y.size())) { y[i] = val; }
                    }
                } else {
                    for (auto i : dynamic_range(y.size())) { y[i] = val; }
                }
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(ConstantOfShape) {
    float value = 0.0f;
    if (auto p = node.try_get_attr("value"))
        value = p->get<onnx::AttributeType::FLOAT>();
    return luisa::make_unique<ConstantOfShape>(value);
};

// Range: generates a 1-D tensor of values [start, start+delta, start+2*delta, ...)
// ONNX spec: inputs: start, limit, delta (all scalar tensors)
class Range : public Operator {
public:
    Range() : Operator("Range") {}

    void forward(luisa::span<std::reference_wrapper<ITensor>> inputs,
                 luisa::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 3 && outputs.size() == 1, "Range requires 3 inputs and 1 output.");
#endif
        auto &Y = outputs[0].get();

        visit_type_index<NNFilteredTypeList<IsFloatingPoint>>(Y.element_type_index(), [&]<typename T>() {
            using VT = nn_storage_type_t<T>;
            auto &start_t = static_cast<NNTensor<T> &>(inputs[0].get());
            auto &delta_t = static_cast<NNTensor<T> &>(inputs[2].get());
            auto &y = static_cast<NNTensor<T> &>(Y);

            auto start = start_t[0u];
            auto delta = delta_t[0u];

            for (auto i : dynamic_range(y.size())) {
                y[i] = fma(i.cast<VT>(), delta, start);
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(Range) {
    return luisa::make_unique<Range>();
};

// EyeLike: generates an identity matrix of the same shape as the input.
// ONNX spec: attribute: dtype (optional), k (diagonal offset, default 0)
class EyeLike : public Operator {
private:
    int64_t k_;

public:
    EyeLike(int64_t k) : Operator("EyeLike"), k_(k) {}

    void forward(luisa::span<std::reference_wrapper<ITensor>> inputs,
                 luisa::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 1 && outputs.size() == 1, "EyeLike requires 1 input and 1 output.");
#endif
        auto &Y = outputs[0].get();
        auto const &y_shape = Y.shape();

#ifndef NDEBUG
        LUISA_ASSERT(y_shape.size() == 2, "EyeLike: input must be 2D.");
#endif

        uint32_t rows = y_shape[0], cols = y_shape[1];

        visit_type_index<NNTypeList>(Y.element_type_index(), [&]<typename T>() {
            using VT = nn_storage_type_t<T>;
            auto &y = static_cast<NNTensor<T> &>(Y);
            auto k_int = def(static_cast<int32_t>(k_));

            if constexpr (std::is_same_v<VT, FP4E2M1>) {
                auto enc = fp4e2m1_from_float();
                Var<VT> zero;
                Var<VT> one;
                one.bits = enc(def(static_cast<half>(1.0f))).cast<uint16_t>();
                for (auto r : dynamic_range(rows)) {
                    for (auto c : dynamic_range(cols)) {
                        auto col_with_offset = r.cast<int>() + k_int;
                        y[r * y.strides()[0] + c * y.strides()[1]] =
                            select(zero, one, c.cast<int>() == col_with_offset);
                    }
                }
            } else if constexpr (std::is_same_v<VT, FP8E4M3FN>) {
                auto enc = fp8e4m3_from_float();
                Var<VT> zero;
                Var<VT> one;
                one.bits = enc(def(static_cast<half>(1.0f))).cast<uint16_t>();
                for (auto r : dynamic_range(rows)) {
                    for (auto c : dynamic_range(cols)) {
                        auto col_with_offset = r.cast<int>() + k_int;
                        y[r * y.strides()[0] + c * y.strides()[1]] =
                            select(zero, one, c.cast<int>() == col_with_offset);
                    }
                }
            } else if constexpr (std::is_same_v<VT, FP8E5M2>) {
                auto enc = fp8e5m2_from_float();
                Var<VT> zero;
                Var<VT> one;
                one.bits = enc(def(static_cast<half>(1.0f))).cast<uint16_t>();
                for (auto r : dynamic_range(rows)) {
                    for (auto c : dynamic_range(cols)) {
                        auto col_with_offset = r.cast<int>() + k_int;
                        y[r * y.strides()[0] + c * y.strides()[1]] =
                            select(zero, one, c.cast<int>() == col_with_offset);
                    }
                }
            } else if constexpr (std::is_same_v<VT, FP16Quantized>) {
                Var<VT> zero;
                Var<VT> one;
                one.bits = def(static_cast<half>(1.0f));
                for (auto r : dynamic_range(rows)) {
                    for (auto c : dynamic_range(cols)) {
                        auto col_with_offset = r.cast<int>() + k_int;
                        y[r * y.strides()[0] + c * y.strides()[1]] =
                            select(zero, one, c.cast<int>() == col_with_offset);
                    }
                }
            } else {
                for (auto r : dynamic_range(rows)) {
                    for (auto c : dynamic_range(cols)) {
                        auto col_with_offset = r.cast<int>() + k_int;
                        y[r * y.strides()[0] + c * y.strides()[1]] =
                            select(Var<VT>{VT{0}}, Var<VT>{VT{1}}, c.cast<int>() == col_with_offset);
                    }
                }
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(EyeLike) {
    int64_t k = 0;
    if (auto p = node.try_get_attr("k"))
        k = p->get<onnx::AttributeType::INT>();
    return luisa::make_unique<EyeLike>(k);
};

// RandomUniform: generates a tensor with random uniform values.
// ONNX spec: attributes: dtype, high (default 1.0), low (default 0.0), seed, shape
// Since DSL may not have true random, we use a simple LCG pseudo-random
class RandomUniform : public Operator {
private:
    float low_;
    float high_;
    float seed_;

public:
    RandomUniform(float low, float high, float seed)
        : Operator("RandomUniform"), low_(low), high_(high), seed_(seed) {}

    void forward(luisa::span<std::reference_wrapper<ITensor>> inputs,
                 luisa::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(outputs.size() == 1, "RandomUniform requires 1 output.");
#endif
        auto &Y = outputs[0].get();

        visit_type_index<NNFilteredTypeList<IsFloatingPoint>>(Y.element_type_index(), [&]<typename T>() {
            using VT = nn_storage_type_t<T>;
            auto &y = static_cast<NNTensor<T> &>(Y);
            auto low = Var<VT>{VT(low_)};
            auto range = Var<VT>{VT(high_ - low_)};

            // Simple LCG: state = state * 1103515245 + 12345
            auto state = def(static_cast<uint32_t>(seed_ != 0.0f ? seed_ : 42.0f));
            for (auto i : dynamic_range(y.size())) {
                state = state * 1103515245u + 12345u;
                // Map to [0, 1)
                auto norm = (state >> 8u).cast<VT>() / Var<VT>{VT(16777216.0f)};
                y[i] = fma(norm, range, low);
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(RandomUniform) {
    float low = 0.0f, high = 1.0f, seed = 0.0f;
    if (auto p = node.try_get_attr("low")) low = p->get<onnx::AttributeType::FLOAT>();
    if (auto p = node.try_get_attr("high")) high = p->get<onnx::AttributeType::FLOAT>();
    if (auto p = node.try_get_attr("seed")) seed = p->get<onnx::AttributeType::FLOAT>();
    return luisa::make_unique<RandomUniform>(low, high, seed);
};

}// namespace lcml::onnx
