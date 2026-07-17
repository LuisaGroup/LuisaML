#include <luisa/core/stl/string.h>
#include <luisa/core/stl/memory.h>
#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

namespace lcml::onnx {

template<typename Op, typename TypeList = NNTypeList>
class Unary : public Operator {
private:
    template<typename T>
    void apply_to(T &input, T &output) const {
        auto &derived = static_cast<Op const &>(*this);
        using ST = typename T::value_type;
        if constexpr (detail::VecDispatch<ST>::supported) {
            if (detail::all_byte_buffer(input, output)) {
                using VecT = typename detail::VecDispatch<ST>::VecT;
                auto buf_in = input.container().get_byte_buffer();
                auto buf_o = output.container().get_byte_buffer();
                auto off_in = static_cast<uint>(input.container().get_byte_offset());
                auto off_o = static_cast<uint>(output.container().get_byte_offset());
                auto n = output.size();
                auto vec_n = static_cast<uint>(n / detail::VecDispatch<ST>::N);
                auto rem = static_cast<uint>(n % detail::VecDispatch<ST>::N);
                for (auto i : dynamic_range(vec_n)) {
                    auto byte_idx = i * static_cast<uint>(sizeof(VecT));
                    auto v = buf_in->read<VecT>(off_in + byte_idx);
                    auto vr = detail::vectorized_unary_scalar(
                        v, [&](auto s) { return derived.apply(s); });
                    buf_o->write(off_o + byte_idx, vr);
                }
                for (auto i : dynamic_range(rem)) {
                    auto idx = vec_n * detail::VecDispatch<ST>::N + i;
                    auto byte_idx = idx * static_cast<uint>(sizeof(ST));
                    auto v = buf_in->read<ST>(off_in + byte_idx);
                    buf_o->write(off_o + byte_idx, derived.apply(v));
                }
                return;
            }
        }
        for (auto i : dynamic_range(input.size())) {
            output[i] = derived.apply(input[i]);
        }
    }
public:
    Unary(luisa::string name) : Operator(std::move(name)) {}

    /// Element-wise unary ops can safely operate in-place: output[i] = f(input[i]).
    bool can_operate_inplace() const override { return true; }

    void forward(luisa::span<std::reference_wrapper<ITensor>> inputs, luisa::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 1 && outputs.size() == 1,
                     "Unary operator requires 1 input and 1 output.");
        auto &input = inputs[0].get();
        auto &output = outputs[0].get();
        LUISA_ASSERT(input.element_type_index() == output.element_type_index(),
                     "Input and output must have the same element type.");
        LUISA_ASSERT(input.size() == output.size(),
                     "Input and output must have the same size.");
#else
        auto &input = inputs[0].get();
        auto &output = outputs[0].get();
#endif
        visit_type_index<TypeList>(input.element_type_index(), [&]<typename T>() {
            apply_to(static_cast<NNTensor<T> &>(input), static_cast<NNTensor<T> &>(output));
        });
    }
};

#define DEFINE_UNARY_OP_EX(Name, TypeList, expr) \
    class Name : public Unary<Name, TypeList> {  \
    public:                                      \
        Name() : Unary(#Name) {}                 \
        template<typename T>                     \
        T apply(T x) const { return expr; }      \
    };                                           \
    REGISTER_TO_DEFAULT_OPSET(Name) {            \
        return luisa::make_unique<Name>();         \
    }

#define DEFINE_UNARY_OP(Name, expr) DEFINE_UNARY_OP_EX(Name, NNTypeList, expr)

DEFINE_UNARY_OP_EX(Relu, NNFilteredTypeList<IsNativeArithmetic>, max(T{0}, x));
DEFINE_UNARY_OP_EX(Sigmoid, NNFilteredTypeList<IsFloatingPoint>,
                   [&]()->T { auto z = exp(-abs(x)); return select(T{1} / (T{1} + z), z / (T{1} + z), x < T{0}); }());

// --- Math unary ops (floating point) ---
DEFINE_UNARY_OP_EX(Acos, NNFilteredTypeList<IsFloatingPoint>, acos(x));
DEFINE_UNARY_OP_EX(Asin, NNFilteredTypeList<IsFloatingPoint>, asin(x));
DEFINE_UNARY_OP_EX(Atan, NNFilteredTypeList<IsFloatingPoint>, atan(x));
DEFINE_UNARY_OP_EX(Ceil, NNFilteredTypeList<IsFloatingPoint>, ceil(x));
DEFINE_UNARY_OP_EX(Cos, NNFilteredTypeList<IsFloatingPoint>, cos(x));
DEFINE_UNARY_OP_EX(Exp, NNFilteredTypeList<IsFloatingPoint>, exp(x));
DEFINE_UNARY_OP_EX(Floor, NNFilteredTypeList<IsFloatingPoint>, floor(x));
DEFINE_UNARY_OP_EX(Log, NNFilteredTypeList<IsFloatingPoint>, log(x));
DEFINE_UNARY_OP_EX(Reciprocal, NNFilteredTypeList<IsFloatingPoint>, T{1} / x);
DEFINE_UNARY_OP_EX(Round, NNFilteredTypeList<IsFloatingPoint>, round(x));
DEFINE_UNARY_OP_EX(Sin, NNFilteredTypeList<IsFloatingPoint>, sin(x));
DEFINE_UNARY_OP_EX(Sqrt, NNFilteredTypeList<IsFloatingPoint>, sqrt(x));
DEFINE_UNARY_OP_EX(Tan, NNFilteredTypeList<IsFloatingPoint>, tan(x));
DEFINE_UNARY_OP_EX(Tanh, NNFilteredTypeList<IsFloatingPoint>, tanh(x));
DEFINE_UNARY_OP_EX(Softplus, NNFilteredTypeList<IsFloatingPoint>,
                   [&]()->T { auto ax = abs(x); return max(x, T{0}) + log(T{1} + exp(-ax)); }());
DEFINE_UNARY_OP_EX(Softsign, NNFilteredTypeList<IsFloatingPoint>, x / (T{1} + abs(x)));
DEFINE_UNARY_OP_EX(Erf, NNFilteredTypeList<IsFloatingPoint>, erf(x));

// --- Abs: signed numeric types (float, half, double, short, int, slong) ---
DEFINE_UNARY_OP_EX(Abs, NNFilteredTypeList<IsSigned>, abs(x));

// --- Neg: signed numeric types ---
DEFINE_UNARY_OP_EX(Neg, NNFilteredTypeList<IsSigned>, -x);

// --- Not: boolean input/output ---
DEFINE_UNARY_OP_EX(Not, NNFilteredTypeList<IsBool>, !x);

// --- BitwiseNot: integer input/output ---
DEFINE_UNARY_OP_EX(BitwiseNot, NNFilteredTypeList<IsInteger>, ~x);

// --- IsNaN: float in, bool out ---
class IsNaN : public Operator {
public:
    IsNaN() : Operator("IsNaN") {}

    void forward(luisa::span<std::reference_wrapper<ITensor>> inputs,
                 luisa::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 1 && outputs.size() == 1,
                     "IsNaN requires 1 input and 1 output.");
        auto &input = inputs[0].get();
        auto &output = outputs[0].get();
        LUISA_ASSERT(input.size() == output.size(),
                     "IsNaN: input and output must have the same size.");
        LUISA_ASSERT(output.element_type_index() == refl::type_index_of<bool>(),
                     "IsNaN: output must be bool type.");
#else
        auto &input = inputs[0].get();
        auto &output = outputs[0].get();
#endif
        visit_type_index<NNFilteredTypeList<IsFloatingPoint>>(input.element_type_index(), [&]<typename T>() {
            auto &in = static_cast<NNTensor<T> &>(input);
            auto &out = static_cast<NNTensor<bool> &>(output);
            if constexpr (detail::VecDispatch<T>::supported) {
                if (detail::all_byte_buffer(in, out)) {
                    using VecT = typename detail::VecDispatch<T>::VecT;
                    auto buf_in = in.container().get_byte_buffer();
                    auto buf_out = out.container().get_byte_buffer();
                    auto off_in = static_cast<uint>(in.container().get_byte_offset());
                    auto off_out = static_cast<uint>(out.container().get_byte_offset());
                    auto n = in.size();
                    auto vec_n = static_cast<uint>(n / detail::VecDispatch<T>::N);
                    auto rem = static_cast<uint>(n % detail::VecDispatch<T>::N);
                    for (auto i : dynamic_range(vec_n)) {
                        auto byte_idx = i * static_cast<uint>(sizeof(VecT));
                        auto v = buf_in->read<VecT>(off_in + byte_idx);
                        auto base_idx = i * detail::VecDispatch<T>::N;
                        buf_out->write(off_out + base_idx + 0, luisa::compute::isnan(v.x));
                        buf_out->write(off_out + base_idx + 1, luisa::compute::isnan(v.y));
                        buf_out->write(off_out + base_idx + 2, luisa::compute::isnan(v.z));
                        buf_out->write(off_out + base_idx + 3, luisa::compute::isnan(v.w));
                    }
                    for (auto i : dynamic_range(rem)) {
                        auto idx = vec_n * detail::VecDispatch<T>::N + i;
                        auto v = buf_in->read<T>(off_in + idx * static_cast<uint>(sizeof(T)));
                        buf_out->write(off_out + idx, luisa::compute::isnan(v));
                    }
                    return;
                }
            }
            for (auto i : dynamic_range(in.size())) {
                out[i] = luisa::compute::isnan(in[i]);
            }
        });
    }
};
REGISTER_TO_DEFAULT_OPSET(IsNaN) {
    return luisa::make_unique<IsNaN>();
};

// --- IsInf: float in, bool out ---
class IsInf : public Operator {
private:
    int32_t detect_negative_;
    int32_t detect_positive_;

public:
    IsInf(int32_t detect_negative = 1, int32_t detect_positive = 1)
        : Operator("IsInf"), detect_negative_(detect_negative), detect_positive_(detect_positive) {}

    void forward(luisa::span<std::reference_wrapper<ITensor>> inputs,
                 luisa::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 1 && outputs.size() == 1,
                     "IsInf requires 1 input and 1 output.");
        auto &input = inputs[0].get();
        auto &output = outputs[0].get();
        LUISA_ASSERT(input.size() == output.size(),
                     "IsInf: input and output must have the same size.");
        LUISA_ASSERT(output.element_type_index() == refl::type_index_of<bool>(),
                     "IsInf: output must be bool type.");
#else
        auto &input = inputs[0].get();
        auto &output = outputs[0].get();
#endif
        visit_type_index<NNFilteredTypeList<IsFloatingPoint>>(input.element_type_index(), [&]<typename T>() {
            auto &in = static_cast<NNTensor<T> &>(input);
            auto &out = static_cast<NNTensor<bool> &>(output);
            auto compute_is_inf = [&](auto val) {
                auto is_inf_val = luisa::compute::isinf(val);
                if (detect_negative_ && detect_positive_) {
                    return is_inf_val;
                } else if (detect_positive_) {
                    return is_inf_val & (val > decltype(val){0});
                } else if (detect_negative_) {
                    return is_inf_val & (val < decltype(val){0});
                } else {
                    return decltype(is_inf_val){false};
                }
            };
            if constexpr (detail::VecDispatch<T>::supported) {
                if (detail::all_byte_buffer(in, out)) {
                    using VecT = typename detail::VecDispatch<T>::VecT;
                    auto buf_in = in.container().get_byte_buffer();
                    auto buf_out = out.container().get_byte_buffer();
                    auto off_in = static_cast<uint>(in.container().get_byte_offset());
                    auto off_out = static_cast<uint>(out.container().get_byte_offset());
                    auto n = in.size();
                    auto vec_n = static_cast<uint>(n / detail::VecDispatch<T>::N);
                    auto rem = static_cast<uint>(n % detail::VecDispatch<T>::N);
                    for (auto i : dynamic_range(vec_n)) {
                        auto byte_idx = i * static_cast<uint>(sizeof(VecT));
                        auto v = buf_in->read<VecT>(off_in + byte_idx);
                        auto base_idx = i * detail::VecDispatch<T>::N;
                        buf_out->write(off_out + base_idx + 0, compute_is_inf(v.x));
                        buf_out->write(off_out + base_idx + 1, compute_is_inf(v.y));
                        buf_out->write(off_out + base_idx + 2, compute_is_inf(v.z));
                        buf_out->write(off_out + base_idx + 3, compute_is_inf(v.w));
                    }
                    for (auto i : dynamic_range(rem)) {
                        auto idx = vec_n * detail::VecDispatch<T>::N + i;
                        auto v = buf_in->read<T>(off_in + idx * static_cast<uint>(sizeof(T)));
                        buf_out->write(off_out + idx, compute_is_inf(v));
                    }
                    return;
                }
            }
            for (auto i : dynamic_range(in.size())) {
                out[i] = compute_is_inf(in[i]);
            }
        });
    }
};
REGISTER_TO_DEFAULT_OPSET(IsInf) {
    int32_t detect_negative = 1;
    int32_t detect_positive = 1;
    if (auto p = node.try_get_attr("detect_negative"))
        detect_negative = p->get<onnx::AttributeType::INT>();
    if (auto p = node.try_get_attr("detect_positive"))
        detect_positive = p->get<onnx::AttributeType::INT>();
    return luisa::make_unique<IsInf>(detect_negative, detect_positive);
};

}// namespace lcml::onnx
