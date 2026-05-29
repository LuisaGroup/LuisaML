#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

namespace lcml::onnx {

// ============================================================================
// Type traits for quantization support
// ============================================================================

template<typename T>
struct IsQuantized : std::false_type {};
template<>
struct IsQuantized<FP4E2M1> : std::true_type {};
template<>
struct IsQuantized<FP8E4M3FN> : std::true_type {};
template<>
struct IsQuantized<FP8E5M2> : std::true_type {};
template<>
struct IsQuantized<FP16Quantized> : std::true_type {};

template<typename T>
inline constexpr bool is_quantized_v = IsQuantized<T>::value;

template<typename T>
struct IsNativeOrQuantizedArithmetic : std::bool_constant<
    IsNativeArithmetic<T>::value ||
    std::is_same_v<T, FP4E2M1> ||
    std::is_same_v<T, FP8E4M3FN> ||
    std::is_same_v<T, FP8E5M2>> {};

// ============================================================================
// Binary operator CRTP base class with NumPy-style broadcasting
// ============================================================================

template<typename Op, typename TypeList = NNTypeList>
class Binary : public Operator {
private:
    // Broadcast two shapes following NumPy rules.
    // Returns the broadcasted output shape.
    static ITensor::shape_type broadcast_shape(ITensor::shape_type const &a,
                                               ITensor::shape_type const &b) {
        auto ndim = std::max(a.size(), b.size());
        ITensor::shape_type result(ndim);
        for (size_t i = 0; i < ndim; ++i) {
            auto da = (i < ndim - a.size()) ? 1u : a[i - (ndim - a.size())];
            auto db = (i < ndim - b.size()) ? 1u : b[i - (ndim - b.size())];
            LUISA_ASSERT(da == db || da == 1 || db == 1,
                         "Binary op: shapes are not broadcastable");
            result[i] = std::max(da, db);
        }
        return result;
    }

    template<typename TensorT>
    auto apply_to(TensorT &input_a, TensorT &input_b, TensorT &output) const
        -> std::enable_if_t<!is_quantized_v<typename TensorT::value_type>, void> {
        auto &derived = static_cast<Op const &>(*this);
        using ST = typename TensorT::value_type;

        auto const &sa = input_a.shape();
        auto const &sb = input_b.shape();

        // Fast path 1: same shape — no broadcasting needed
        if (sa == sb) {
            if constexpr (detail::VecDispatch<ST>::supported) {
                if (detail::all_byte_buffer(input_a, input_b, output)) {
                    detail::vectorized_same_shape<ST>(
                        input_a, input_b, output,
                        [&](auto a, auto b) { return derived.apply(a, b); });
                    return;
                }
            }
            for (auto i : dynamic_range(output.size())) {
                output[i] = derived.apply(input_a[i], input_b[i]);
            }
            return;
        }

        // Fast path 2: scalar broadcast — one input is a single element
        if (input_b.size() == 1) {
            auto scalar_b = input_b[0u];
            if constexpr (detail::VecDispatch<ST>::supported) {
                if (detail::all_byte_buffer(input_a, output)) {
                    detail::vectorized_scalar_b<ST>(
                        input_a, scalar_b, output,
                        [&](auto a, auto b) { return derived.apply(a, b); });
                    return;
                }
            }
            for (auto i : dynamic_range(output.size())) {
                output[i] = derived.apply(input_a[i], scalar_b);
            }
            return;
        }
        if (input_a.size() == 1) {
            auto scalar_a = input_a[0u];
            if constexpr (detail::VecDispatch<ST>::supported) {
                if (detail::all_byte_buffer(input_b, output)) {
                    detail::vectorized_scalar_a<ST>(
                        scalar_a, input_b, output,
                        [&](auto a, auto b) { return derived.apply(a, b); });
                    return;
                }
            }
            for (auto i : dynamic_range(output.size())) {
                output[i] = derived.apply(scalar_a, input_b[i]);
            }
            return;
        }

        // General broadcast path
        auto const &out_shape = output.shape();
        auto out_ndim = out_shape.size();

        std::vector<uint32_t> pa(out_ndim, 1u), pb(out_ndim, 1u);
        for (size_t i = 0; i < sa.size(); ++i)
            pa[out_ndim - sa.size() + i] = sa[i];
        for (size_t i = 0; i < sb.size(); ++i)
            pb[out_ndim - sb.size() + i] = sb[i];

        std::vector<uint32_t> stride_a(out_ndim, 0u), stride_b(out_ndim, 0u);
        {
            uint32_t sa_acc = 1, sb_acc = 1;
            for (int i = static_cast<int>(out_ndim) - 1; i >= 0; --i) {
                stride_a[i] = (pa[i] == 1) ? 0u : sa_acc;
                sa_acc *= pa[i];
                stride_b[i] = (pb[i] == 1) ? 0u : sb_acc;
                sb_acc *= pb[i];
            }
        }

        // Vectorized inner-dimension broadcast fast path
        if constexpr (detail::VecDispatch<ST>::supported) {
            bool can_vec_inner = false;
            if (out_ndim > 0 && output.strides()[out_ndim - 1] == 1 &&
                input_a.container().is_byte_buffer() &&
                input_b.container().is_byte_buffer() &&
                output.container().is_byte_buffer()) {
                auto last_dim = out_shape[out_ndim - 1];
                if (last_dim >= detail::VecDispatch<ST>::N &&
                    (stride_a[out_ndim - 1] == 0 || stride_a[out_ndim - 1] == 1) &&
                    (stride_b[out_ndim - 1] == 0 || stride_b[out_ndim - 1] == 1)) {
                    can_vec_inner = true;
                }
            }
            if (can_vec_inner) {
                detail::vectorized_broadcast_2in<ST>(
                    input_a, input_b, output, out_shape, static_cast<uint32_t>(out_ndim),
                    stride_a, stride_b,
                    [&](auto a, auto b) { return derived.apply(a, b); });
                return;
            }
        }

        if (detail::all_byte_buffer(input_a, input_b, output)) {
            auto buf_a = input_a.container().get_byte_buffer();
            auto buf_b = input_b.container().get_byte_buffer();
            auto buf_o = output.container().get_byte_buffer();
            auto off_a = static_cast<uint>(input_a.container().get_byte_offset());
            auto off_b = static_cast<uint>(input_b.container().get_byte_offset());
            auto off_o = static_cast<uint>(output.container().get_byte_offset());
            for (auto linear : dynamic_range(output.size())) {
                auto [idx_a, idx_b] = broadcast_linear_remap(
                    linear, output.strides(), static_cast<uint32_t>(out_ndim),
                    stride_a, stride_b);
                auto va = buf_a->read<ST>(off_a + idx_a * static_cast<uint>(sizeof(ST)));
                auto vb = buf_b->read<ST>(off_b + idx_b * static_cast<uint>(sizeof(ST)));
                buf_o->write(off_o + linear * static_cast<uint>(sizeof(ST)), derived.apply(va, vb));
            }
        } else {
            for (auto linear : dynamic_range(output.size())) {
                auto [idx_a, idx_b] = broadcast_linear_remap(
                    linear, output.strides(), static_cast<uint32_t>(out_ndim),
                    stride_a, stride_b);
                output[linear] = derived.apply(input_a[idx_a], input_b[idx_b]);
            }
        }
    }

    template<typename TensorT>
    auto apply_to(TensorT &input_a, TensorT &input_b, TensorT &output) const
        -> std::enable_if_t<is_quantized_v<typename TensorT::value_type>, void> {
        auto &derived = static_cast<Op const &>(*this);
        using ST = typename TensorT::value_type;
        auto const &sa = input_a.shape();
        auto const &sb = input_b.shape();

        if (sa == sb) {
            for (auto i : dynamic_range(output.size())) {
                auto va = dequantize(input_a[i]);
                auto vb = dequantize(input_b[i]);
                auto vr = derived.apply(va, vb);
                output[i].bits = quantize<ST>(vr);
            }
            return;
        }

        if (input_b.size() == 1) {
            auto scalar_b = input_b[0u];
            for (auto i : dynamic_range(output.size())) {
                auto va = dequantize(input_a[i]);
                auto vb = dequantize(scalar_b);
                auto vr = derived.apply(va, vb);
                output[i].bits = quantize<ST>(vr);
            }
            return;
        }
        if (input_a.size() == 1) {
            auto scalar_a = input_a[0u];
            for (auto i : dynamic_range(output.size())) {
                auto va = dequantize(scalar_a);
                auto vb = dequantize(input_b[i]);
                auto vr = derived.apply(va, vb);
                output[i].bits = quantize<ST>(vr);
            }
            return;
        }

        auto const &out_shape = output.shape();
        auto out_ndim = out_shape.size();

        std::vector<uint32_t> pa(out_ndim, 1u), pb(out_ndim, 1u);
        for (size_t i = 0; i < sa.size(); ++i)
            pa[out_ndim - sa.size() + i] = sa[i];
        for (size_t i = 0; i < sb.size(); ++i)
            pb[out_ndim - sb.size() + i] = sb[i];

        std::vector<uint32_t> stride_a(out_ndim, 0u), stride_b(out_ndim, 0u);
        {
            uint32_t sa_acc = 1, sb_acc = 1;
            for (int i = static_cast<int>(out_ndim) - 1; i >= 0; --i) {
                stride_a[i] = (pa[i] == 1) ? 0u : sa_acc;
                sa_acc *= pa[i];
                stride_b[i] = (pb[i] == 1) ? 0u : sb_acc;
                sb_acc *= pb[i];
            }
        }

        for (auto linear : dynamic_range(output.size())) {
            auto [idx_a, idx_b] = broadcast_linear_remap(
                linear, output.strides(), static_cast<uint32_t>(out_ndim),
                stride_a, stride_b);
            auto va = dequantize(input_a[idx_a]);
            auto vb = dequantize(input_b[idx_b]);
            auto vr = derived.apply(va, vb);
            output[linear].bits = quantize<ST>(vr);
        }
    }

    // Quantization helpers
    template<typename Q>
    static auto dequantize(Var<Q> &val) {
        if constexpr (std::is_same_v<Q, FP4E2M1>) {
            static auto c = fp4e2m1_to_float();
            return c(val.bits);
        } else if constexpr (std::is_same_v<Q, FP8E4M3FN>) {
            static auto c = fp8e4m3_to_float();
            return c(val.bits);
        } else if constexpr (std::is_same_v<Q, FP8E5M2>) {
            static auto c = fp8e5m2_to_float();
            return c(val.bits);
        } else {
            return val;
        }
    }

    template<typename Q, typename V>
    static auto quantize(V &&val) {
        if constexpr (std::is_same_v<Q, FP4E2M1>) {
            static auto c = fp4e2m1_from_float();
            return c(std::forward<V>(val));
        } else if constexpr (std::is_same_v<Q, FP8E4M3FN>) {
            static auto c = fp8e4m3_from_float();
            return c(std::forward<V>(val));
        } else if constexpr (std::is_same_v<Q, FP8E5M2>) {
            static auto c = fp8e5m2_from_float();
            return c(std::forward<V>(val));
        } else {
            return std::forward<V>(val);
        }
    }

public:
    Binary(std::string name) : Operator(std::move(name)) {}

    /// Element-wise binary ops can safely operate in-place on input[0]
    /// when input[0] and output have the same number of elements (no broadcast on input[0]).
    bool can_operate_inplace() const override { return true; }

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
        LUISA_ASSERT(inputs.size() == 2 && outputs.size() == 1,
                     "Binary operator requires 2 inputs and 1 output.");
        auto &a = inputs[0].get();
        auto &b = inputs[1].get();
        auto &out = outputs[0].get();

        LUISA_ASSERT(a.element_type() == b.element_type() &&
                         a.element_type() == out.element_type(),
                     "Binary op: all tensors must have the same element type.");

        // Verify output shape matches broadcast result
        auto expected = broadcast_shape(a.shape(), b.shape());
        LUISA_ASSERT(out.shape() == expected,
                     "Binary op: output shape does not match broadcast result.");

        visit_typeid<TypeList>(a.element_type(), [&]<typename T>() {
            apply_to(static_cast<NNTensor<T> &>(a),
                     static_cast<NNTensor<T> &>(b),
                     static_cast<NNTensor<T> &>(out));
        });
    }
};

// ============================================================================
// Macros for quick binary operator definitions
// ============================================================================

#define DEFINE_BINARY_OP_EX(Name, TypeList, expr) \
    class Name : public Binary<Name, TypeList> {  \
    public:                                       \
        Name() : Binary(#Name) {}                 \
        template<typename T>                      \
        T apply(T a, T b) const { return expr; }  \
    };                                            \
    REGISTER_TO_DEFAULT_OPSET(Name) {             \
        return std::make_unique<Name>();          \
    }

#define DEFINE_BINARY_OP(Name, expr) DEFINE_BINARY_OP_EX(Name, NNTypeList, expr)

// ============================================================================
// Arithmetic binary operators (ONNX standard)
// ============================================================================

// Add: element-wise addition. Supports all numeric types.
DEFINE_BINARY_OP_EX(Add, NNFilteredTypeList<IsNativeOrQuantizedArithmetic>, a + b);

// Sub: element-wise subtraction. Supports all numeric types.
DEFINE_BINARY_OP_EX(Sub, NNFilteredTypeList<IsNativeOrQuantizedArithmetic>, a - b);

// Mul: element-wise multiplication. Supports all numeric types.
DEFINE_BINARY_OP_EX(Mul, NNFilteredTypeList<IsNativeOrQuantizedArithmetic>, a *b);

// Div: element-wise division. Supports all numeric types.
DEFINE_BINARY_OP_EX(Div, NNFilteredTypeList<IsNativeOrQuantizedArithmetic>, a / b);
// Pow: element-wise power. Floating point types.
DEFINE_BINARY_OP_EX(Pow, NNFilteredTypeList<IsFloatingPoint>, pow(a, b));

// Mod: element-wise modulo. Integer types only (fmod = 0 default per ONNX spec).
// ONNX Mod op: for integer types, result = a - (a / b) * b (truncated division)
DEFINE_BINARY_OP_EX(Mod, NNFilteredTypeList<IsInteger>, a % b);

// ============================================================================
// Comparison binary operators (input: numeric, output: bool)
// ============================================================================

template<typename Op, typename TypeList = NNFilteredTypeList<IsNativeArithmetic>>
class Comparison : public Operator {
private:
    static ITensor::shape_type broadcast_shape(ITensor::shape_type const &a,
                                               ITensor::shape_type const &b) {
        auto ndim = std::max(a.size(), b.size());
        ITensor::shape_type result(ndim);
        for (size_t i = 0; i < ndim; ++i) {
            auto da = (i < ndim - a.size()) ? 1u : a[i - (ndim - a.size())];
            auto db = (i < ndim - b.size()) ? 1u : b[i - (ndim - b.size())];
            LUISA_ASSERT(da == db || da == 1 || db == 1,
                         "Comparison op: shapes are not broadcastable");
            result[i] = std::max(da, db);
        }
        return result;
    }

    template<typename InTensor>
    void apply_to(InTensor &input_a, InTensor &input_b, NNTensor<bool> &output) const {
        auto &derived = static_cast<Op const &>(*this);
        using ST = typename InTensor::value_type;
        auto const &sa = input_a.shape();
        auto const &sb = input_b.shape();

        // Fast path 1: same shape — no broadcasting needed
        if (sa == sb) {
            if constexpr (detail::VecDispatch<ST>::supported) {
                if (detail::all_byte_buffer(input_a, input_b, output)) {
                    detail::vectorized_same_shape<ST>(
                        input_a, input_b, output,
                        [&](auto a, auto b) { return derived.apply(a, b); });
                    return;
                }
            }
            for (auto i : dynamic_range(output.size())) {
                output[i] = derived.apply(input_a[i], input_b[i]);
            }
            return;
        }

        // Fast path 2: scalar broadcast
        if (input_b.size() == 1) {
            auto scalar_b = input_b[0u];
            if constexpr (detail::VecDispatch<ST>::supported) {
                if (detail::all_byte_buffer(input_a, output)) {
                    detail::vectorized_scalar_b<ST>(
                        input_a, scalar_b, output,
                        [&](auto a, auto b) { return derived.apply(a, b); });
                    return;
                }
            }
            for (auto i : dynamic_range(output.size())) {
                output[i] = derived.apply(input_a[i], scalar_b);
            }
            return;
        }
        if (input_a.size() == 1) {
            auto scalar_a = input_a[0u];
            if constexpr (detail::VecDispatch<ST>::supported) {
                if (detail::all_byte_buffer(input_b, output)) {
                    detail::vectorized_scalar_a<ST>(
                        scalar_a, input_b, output,
                        [&](auto a, auto b) { return derived.apply(a, b); });
                    return;
                }
            }
            for (auto i : dynamic_range(output.size())) {
                output[i] = derived.apply(scalar_a, input_b[i]);
            }
            return;
        }

        // General broadcast path
        auto const &out_shape = output.shape();
        auto out_ndim = out_shape.size();

        std::vector<uint32_t> pa(out_ndim, 1u), pb(out_ndim, 1u);
        for (size_t i = 0; i < sa.size(); ++i) pa[out_ndim - sa.size() + i] = sa[i];
        for (size_t i = 0; i < sb.size(); ++i) pb[out_ndim - sb.size() + i] = sb[i];

        std::vector<uint32_t> stride_a(out_ndim, 0u), stride_b(out_ndim, 0u);
        {
            uint32_t sa_acc = 1, sb_acc = 1;
            for (int i = static_cast<int>(out_ndim) - 1; i >= 0; --i) {
                stride_a[i] = (pa[i] == 1) ? 0u : sa_acc;
                sa_acc *= pa[i];
                stride_b[i] = (pb[i] == 1) ? 0u : sb_acc;
                sb_acc *= pb[i];
            }
        }

        for (auto linear : dynamic_range(output.size())) {
            auto [idx_a, idx_b] = broadcast_linear_remap(
                linear, output.strides(), static_cast<uint32_t>(out_ndim),
                stride_a, stride_b);
            output[linear] = derived.apply(input_a[idx_a], input_b[idx_b]);
        }
    }

public:
    Comparison(std::string name) : Operator(std::move(name)) {}

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
        LUISA_ASSERT(inputs.size() == 2 && outputs.size() == 1,
                     "Comparison operator requires 2 inputs and 1 output.");
        auto &a = inputs[0].get();
        auto &b = inputs[1].get();
        auto &out = outputs[0].get();

        LUISA_ASSERT(a.element_type() == b.element_type(),
                     "Comparison op: inputs must have the same element type.");
        LUISA_ASSERT(out.element_type() == typeid(bool),
                     "Comparison op: output must be bool type.");

        auto expected = broadcast_shape(a.shape(), b.shape());
        LUISA_ASSERT(out.shape() == expected,
                     "Comparison op: output shape mismatch.");

        visit_typeid<TypeList>(a.element_type(), [&]<typename T>() {
            apply_to(static_cast<NNTensor<T> &>(a),
                     static_cast<NNTensor<T> &>(b),
                     static_cast<NNTensor<bool> &>(out));
        });
    }
};

#define DEFINE_COMPARISON_OP_EX(Name, TypeList, expr) \
    class Name : public Comparison<Name, TypeList> {  \
    public:                                           \
        Name() : Comparison(#Name) {}                 \
        template<typename T>                          \
        auto apply(T a, T b) const { return expr; }   \
    };                                                \
    REGISTER_TO_DEFAULT_OPSET(Name) {                 \
        return std::make_unique<Name>();              \
    }

#define DEFINE_COMPARISON_OP(Name, expr) DEFINE_COMPARISON_OP_EX(Name, NNFilteredTypeList<IsNativeArithmetic>, expr)

DEFINE_COMPARISON_OP(Equal, a == b);
DEFINE_COMPARISON_OP(Greater, a > b);
DEFINE_COMPARISON_OP(Less, a < b);
DEFINE_COMPARISON_OP(GreaterOrEqual, a >= b);
DEFINE_COMPARISON_OP(LessOrEqual, a <= b);

// ============================================================================
// Logical binary operators (bool input/output)
// ============================================================================

DEFINE_BINARY_OP_EX(And, NNFilteredTypeList<IsBool>, a &b);
DEFINE_BINARY_OP_EX(Or, NNFilteredTypeList<IsBool>, a | b);
DEFINE_BINARY_OP_EX(Xor, NNFilteredTypeList<IsBool>, a ^ b);

// ============================================================================
// Bitwise binary operators (integer types)
// ============================================================================

DEFINE_BINARY_OP_EX(BitwiseAnd, NNFilteredTypeList<IsInteger>, a &b);
DEFINE_BINARY_OP_EX(BitwiseOr, NNFilteredTypeList<IsInteger>, a | b);
DEFINE_BINARY_OP_EX(BitwiseXor, NNFilteredTypeList<IsInteger>, a ^ b);
DEFINE_BINARY_OP_EX(ShiftLeft, NNFilteredTypeList<IsInteger>, a << b);
DEFINE_BINARY_OP_EX(ShiftRight, NNFilteredTypeList<IsInteger>, a >> b);

// ============================================================================
// Variadic aggregation operators (Min, Max, Mean, Sum)
// These accept >=1 inputs and perform pairwise accumulation.
// ============================================================================

template<typename Op, typename TypeList = NNFilteredTypeList<IsNumeric>>
class VariadicBinary : public Operator {
private:
    static ITensor::shape_type broadcast_shape(ITensor::shape_type const &a,
                                               ITensor::shape_type const &b) {
        auto ndim = std::max(a.size(), b.size());
        ITensor::shape_type result(ndim);
        for (size_t i = 0; i < ndim; ++i) {
            auto da = (i < ndim - a.size()) ? 1u : a[i - (ndim - a.size())];
            auto db = (i < ndim - b.size()) ? 1u : b[i - (ndim - b.size())];
            LUISA_ASSERT(da == db || da == 1 || db == 1,
                         "Variadic op: shapes are not broadcastable");
            result[i] = std::max(da, db);
        }
        return result;
    }

    template<typename TensorT>
    void accumulate(TensorT &acc, TensorT &input) const {
        auto &derived = static_cast<Op const &>(*this);
        using ST = typename TensorT::value_type;
        auto const &out_shape = acc.shape();
        auto const &sb = input.shape();

        // Fast path 1: same shape — no broadcasting needed
        if (out_shape == sb) {
            if constexpr (detail::VecDispatch<ST>::supported) {
                if (detail::all_byte_buffer(acc, input)) {
                    detail::vectorized_same_shape<ST>(
                        acc, input, acc,
                        [&](auto a, auto b) { return derived.apply(a, b); });
                    return;
                }
            }
            for (auto i : dynamic_range(acc.size())) {
                acc[i] = derived.apply(acc[i], input[i]);
            }
            return;
        }

        // Fast path 2: scalar input — broadcast single element
        if (input.size() == 1) {
            auto scalar = input[0u];
            if constexpr (detail::VecDispatch<ST>::supported) {
                if (acc.container().is_byte_buffer()) {
                    detail::vectorized_scalar_b<ST>(
                        acc, scalar, acc,
                        [&](auto a, auto b) { return derived.apply(a, b); });
                    return;
                }
            }
            for (auto i : dynamic_range(acc.size())) {
                acc[i] = derived.apply(acc[i], scalar);
            }
            return;
        }

        // General broadcast path
        auto out_ndim = out_shape.size();

        std::vector<uint32_t> pb(out_ndim, 1u);
        for (size_t i = 0; i < sb.size(); ++i) pb[out_ndim - sb.size() + i] = sb[i];

        std::vector<uint32_t> stride_b(out_ndim, 0u);
        {
            uint32_t sb_acc = 1;
            for (int i = static_cast<int>(out_ndim) - 1; i >= 0; --i) {
                stride_b[i] = (pb[i] == 1) ? 0u : sb_acc;
                sb_acc *= pb[i];
            }
        }

        // Vectorized inner-dimension broadcast fast path
        if constexpr (detail::VecDispatch<ST>::supported) {
            bool can_vec_inner = false;
            if (out_ndim > 0 && acc.strides()[out_ndim - 1] == 1 &&
                acc.container().is_byte_buffer() &&
                input.container().is_byte_buffer()) {
                auto last_dim = out_shape[out_ndim - 1];
                if (last_dim >= detail::VecDispatch<ST>::N &&
                    (stride_b[out_ndim - 1] == 0 || stride_b[out_ndim - 1] == 1)) {
                    can_vec_inner = true;
                }
            }
            if (can_vec_inner) {
                detail::vectorized_broadcast_1in_acc<ST>(
                    acc, input, out_shape, static_cast<uint32_t>(out_ndim), stride_b,
                    [&](auto a, auto b) { return derived.apply(a, b); });
                return;
            }
        }

        if (acc.container().is_byte_buffer() && input.container().is_byte_buffer()) {
            auto buf_acc = acc.container().get_byte_buffer();
            auto buf_b = input.container().get_byte_buffer();
            auto off_acc = static_cast<uint>(acc.container().get_byte_offset());
            auto off_b = static_cast<uint>(input.container().get_byte_offset());
            for (auto linear : dynamic_range(acc.size())) {
                auto [idx_b] = broadcast_linear_remap(
                    linear, acc.strides(), static_cast<uint32_t>(out_ndim),
                    stride_b);
                auto vacc = buf_acc->read<ST>(off_acc + linear * static_cast<uint>(sizeof(ST)));
                auto vb = buf_b->read<ST>(off_b + idx_b * static_cast<uint>(sizeof(ST)));
                buf_acc->write(off_acc + linear * static_cast<uint>(sizeof(ST)), derived.apply(vacc, vb));
            }
        } else {
            for (auto linear : dynamic_range(acc.size())) {
                auto [idx_b] = broadcast_linear_remap(
                    linear, acc.strides(), static_cast<uint32_t>(out_ndim),
                    stride_b);
                acc[linear] = derived.apply(acc[linear], input[idx_b]);
            }
        }
    }

public:
    VariadicBinary(std::string name) : Operator(std::move(name)) {}

    /// Variadic ops copy input[0] to output then accumulate; safe for in-place.
    bool can_operate_inplace() const override { return true; }

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
        LUISA_ASSERT(!inputs.empty() && outputs.size() == 1,
                     "Variadic op requires >=1 inputs and 1 output.");
        auto &out = outputs[0].get();

        // Check all inputs and output have same element type
        for (size_t i = 0; i < inputs.size(); ++i) {
            LUISA_ASSERT(inputs[i].get().element_type() == out.element_type(),
                         "Variadic op: all inputs and output must have the same element type.");
        }

        // Copy first input to output
        visit_typeid<TypeList>(inputs[0].get().element_type(), [&]<typename T>() {
            auto &first = static_cast<NNTensor<T> &>(inputs[0].get());
            auto &dst = static_cast<NNTensor<T> &>(out);
            using ST = typename NNTensor<T>::value_type;
            // Skip initial copy if already sharing storage (inplace)
            if (!first.container().shares_storage_with(dst.container())) {
                if constexpr (detail::VecDispatch<ST>::supported) {
                    if (detail::all_byte_buffer(first, dst)) {
                        detail::vectorized_unary<ST>(
                            first, dst,
                            [&](auto v) { return v; });
                    } else {
                        for (auto i : dynamic_range(first.size())) {
                            dst[i] = first[i];
                        }
                    }
                } else {
                    for (auto i : dynamic_range(first.size())) {
                        dst[i] = first[i];
                    }
                }
            }

            // Accumulate remaining inputs
            for (size_t n = 1; n < inputs.size(); ++n) {
                accumulate(dst, static_cast<NNTensor<T> &>(inputs[n].get()));
            }
        });
    }
};

#define DEFINE_VARIADIC_OP_EX(Name, TypeList, expr)      \
    class Name : public VariadicBinary<Name, TypeList> { \
    public:                                              \
        Name() : VariadicBinary(#Name) {}                \
        template<typename T>                             \
        T apply(T a, T b) const { return expr; }         \
    };                                                   \
    REGISTER_TO_DEFAULT_OPSET(Name) {                    \
        return std::make_unique<Name>();                 \
    }

DEFINE_VARIADIC_OP_EX(Min, NNFilteredTypeList<IsNativeArithmetic>, min(a, b));
DEFINE_VARIADIC_OP_EX(Max, NNFilteredTypeList<IsNativeArithmetic>, max(a, b));
DEFINE_VARIADIC_OP_EX(Sum, NNFilteredTypeList<IsNativeArithmetic>, a + b);

// Mean: sum all inputs then divide by count
class Mean : public Operator {
public:
    Mean() : Operator("Mean") {}

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
        LUISA_ASSERT(!inputs.empty() && outputs.size() == 1,
                     "Mean requires >=1 inputs and 1 output.");
        auto &out = outputs[0].get();
        auto count = inputs.size();

        visit_typeid<NNFilteredTypeList<IsFloatingPoint>>(inputs[0].get().element_type(), [&]<typename T>() {
            using VT = nn_storage_type_t<T>;
            auto &first = static_cast<NNTensor<T> &>(inputs[0].get());
            auto &dst = static_cast<NNTensor<T> &>(out);

            // Copy first input (skip if already sharing storage)
            if (!first.container().shares_storage_with(dst.container())) {
                if constexpr (detail::VecDispatch<VT>::supported) {
                    if (detail::all_byte_buffer(first, dst)) {
                        detail::vectorized_unary<VT>(
                            first, dst,
                            [&](auto v) { return v; });
                    } else {
                        for (auto i : dynamic_range(first.size())) {
                            dst[i] = first[i];
                        }
                    }
                } else {
                    for (auto i : dynamic_range(first.size())) {
                        dst[i] = first[i];
                    }
                }
            }
            // Accumulate remaining inputs
            for (size_t n = 1; n < inputs.size(); ++n) {
                auto &src = static_cast<NNTensor<T> &>(inputs[n].get());
                if constexpr (detail::VecDispatch<VT>::supported) {
                    if (detail::all_byte_buffer(dst, src)) {
                        detail::vectorized_same_shape<VT>(
                            dst, src, dst,
                            [&](auto a, auto b) { return a + b; });
                        continue;
                    }
                }
                for (auto i : dynamic_range(dst.size())) {
                    dst[i] = dst[i] + src[i];
                }
            }
            // Divide by count
            auto divisor = Var<VT>{VT(count)};
            if constexpr (detail::VecDispatch<VT>::supported) {
                if (dst.container().is_byte_buffer()) {
                    detail::vectorized_in_place<VT>(
                        dst, [&](auto v) { return v / divisor; });
                    return;
                }
            }
            for (auto i : dynamic_range(dst.size())) {
                dst[i] = dst[i] / divisor;
            }
        });
    }
};
REGISTER_TO_DEFAULT_OPSET(Mean) {
    return std::make_unique<Mean>();
};

}// namespace lcml::onnx
