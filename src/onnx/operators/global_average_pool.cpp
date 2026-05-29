#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"
#include "onnx/fp_quantized.h"

namespace lcml::onnx {

// GlobalAveragePool: computes the average of all spatial elements for each channel.
// ONNX spec: input (N, C, D1, ..., Dn) -> output (N, C, 1, ..., 1)
class GlobalAveragePool : public Operator {
public:
    GlobalAveragePool() : Operator("GlobalAveragePool") {}

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 1 && outputs.size() == 1,
                     "GlobalAveragePool requires 1 input and 1 output.");
        auto &X = inputs[0].get();
        auto &Y = outputs[0].get();
#else
        auto &X = inputs[0].get();
        auto &Y = outputs[0].get();
#endif

        auto const &x_shape = X.shape();
        uint32_t N = x_shape[0];
        uint32_t C = x_shape[1];

        // Compute total spatial size
        uint32_t spatial_size = 1;
        for (size_t i = 2; i < x_shape.size(); ++i)
            spatial_size *= x_shape[i];

        visit_typeid<NNTypeList>(X.element_type(), [&]<typename T>() {
            using VT = nn_storage_type_t<T>;
            auto &x = static_cast<NNTensor<T> &>(X);
            auto &y = static_cast<NNTensor<T> &>(Y);

            if constexpr (IsFloatingPoint<T>::value) {
                auto inv_spatial = def(VT(1.0) / static_cast<VT>(spatial_size));
                auto one = Var<VT>{VT{1}};

                if constexpr (std::is_same_v<VT, float> || std::is_same_v<VT, half>) {
                    using VecT = typename detail::VecDispatch<VT>::VecT;
                    constexpr uint32_t VecN = detail::VecDispatch<VT>::N;
                    auto one_vec = detail::VecDispatch<VT>::broadcast(one);

                    bool use_buf_vec = x.container().is_byte_buffer();
                    Var<ByteBuffer> *buf_x = nullptr;
                    uint off_x = 0;
                    uint elem_size = 0;
                    if (use_buf_vec) {
                        buf_x = x.container().get_byte_buffer();
                        off_x = static_cast<uint>(x.container().get_byte_offset());
                        elem_size = static_cast<uint>(sizeof(VT));
                    }

                    for (auto n : dynamic_range(N)) {
                        for (auto c : dynamic_range(C)) {
                            auto base = n * x.strides()[0] + c * x.strides()[1];
                            auto vec_count = spatial_size / VecN;
                            auto rem = spatial_size % VecN;

                            auto sum = def(VT{0});
                            Var<VecT> vec_sum = detail::VecDispatch<VT>::broadcast(def(VT{0}));

                            for (auto i : dynamic_range(vec_count)) {
                                Var<VecT> v;
                                if (use_buf_vec) {
                                    auto byte_off = off_x + (base + i * VecN) * elem_size;
                                    v = buf_x->read<VecT>(byte_off);
                                } else {
                                    auto off = base + i * VecN;
                                    if constexpr (std::is_same_v<VT, float>) {
                                        v = make_float4(x[off + 0u], x[off + 1u], x[off + 2u], x[off + 3u]);
                                    } else {
                                        v = make_half4(x[off + 0u], x[off + 1u], x[off + 2u], x[off + 3u]);
                                    }
                                }
                                vec_sum = luisa::compute::fma(v, one_vec, vec_sum);
                            }

                            sum += vec_sum.x + vec_sum.y + vec_sum.z + vec_sum.w;

                            auto start = vec_count * VecN;
                            for (auto i : dynamic_range(rem)) {
                                Var<VT> v;
                                if (use_buf_vec) {
                                    auto byte_off = off_x + (base + start + i) * elem_size;
                                    v = buf_x->read<VT>(byte_off);
                                } else {
                                    v = x[base + start + i];
                                }
                                sum = luisa::compute::fma(v, one, sum);
                            }

                            y[n * y.strides()[0] + c * y.strides()[1]] = sum * inv_spatial;
                        }
                    }
                } else {
                    for (auto n : dynamic_range(N)) {
                        for (auto c : dynamic_range(C)) {
                            auto sum = def(VT{0});
                            auto base = n * x.strides()[0] + c * x.strides()[1];
                            for (auto s : dynamic_range(spatial_size)) {
                                sum = luisa::compute::fma(x[base + s], one, sum);
                            }
                            y[n * y.strides()[0] + c * y.strides()[1]] = sum * inv_spatial;
                        }
                    }
                }
            } else if constexpr (std::is_same_v<T, FP8E4M3FN> || std::is_same_v<T, FP8E5M2> || std::is_same_v<T, FP4E2M1>) {
                auto deq = [&]() {
                    if constexpr (std::is_same_v<T, FP8E4M3FN>) return fp8e4m3_to_float();
                    else if constexpr (std::is_same_v<T, FP8E5M2>) return fp8e5m2_to_float();
                    else return fp4e2m1_to_float();
                }();
                auto q = [&]() {
                    if constexpr (std::is_same_v<T, FP8E4M3FN>) return fp8e4m3_from_float();
                    else if constexpr (std::is_same_v<T, FP8E5M2>) return fp8e5m2_from_float();
                    else return fp4e2m1_from_float();
                }();
                auto inv_spatial_f = 1.0f / static_cast<float>(spatial_size);
                auto one_f = def(1.0f);

                for (auto n : dynamic_range(N)) {
                    for (auto c : dynamic_range(C)) {
                        auto sum = def(0.0f);
                        auto base = n * x.strides()[0] + c * x.strides()[1];
                        for (auto s : dynamic_range(spatial_size)) {
                            auto bits = x[base + s].bits.cast<ushort>();
                            auto val = deq(bits);
                            sum = luisa::compute::fma(val.cast<float>(), one_f, sum);
                        }
                        auto result = sum * inv_spatial_f;
                        auto out_bits = q(cast<half>(result));
                        y[n * y.strides()[0] + c * y.strides()[1]].bits = out_bits.cast<uint16_t>();
                    }
                }
            } else {
                LUISA_ASSERT(false, "GlobalAveragePool: unsupported element type");
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(GlobalAveragePool) {
    return std::make_unique<GlobalAveragePool>();
};

}// namespace lcml::onnx
