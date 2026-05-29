#include "onnx/operator.h"
#include "onnx/network_instance.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"
#include <luisa/dsl/coop_vector.h>

namespace lcml::onnx {

template<typename T>
struct GemmTensorRequirements {
    static constexpr bool value = !std::is_same_v<T, bool> && !std::is_same_v<T, ushort> && !std::is_same_v<T, short> &&
                                  !std::is_same_v<T, FP4E2M1> && !std::is_same_v<T, FP8E4M3FN> && !std::is_same_v<T, FP8E5M2> &&
                                  !std::is_same_v<T, FP16Quantized>;
};

class Gemm : public Operator {
private:
    float alpha_;
    float beta_;
    int transA_;
    int transB_;
    uint warp_size_ = 0;
    bool use_coop_vec_ = false;

    template<typename T>
    void apply(T &A, T &B, ITensor *C_ptr, T &Y) const {
        using value_type = typename T::value_type;
        // Determine dimensions after transpose
        auto const &shapeA = A.shape();
        auto const &shapeB = B.shape();

        // A is 2D: (M, K) or (K, M) if transA
        uint M = transA_ ? shapeA[1] : shapeA[0];
        uint K = transA_ ? shapeA[0] : shapeA[1];

        // B is 2D: (K, N) or (N, K) if transB
        uint K2 = transB_ ? shapeB[1] : shapeB[0];
        uint N = transB_ ? shapeB[0] : shapeB[1];

#ifndef NDEBUG
        LUISA_ASSERT(K == K2, "Gemm: inner dimensions mismatch (K={} vs {})", K, K2);

        auto const &shapeY = Y.shape();
        LUISA_ASSERT(shapeY.size() == 2 && shapeY[0] == M && shapeY[1] == N,
                     "Gemm: Y shape mismatch, expected ({}, {})", M, N);
#endif

        // Compile-time alpha/beta checks to eliminate dead code
        bool const has_alpha = (alpha_ != 1.0f);
        bool const has_C = (C_ptr != nullptr && beta_ != 0.0f);
        bool const beta_is_one = (beta_ == 1.0f);

        // Helper lambdas for element access considering transpose
        auto get_A = [&](auto r, auto c) -> decltype(auto) {
            if (transA_)
                return A(c, r);
            else
                return A(r, c);
        };
        auto get_B = [&](auto r, auto c) -> decltype(auto) {
            if (transB_)
                return B(c, r);
            else
                return B(r, c);
        };

        // Resolve C info once
        T *C_typed = nullptr;
        uint c_ndim = 0, c_dim0 = 0, c_dim1 = 0;
        if (has_C) {
            C_typed = &static_cast<T &>(*C_ptr);
            auto const &shapeC = C_typed->shape();
            c_ndim = shapeC.size();
            if (c_ndim >= 1) c_dim0 = shapeC[0];
            if (c_ndim >= 2) c_dim1 = shapeC[1];
        }

        // Add bias: beta * C with broadcasting, optimized for beta==1
        auto add_bias = [&](auto &result, auto m, auto n) {
            if (!has_C) return;
            auto &C = *C_typed;
            auto get_c_val = [&]() -> decltype(auto) {
                if (c_ndim == 0 || (c_ndim == 1 && c_dim0 == 1)) {
                    return C[0u];
                } else if (c_ndim == 1) {
                    if (c_dim0 == N)
                        return C[n];
                    else
                        return C[m];
                } else {
                    auto ci = (c_dim0 == 1) ? 0u : m;
                    auto cj = (c_dim1 == 1) ? 0u : n;
                    return C(ci, cj);
                }
            };
            if (beta_is_one) {
                result += get_c_val();
            } else {
                result += value_type(beta_) * get_c_val();
            }
        };

        // Apply alpha scaling (skip if alpha==1.0)
        auto apply_alpha = [&](auto &sum) {
            if (has_alpha) {
                sum *= value_type(alpha_);
            }
        };

        // Accumulate: fma for floating-point, plain multiply-add for integer
        auto accumulate = [&](auto &sum, auto a, auto b) {
            if constexpr (luisa::is_floating_point_v<value_type>) {
                sum = luisa::compute::fma(a, b, sum);
            } else {
                sum += a * b;
            }
        };

        using VT = value_type;

        auto compute_dot = [&](auto m, auto n) {
            auto sum = def(VT{0});
            if constexpr (detail::VecDispatch<VT>::supported) {
                bool can_vec = A.container().is_byte_buffer() && B.container().is_byte_buffer() &&
                               ((transA_ == 0 && A.strides()[1] == 1) || (transA_ == 1 && A.strides()[0] == 1)) &&
                               ((transB_ == 0 && B.strides()[0] == 1) || (transB_ == 1 && B.strides()[1] == 1));
                if (can_vec) {
                    using VecT = typename detail::VecDispatch<VT>::VecT;
                    auto buf_a = A.container().get_byte_buffer();
                    auto buf_b = B.container().get_byte_buffer();
                    auto off_a = static_cast<uint>(A.container().get_byte_offset());
                    auto off_b = static_cast<uint>(B.container().get_byte_offset());
                    auto vec_n = K / detail::VecDispatch<VT>::N;
                    auto rem = K % detail::VecDispatch<VT>::N;
                    auto type_size = static_cast<uint>(sizeof(VT));
                    auto a_base_stride = transA_ ? A.strides()[1] : A.strides()[0];
                    auto b_base_stride = transB_ ? B.strides()[0] : B.strides()[1];
                    auto base_a = off_a + (cast<uint>(m) * a_base_stride) * type_size;
                    auto base_b = off_b + (cast<uint>(n) * b_base_stride) * type_size;
                    auto acc4 = def(VecT(VT{0}));
                    for (auto kblk : dynamic_range(vec_n)) {
                        auto byte_k = kblk * detail::VecDispatch<VT>::N * type_size;
                        auto va = buf_a->read<VecT>(base_a + byte_k);
                        auto vb = buf_b->read<VecT>(base_b + byte_k);
                        accumulate(acc4, va, vb);
                    }
                    sum += acc4.x + acc4.y + acc4.z + acc4.w;
                    for (auto k : dynamic_range(rem)) {
                        auto kk = vec_n * detail::VecDispatch<VT>::N + k;
                        accumulate(sum, get_A(m, kk), get_B(kk, n));
                    }
                } else {
                    for (auto k : dynamic_range(K)) {
                        accumulate(sum, get_A(m, k), get_B(k, n));
                    }
                }
            } else {
                for (auto k : dynamic_range(K)) {
                    accumulate(sum, get_A(m, k), get_B(k, n));
                }
            }
            return sum;
        };

        // Compute Y = alpha * A' * B' + beta * C

        // CooperativeVector hardware-accelerated path:
        // Replaces the entire K*N loop with a single cooperative_mul_add instruction.
        // Requirements: float type, M==1, B is in ByteBuffer, transB==1 (column-major compatible),
        //               alpha==1.0 (no scaling needed), beta==1.0 or no C.
        if constexpr (luisa::is_floating_point_v<value_type> && std::is_same_v<value_type, float>) {
            if (use_coop_vec_ && M == 1 && transB_ == 1 &&
                B.container().is_byte_buffer() && !has_alpha) {

                // Check if bias C is also in ByteBuffer and compatible for cooperative_mul_add
                bool use_coop_mul_add = false;
                if (has_C && beta_is_one && C_typed != nullptr &&
                    c_ndim == 1 && c_dim0 == N &&
                    C_typed->container().is_byte_buffer() &&
                    C_typed->container().get_byte_buffer() == B.container().get_byte_buffer()) {
                    use_coop_mul_add = true;
                }

                // Build input CoopVector from A
                CoopVector<float> input_vec(K);
                if (A.container().is_byte_buffer() &&
                    ((transA_ == 0 && A.strides()[1] == 1) || (transA_ == 1 && A.strides()[0] == 1))) {
                    auto buf_a = A.container().get_byte_buffer();
                    auto off_a = static_cast<uint>(A.container().get_byte_offset());
                    auto vec_n = K / 4u;
                    auto rem = K % 4u;
                    auto type_size = static_cast<uint>(sizeof(float));
                    auto a_base_stride = transA_ ? A.strides()[1] : A.strides()[0];
                    auto base_a = off_a + (0u * a_base_stride) * type_size;
                    for (auto kblk : dynamic_range(vec_n)) {
                        auto av = buf_a->read<float4>(base_a + kblk * 4u * type_size);
                        input_vec[kblk * 4u + 0u] = av.x;
                        input_vec[kblk * 4u + 1u] = av.y;
                        input_vec[kblk * 4u + 2u] = av.z;
                        input_vec[kblk * 4u + 3u] = av.w;
                    }
                    for (auto k : dynamic_range(rem)) {
                        auto kk = vec_n * 4u + k;
                        input_vec[kk] = buf_a->read<float>(base_a + kk * type_size);
                    }
                } else {
                    for (auto k : dynamic_range(K)) {
                        input_vec[k] = get_A(0u, k);
                    }
                }

                // Build matrix ref pointing to B's ByteBuffer location
                // transB==1: B stored as (N,K) row-major ≡ column-major (K,N) for CoopMatrix
                CoopMatrixRef mat_ref(CoopRefVecType::FLOAT32, K, N);
                mat_ref.set_byte_offset(static_cast<uint>(B.container().get_byte_offset()));

                auto &weight_buffer = *B.container().get_byte_buffer();

                if (use_coop_mul_add) {
                    // Bias is in the same ByteBuffer — use cooperative_mul_add
                    CoopVectorRef bias_ref(CoopRefVecType::FLOAT32, N);
                    bias_ref.set_byte_offset(static_cast<uint>(C_typed->container().get_byte_offset()));

                    auto result = cooperative_mat_mul_add<float, float>(
                        weight_buffer, mat_ref,
                        weight_buffer, bias_ref,
                        input_vec);

                    // Write result to Y
                    for (auto n : dynamic_range(N)) {
                        Y[n] = result[n];
                    }
                } else {
                    // No compatible bias — use cooperative_mat_mul, then add bias manually
                    auto result = cooperative_mat_mul<float, float>(
                        weight_buffer, mat_ref,
                        input_vec);

                    // Write result to Y with optional bias
                    for (auto n : dynamic_range(N)) {
                        auto val = result[n];
                        add_bias(val, 0u, n);
                        Y[n] = val;
                    }
                }
                return;
            }
        }

        // Vectorized warp path: each lane accumulates a chunk of K with VecT reads,
        // then warp_active_sum reduces across the warp. No warp_read_lane inside loops.
        if constexpr (luisa::is_floating_point_v<value_type>) {
            bool can_vec_warp = detail::VecDispatch<VT>::supported &&
                                A.container().is_byte_buffer() && B.container().is_byte_buffer() &&
                                M == 1 && transB_ == 1 && B.strides()[1] == 1 &&
                                ((transA_ == 0 && A.strides()[1] == 1) || (transA_ == 1 && A.strides()[0] == 1)) &&
                                warp_size_ > 1 && N >= warp_size_ && K >= warp_size_ &&
                                K % warp_size_ == 0 &&
                                K % (warp_size_ * detail::VecDispatch<VT>::N) == 0;
            if (can_vec_warp) {
                using VecT = typename detail::VecDispatch<VT>::VecT;
                auto lane_id = warp_lane_id();
                auto chunk_size = K / warp_size_;
                auto vec_chunk = chunk_size / detail::VecDispatch<VT>::N;
                auto buf_a = A.container().get_byte_buffer();
                auto buf_b = B.container().get_byte_buffer();
                auto off_a = static_cast<uint>(A.container().get_byte_offset());
                auto off_b = static_cast<uint>(B.container().get_byte_offset());
                auto type_size = static_cast<uint>(sizeof(VT));
                auto a_base_stride = transA_ ? A.strides()[1] : A.strides()[0];
                auto b_base_stride = transB_ ? B.strides()[0] : B.strides()[1];
                auto base_a = off_a + (0u * a_base_stride) * type_size;
                for (auto n : dynamic_range(N)) {
                    auto local_sum = def(VT{0});
                    auto base_b = off_b + (cast<uint>(n) * b_base_stride) * type_size;
                    for (auto v : dynamic_range(vec_chunk)) {
                        auto k_offset = (lane_id * chunk_size + v * detail::VecDispatch<VT>::N) * type_size;
                        auto a_vec = buf_a->read<VecT>(base_a + k_offset);
                        auto b_vec = buf_b->read<VecT>(base_b + k_offset);
                        local_sum = luisa::compute::fma(a_vec.x, b_vec.x, local_sum);
                        local_sum = luisa::compute::fma(a_vec.y, b_vec.y, local_sum);
                        local_sum = luisa::compute::fma(a_vec.z, b_vec.z, local_sum);
                        local_sum = luisa::compute::fma(a_vec.w, b_vec.w, local_sum);
                    }
                    auto sum = warp_active_sum(local_sum);
                    apply_alpha(sum);
                    add_bias(sum, 0u, n);
                    Y[n] = sum;
                }
                return;
            }
        }

        if (M == 1 && N == 1) {
            auto sum = compute_dot(0u, 0u);
            apply_alpha(sum);
            add_bias(sum, 0u, 0u);
            Y[0u] = sum;
        } else if (M == 1) {
            for (auto n : dynamic_range(N)) {
                auto sum = compute_dot(0u, n);
                apply_alpha(sum);
                add_bias(sum, 0u, n);
                Y[n] = sum;
            }
        } else if (N == 1) {
            for (auto m : dynamic_range(M)) {
                auto sum = compute_dot(m, 0u);
                apply_alpha(sum);
                add_bias(sum, m, 0u);
                Y[m] = sum;
            }
        } else {
            for (auto m : dynamic_range(M)) {
                for (auto n : dynamic_range(N)) {
                    auto sum = compute_dot(m, n);
                    apply_alpha(sum);
                    add_bias(sum, m, n);
                    Y(m, n) = sum;
                }
            }
        }
    }

public:
    Gemm(float alpha, float beta, int transA, int transB)
        : Operator("Gemm"), alpha_(alpha), beta_(beta), transA_(transA), transB_(transB) {}

    void set_environment(NetworkInstance &env, TensorTable &) override {
        warp_size_ = env.warp_size();
        use_coop_vec_ = env.use_cooperative_vector();
    }

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() >= 2 && inputs.size() <= 3 && outputs.size() == 1,
                     "Gemm requires 2 or 3 inputs and 1 output.");
#endif
        auto &A = inputs[0].get();
        auto &B = inputs[1].get();
        ITensor *C_ptr = (inputs.size() == 3) ? &inputs[2].get() : nullptr;
        auto &Y = outputs[0].get();

#ifndef NDEBUG
        LUISA_ASSERT(A.element_type() == B.element_type() && A.element_type() == Y.element_type(),
                     "Gemm: all tensors must have the same element type.");
        LUISA_ASSERT(!C_ptr || C_ptr->element_type() == A.element_type(),
                     "Gemm: C must have the same element type as A and B.");
        LUISA_ASSERT(A.ndim() == 2 && B.ndim() == 2,
                     "Gemm: A and B must be 2D tensors.");
#endif

        visit_typeid<NNFilteredTypeList<GemmTensorRequirements>>(A.element_type(), [&]<typename T>() {
            apply(static_cast<NNTensor<T> &>(A),
                  static_cast<NNTensor<T> &>(B),
                  C_ptr,
                  static_cast<NNTensor<T> &>(Y));
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(Gemm) {
    float alpha = 1.0f;
    float beta = 1.0f;
    int transA = 0;
    int transB = 0;

    if (auto p = node.try_get_attr("alpha")) {
        alpha = p->get<onnx::AttributeType::FLOAT>();
    }
    if (auto p = node.try_get_attr("beta")) {
        beta = p->get<onnx::AttributeType::FLOAT>();
    }
    if (auto p = node.try_get_attr("transA")) {
        transA = p->get<onnx::AttributeType::INT>();
    }
    if (auto p = node.try_get_attr("transB")) {
        transB = p->get<onnx::AttributeType::INT>();
    }

    return std::make_unique<Gemm>(alpha, beta, transA, transB);
};

}// namespace lcml::onnx
