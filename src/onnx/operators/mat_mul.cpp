#include "onnx/operator.h"
#include "onnx/network_instance.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

namespace lcml::onnx {

// MatMul: general matrix multiplication supporting N-D batch dimensions.
// ONNX spec: input[0]=A, input[1]=B; output = A @ B with NumPy broadcasting on batch dims
class MatMul : public Operator {
    uint warp_size_ = 0;
public:
    MatMul() : Operator("MatMul") {}

    void set_environment(NetworkInstance &env, TensorTable &) override {
        warp_size_ = env.warp_size();
    }

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 2 && outputs.size() == 1,
                     "MatMul requires 2 inputs and 1 output.");
#endif
        auto &A = inputs[0].get();
        auto &B = inputs[1].get();
        auto &Y = outputs[0].get();

        auto const &a_shape = A.shape();
        auto const &b_shape = B.shape();
        auto const &y_shape = Y.shape();

        visit_typeid<NNFilteredTypeList<IsFloatingPoint>>(A.element_type(), [&]<typename T>() {
            using VT = nn_storage_type_t<T>;
            auto &a = static_cast<NNTensor<T> &>(A);
            auto &b = static_cast<NNTensor<T> &>(B);
            auto &y = static_cast<NNTensor<T> &>(Y);

            auto a_ndim = a_shape.size();
            auto b_ndim = b_shape.size();
            auto y_ndim = y_shape.size();

            // Extract M, K, N from the last two dims (ONNX promotes 1D -> 2D)
            uint32_t M = (a_ndim >= 2) ? a_shape[a_ndim - 2] : 1;
            uint32_t K = a_shape[a_ndim - 1];
            uint32_t N = (b_ndim >= 2) ? b_shape[b_ndim - 1] : 1;

            // No batch dims: use direct 2D indexing (also covers 1D tensor promotion)
            if (y_ndim <= 2) {
                auto a_col_stride = a.strides()[a_ndim - 1];
                auto b_row_stride = (b_ndim >= 2) ? b.strides()[b_ndim - 2] : b.strides()[0];

                // Vectorized warp path: each lane accumulates a chunk of K with VecT reads,
                // then warp_active_sum reduces across the warp. No warp ops inside inner loops.
                if constexpr (luisa::is_floating_point_v<VT>) {
                    bool can_vec_warp = detail::VecDispatch<VT>::supported &&
                                        M == 1 &&
                                        a_col_stride == 1 && b_row_stride == 1 &&
                                        a.container().is_byte_buffer() && b.container().is_byte_buffer() &&
                                        warp_size_ > 1 && K >= warp_size_ &&
                                        K % (warp_size_ * detail::VecDispatch<VT>::N) == 0;
                    if (can_vec_warp) {
                        using VecT = typename detail::VecDispatch<VT>::VecT;
                        auto lane_id = warp_lane_id();
                        auto chunk_size = K / warp_size_;
                        auto vec_chunk = chunk_size / detail::VecDispatch<VT>::N;
                        auto buf_a = a.container().get_byte_buffer();
                        auto buf_b = b.container().get_byte_buffer();
                        auto off_a = static_cast<uint>(a.container().get_byte_offset());
                        auto off_b = static_cast<uint>(b.container().get_byte_offset());
                        auto type_size = static_cast<uint>(sizeof(VT));
                        auto a_row_stride = (a_ndim >= 2) ? a.strides()[a_ndim - 2] : 0u;
                        auto b_col_stride = b.strides()[b_ndim - 1];
                        auto base_a = off_a + (0u * a_row_stride) * type_size;
                        for (auto n : dynamic_range(N)) {
                            auto local_sum = def(VT{0});
                            auto base_b = off_b + (cast<uint>(n) * b_col_stride) * type_size;
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
                            if (N == 1) {
                                y[0u] = sum;
                            } else {
                                y[n] = sum;
                            }
                        }
                        return;
                    }
                }

                if constexpr (detail::VecDispatch<VT>::supported) {
                    if (a_col_stride == 1 && b_row_stride == 1 &&
                        a.container().is_byte_buffer() && b.container().is_byte_buffer()) {
                        using VecT = typename detail::VecDispatch<VT>::VecT;
                        auto buf_a = a.container().get_byte_buffer();
                        auto buf_b = b.container().get_byte_buffer();
                        auto off_a = static_cast<uint>(a.container().get_byte_offset());
                        auto off_b = static_cast<uint>(b.container().get_byte_offset());
                        auto vec_n = K / detail::VecDispatch<VT>::N;
                        auto rem = K % detail::VecDispatch<VT>::N;
                        auto type_size = static_cast<uint>(sizeof(VT));
                        auto a_row_stride = (a_ndim >= 2) ? a.strides()[a_ndim - 2] : 0u;
                        auto b_col_stride = b.strides()[b_ndim - 1];

                        auto compute_dot_vec = [&](auto m, auto n) {
                            auto sum = def(VT{0});
                            auto base_a = off_a + (cast<uint>(m) * a_row_stride) * type_size;
                            auto base_b = off_b + (cast<uint>(n) * b_col_stride) * type_size;
                            auto acc4 = def(VecT(VT{0}));
                            for (auto kblk : dynamic_range(vec_n)) {
                                auto byte_k = kblk * detail::VecDispatch<VT>::N * type_size;
                                auto va = buf_a->read<VecT>(base_a + byte_k);
                                auto vb = buf_b->read<VecT>(base_b + byte_k);
                                acc4 = luisa::compute::fma(va, vb, acc4);
                            }
                            sum += acc4.x + acc4.y + acc4.z + acc4.w;
                            for (auto k : dynamic_range(rem)) {
                                auto kk = vec_n * detail::VecDispatch<VT>::N + k;
                                auto byte_k = kk * type_size;
                                auto va = buf_a->read<VT>(base_a + byte_k);
                                auto vb = buf_b->read<VT>(base_b + byte_k);
                                sum = luisa::compute::fma(va, vb, sum);
                            }
                            return sum;
                        };

                        if (M == 1 && N == 1) {
                            y[0u] = compute_dot_vec(0u, 0u);
                        } else if (M == 1) {
                            for (auto n : dynamic_range(N)) {
                                y[n] = compute_dot_vec(0u, n);
                            }
                        } else if (N == 1) {
                            for (auto m : dynamic_range(M)) {
                                y[m] = compute_dot_vec(m, 0u);
                            }
                        } else {
                            for (auto m : dynamic_range(M)) {
                                for (auto n : dynamic_range(N)) {
                                    y(m, n) = compute_dot_vec(m, n);
                                }
                            }
                        }
                        return;
                    }
                }

                auto a_row_stride = (a_ndim >= 2) ? a.strides()[a_ndim - 2] : 0u;
                auto b_col_stride = b.strides()[b_ndim - 1];

                auto compute_dot = [&](auto m, auto n) {
                    auto sum = def(VT{0});
                    for (auto k : dynamic_range(K)) {
                        auto a_idx = cast<uint>(m) * a_row_stride + cast<uint>(k) * a_col_stride;
                        auto b_idx = cast<uint>(k) * b_row_stride + cast<uint>(n) * b_col_stride;
                        sum = luisa::compute::fma(a[a_idx], b[b_idx], sum);
                    }
                    return sum;
                };

                if (M == 1 && N == 1) {
                    y[0u] = compute_dot(0u, 0u);
                } else if (M == 1) {
                    for (auto n : dynamic_range(N)) {
                        y[n] = compute_dot(0u, n);
                    }
                } else if (N == 1) {
                    for (auto m : dynamic_range(M)) {
                        y[m] = compute_dot(m, 0u);
                    }
                } else {
                    for (auto m : dynamic_range(M)) {
                        for (auto n : dynamic_range(N)) {
                            y(m, n) = compute_dot(m, n);
                        }
                    }
                }
            } else {
                // Batched matmul: iterate output elements with right-aligned broadcasting
                auto batch_dims = (y_ndim >= 2) ? y_ndim - 2 : 0;
                auto a_batch = (a_ndim >= 2) ? a_ndim - 2 : 0;
                auto b_batch = (b_ndim >= 2) ? b_ndim - 2 : 0;

                auto a_row_stride = (a_ndim >= 2) ? a.strides()[a_ndim - 2] : 0u;
                auto a_col_stride = a.strides()[a_ndim - 1];
                auto b_row_stride = (b_ndim >= 2) ? b.strides()[b_ndim - 2] : b.strides()[0];
                auto b_col_stride = b.strides()[b_ndim - 1];

                auto a_batch_offset_idx = (batch_dims > a_batch) ? (batch_dims - a_batch) : 0;
                auto b_batch_offset_idx = (batch_dims > b_batch) ? (batch_dims - b_batch) : 0;

                for (auto linear_out : dynamic_range(y.size())) {
                    auto m_coord = def(0u);
                    auto n_coord = def(0u);
                    auto a_batch_offset = def(0u);
                    auto b_batch_offset = def(0u);

                    for_each_dim(linear_out, y.strides(), y_ndim, [&](uint32_t d, auto coord) {
                        if (d < batch_dims) {
                            if (d >= a_batch_offset_idx && a_batch > 0) {
                                auto a_dim_idx = d - a_batch_offset_idx;
                                if (a_shape[a_dim_idx] > 1) a_batch_offset += coord * a.strides()[a_dim_idx];
                            }
                            if (d >= b_batch_offset_idx && b_batch > 0) {
                                auto b_dim_idx = d - b_batch_offset_idx;
                                if (b_shape[b_dim_idx] > 1) b_batch_offset += coord * b.strides()[b_dim_idx];
                            }
                        } else if (d == y_ndim - 2) {
                            m_coord = coord;
                        } else {
                            n_coord = coord;
                        }
                    }, y.size());

                    if constexpr (detail::VecDispatch<VT>::supported) {
                        bool can_vec_warp = luisa::is_floating_point_v<VT> &&
                                            M == 1 &&
                                            a_col_stride == 1 && b_row_stride == 1 &&
                                            a.container().is_byte_buffer() && b.container().is_byte_buffer() &&
                                            warp_size_ > 1 && K >= warp_size_ &&
                                            K % (warp_size_ * detail::VecDispatch<VT>::N) == 0;
                        if (can_vec_warp) {
                            using VecT = typename detail::VecDispatch<VT>::VecT;
                            auto lane_id = warp_lane_id();
                            auto chunk_size = K / warp_size_;
                            auto vec_chunk = chunk_size / detail::VecDispatch<VT>::N;
                            auto buf_a = a.container().get_byte_buffer();
                            auto buf_b = b.container().get_byte_buffer();
                            auto off_a = static_cast<uint>(a.container().get_byte_offset());
                            auto off_b = static_cast<uint>(b.container().get_byte_offset());
                            auto type_size = static_cast<uint>(sizeof(VT));
                            auto base_a = off_a + (a_batch_offset + m_coord * a_row_stride) * type_size;
                            auto base_b = off_b + (b_batch_offset + n_coord * b_col_stride) * type_size;
                            auto local_sum = def(VT{0});
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
                            y[linear_out] = sum;
                        } else if (a_col_stride == 1 && b_row_stride == 1 &&
                                   a.container().is_byte_buffer() && b.container().is_byte_buffer()) {
                            using VecT = typename detail::VecDispatch<VT>::VecT;
                            auto buf_a = a.container().get_byte_buffer();
                            auto buf_b = b.container().get_byte_buffer();
                            auto off_a = static_cast<uint>(a.container().get_byte_offset());
                            auto off_b = static_cast<uint>(b.container().get_byte_offset());
                            auto vec_n = K / detail::VecDispatch<VT>::N;
                            auto rem = K % detail::VecDispatch<VT>::N;
                            auto type_size = static_cast<uint>(sizeof(VT));

                            auto sum = def(VT{0});
                            auto base_a = off_a + (a_batch_offset + m_coord * a_row_stride) * type_size;
                            auto base_b = off_b + (b_batch_offset + n_coord * b_col_stride) * type_size;
                            auto acc4 = def(VecT(VT{0}));
                            for (auto kblk : dynamic_range(vec_n)) {
                                auto byte_k = kblk * detail::VecDispatch<VT>::N * type_size;
                                auto va = buf_a->read<VecT>(base_a + byte_k);
                                auto vb = buf_b->read<VecT>(base_b + byte_k);
                                acc4 = luisa::compute::fma(va, vb, acc4);
                            }
                            sum += acc4.x + acc4.y + acc4.z + acc4.w;
                            for (auto k : dynamic_range(rem)) {
                                auto kk = vec_n * detail::VecDispatch<VT>::N + k;
                                auto byte_k = kk * type_size;
                                auto va = buf_a->read<VT>(base_a + byte_k);
                                auto vb = buf_b->read<VT>(base_b + byte_k);
                                sum = luisa::compute::fma(va, vb, sum);
                            }
                            y[linear_out] = sum;
                        } else {
                            auto sum = def(VT{0});
                            for (auto k : dynamic_range(K)) {
                                auto a_idx = a_batch_offset + m_coord * a_row_stride + k * a_col_stride;
                                auto b_idx = b_batch_offset + k * b_row_stride + n_coord * b_col_stride;
                                sum = luisa::compute::fma(a[a_idx], b[b_idx], sum);
                            }
                            y[linear_out] = sum;
                        }
                    } else {
                        auto sum = def(VT{0});
                        for (auto k : dynamic_range(K)) {
                            auto a_idx = a_batch_offset + m_coord * a_row_stride + k * a_col_stride;
                            auto b_idx = b_batch_offset + k * b_row_stride + n_coord * b_col_stride;
                            sum = luisa::compute::fma(a[a_idx], b[b_idx], sum);
                        }
                        y[linear_out] = sum;
                    }
                }
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(MatMul) {
    return std::make_unique<MatMul>();
};

}// namespace lcml::onnx
