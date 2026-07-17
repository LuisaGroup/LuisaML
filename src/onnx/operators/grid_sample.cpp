#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>

namespace lcml::onnx {

template<typename T>
struct GridSampleSupported : std::bool_constant<
    IsFloatingPoint<T>::value ||
    std::is_same_v<T, FP4E2M1> ||
    std::is_same_v<T, FP8E4M3FN> ||
    std::is_same_v<T, FP8E5M2> ||
    std::is_same_v<T, FP16Quantized>> {};

class GridSample : public Operator {
private:
    int32_t align_corners_;
    luisa::string mode_;
    luisa::string padding_mode_;

public:
    GridSample(int32_t align_corners, luisa::string mode, luisa::string padding_mode)
        : Operator("GridSample"), align_corners_(align_corners),
          mode_(std::move(mode)), padding_mode_(std::move(padding_mode)) {}

    void forward(luisa::span<std::reference_wrapper<ITensor>> inputs,
                 luisa::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 2 && outputs.size() == 1,
                     "GridSample requires 2 inputs and 1 output.");
#endif
        auto &X = inputs[0].get();
        auto &Grid = inputs[1].get();
        auto &Y = outputs[0].get();

        auto const &x_shape = X.shape();   // (N, C, H_in, W_in)
        auto const &g_shape = Grid.shape();// (N, H_out, W_out, 2)
        auto const &y_shape = Y.shape();   // (N, C, H_out, W_out)

        uint32_t N = x_shape[0], C = x_shape[1];
        uint32_t iH = x_shape[2], iW = x_shape[3];
        uint32_t oH = g_shape[1], oW = g_shape[2];

        visit_type_index<NNFilteredTypeList<IsFloatingPoint>>(Grid.element_type_index(), [&]<typename GT>() {
            using GT_VT = nn_storage_type_t<GT>;
            visit_type_index<NNFilteredTypeList<GridSampleSupported>>(X.element_type_index(), [&]<typename XT>() {
                using VT = nn_storage_type_t<XT>;
                using CT = std::conditional_t<
                    std::is_same_v<XT, FP4E2M1> || std::is_same_v<XT, FP8E4M3FN> ||
                    std::is_same_v<XT, FP8E5M2> || std::is_same_v<XT, FP16Quantized>,
                    half, VT>;

                auto &grid = static_cast<NNTensor<GT> &>(Grid);
                auto &x = static_cast<NNTensor<XT> &>(X);
                auto &y = static_cast<NNTensor<XT> &>(Y);

                auto dequant_x = [&](auto v) -> Var<CT> {
                    if constexpr (std::is_same_v<XT, FP4E2M1>) {
                        static auto c = fp4e2m1_to_float();
                        return c(v.bits.cast<ushort>());
                    } else if constexpr (std::is_same_v<XT, FP8E4M3FN>) {
                        static auto c = fp8e4m3_to_float();
                        return c(v.bits.cast<ushort>());
                    } else if constexpr (std::is_same_v<XT, FP8E5M2>) {
                        static auto c = fp8e5m2_to_float();
                        return c(v.bits.cast<ushort>());
                    } else if constexpr (std::is_same_v<XT, FP16Quantized>) {
                        return cast<CT>(v.bits);
                    } else {
                        return cast<CT>(v);
                    }
                };

                auto store_y = [&](auto idx, auto val) {
                    if constexpr (std::is_same_v<XT, FP4E2M1>) {
                        static auto c = fp4e2m1_from_float();
                        y[idx].bits = c(val.template cast<half>()).cast<uint16_t>();
                    } else if constexpr (std::is_same_v<XT, FP8E4M3FN>) {
                        static auto c = fp8e4m3_from_float();
                        y[idx].bits = c(val.template cast<half>()).cast<uint16_t>();
                    } else if constexpr (std::is_same_v<XT, FP8E5M2>) {
                        static auto c = fp8e5m2_from_float();
                        y[idx].bits = c(val.template cast<half>()).cast<uint16_t>();
                    } else if constexpr (std::is_same_v<XT, FP16Quantized>) {
                        y[idx].bits = val.template cast<half>();
                    } else {
                        y[idx] = val.template cast<VT>();
                    }
                };

                auto x_stride_n = x.strides()[0];
                auto x_stride_c = x.strides()[1];
                auto x_stride_h = x.strides()[2];
                auto x_stride_w = x.strides()[3];
                auto y_stride_n = y.strides()[0];
                auto y_stride_c = y.strides()[1];
                auto y_stride_h = y.strides()[2];
                auto y_stride_w = y.strides()[3];
                auto g_stride_n = grid.strides()[0];
                auto g_stride_h = grid.strides()[1];
                auto g_stride_w = grid.strides()[2];

                constexpr bool is_native = std::is_same_v<XT, float> || std::is_same_v<XT, half>;
                bool x_scalar_buf = is_native && x.container().is_byte_buffer();
                bool y_scalar_buf = is_native && y.container().is_byte_buffer();
                bool x_vec_buf = x_scalar_buf && x_stride_c == 1;
                bool y_vec_buf = y_scalar_buf && y_stride_c == 1;

                Var<ByteBuffer> *buf_x = nullptr;
                uint off_x = 0;
                if (x_scalar_buf) {
                    buf_x = x.container().get_byte_buffer();
                    off_x = static_cast<uint>(x.container().get_byte_offset());
                }
                Var<ByteBuffer> *buf_y = nullptr;
                uint off_y = 0;
                if (y_scalar_buf) {
                    buf_y = y.container().get_byte_buffer();
                    off_y = static_cast<uint>(y.container().get_byte_offset());
                }

                auto one = CT(1.0f);
                uint32_t c_vec_end = (C / 4u) * 4u;

                // Denormalize: convert normalized grid coordinate [-1,1] to pixel coordinate
                auto denorm_x = [&](auto gx) {
                    if (align_corners_) {
                        return (gx + one) * CT(static_cast<float>(iW - 1)) * CT(0.5f);
                    } else {
                        return ((gx + one) * CT(static_cast<float>(iW)) - one) * CT(0.5f);
                    }
                };
                auto denorm_y = [&](auto gy) {
                    if (align_corners_) {
                        return (gy + one) * CT(static_cast<float>(iH - 1)) * CT(0.5f);
                    } else {
                        return ((gy + one) * CT(static_cast<float>(iH)) - one) * CT(0.5f);
                    }
                };

                // Apply padding mode to a coordinate; returns (clamped_coord, in_bounds_flag)
                auto apply_pad = [&](auto coord, uint32_t dim_size) -> std::pair<decltype(coord), decltype(coord >= CT(0.0f))> {
                    auto lo = CT(0.0f);
                    auto hi = CT(static_cast<float>(dim_size - 1));
                    if (padding_mode_ == "border") {
                        return {clamp(coord, lo, hi), Var<bool>{true}};
                    } else if (padding_mode_ == "reflection") {
                        auto reflect = [&](auto c, float lo_f, float hi_f) {
                            auto range = CT(hi_f - lo_f);
                            auto shifted = c - CT(lo_f);
                            auto double_range = range * CT(2.0f);
                            auto folded = shifted - floor(shifted / double_range) * double_range;
                            auto result = select(double_range - folded, folded, folded <= range);
                            return result + CT(lo_f);
                        };
                        decltype(coord) reflected;
                        if (align_corners_) {
                            reflected = reflect(coord, 0.0f, static_cast<float>(dim_size - 1));
                        } else {
                            reflected = reflect(coord, -0.5f, static_cast<float>(dim_size) - 0.5f);
                        }
                        return {clamp(reflected, lo, hi), Var<bool>{true}};
                    } else {
                        auto in_bounds = (coord >= lo) & (coord <= hi);
                        return {clamp(coord, lo, hi), in_bounds};
                    }
                };

                auto safe_fetch = [&](auto base, auto iy_u, auto ix_u,
                                      auto in_bounds_y, auto in_bounds_x) {
                    auto valid = in_bounds_y & in_bounds_x;
                    Var<CT> val;
                    if constexpr (is_native) {
                        if (x_scalar_buf) {
                            val = buf_x->read<CT>(off_x + (base + iy_u * x_stride_h + ix_u * x_stride_w) * static_cast<uint>(sizeof(CT)));
                        } else {
                            val = dequant_x(x[base + iy_u * x_stride_h + ix_u * x_stride_w]);
                        }
                    } else {
                        val = dequant_x(x[base + iy_u * x_stride_h + ix_u * x_stride_w]);
                    }
                    return select(CT(0.0f), val, valid);
                };

                for (auto n : dynamic_range(N)) {
                    for (auto oh : dynamic_range(oH)) {
                        for (auto ow : dynamic_range(oW)) {
                            auto gather4 = [&](uint32_t cv, auto iy_u, auto ix_u) {
                                if constexpr (is_native) {
                                    if (x_vec_buf) {
                                        auto base = n * x_stride_n + iy_u * x_stride_h + ix_u * x_stride_w;
                                        using VecT = typename detail::VecDispatch<CT>::VecT;
                                        return buf_x->read<VecT>(off_x + (base + cv) * static_cast<uint>(sizeof(CT)));
                                    }
                                }
                                auto b0 = n * x_stride_n + (cv + 0u) * x_stride_c + iy_u * x_stride_h + ix_u * x_stride_w;
                                auto b1 = n * x_stride_n + (cv + 1u) * x_stride_c + iy_u * x_stride_h + ix_u * x_stride_w;
                                auto b2 = n * x_stride_n + (cv + 2u) * x_stride_c + iy_u * x_stride_h + ix_u * x_stride_w;
                                auto b3 = n * x_stride_n + (cv + 3u) * x_stride_c + iy_u * x_stride_h + ix_u * x_stride_w;
                                if constexpr (std::is_same_v<CT, float>) {
                                    return make_float4(dequant_x(x[b0]), dequant_x(x[b1]),
                                                       dequant_x(x[b2]), dequant_x(x[b3]));
                                } else {
                                    return make_half4(dequant_x(x[b0]), dequant_x(x[b1]),
                                                      dequant_x(x[b2]), dequant_x(x[b3]));
                                }
                            };

                            auto store_vec = [&](uint32_t cv, auto vec) {
                                if constexpr (is_native) {
                                    if (y_vec_buf) {
                                        auto y_base = n * y_stride_n + oh * y_stride_h + ow * y_stride_w;
                                        using VecT = typename detail::VecDispatch<CT>::VecT;
                                        buf_y->write(off_y + (y_base + cv) * static_cast<uint>(sizeof(CT)), vec);
                                        return;
                                    }
                                }
                                auto y_base = n * y_stride_n + oh * y_stride_h + ow * y_stride_w;
                                store_y(y_base + (cv + 0u) * y_stride_c, vec.x);
                                store_y(y_base + (cv + 1u) * y_stride_c, vec.y);
                                store_y(y_base + (cv + 2u) * y_stride_c, vec.z);
                                store_y(y_base + (cv + 3u) * y_stride_c, vec.w);
                            };
                            auto grid_base = n * g_stride_n + oh * g_stride_h + ow * g_stride_w;
                            auto gx = cast<CT>(grid[grid_base + 0u]);
                            auto gy = cast<CT>(grid[grid_base + 1u]);

                            auto ix = denorm_x(gx);
                            auto iy = denorm_y(gy);

                            if (mode_ == "nearest") {
                                auto ix_n = floor(ix + CT(0.5f));
                                auto iy_n = floor(iy + CT(0.5f));
                                auto [ix_pad, ix_valid] = apply_pad(ix_n, iW);
                                auto [iy_pad, iy_valid] = apply_pad(iy_n, iH);
                                auto iy_u = iy_pad.template cast<uint>();
                                auto ix_u = ix_pad.template cast<uint>();
                                auto valid = ix_valid & iy_valid;

                                if (c_vec_end > 0) {
                                    for (uint32_t cv = 0; cv < c_vec_end; cv += 4u) {
                                        auto val_vec = gather4(cv, iy_u, ix_u) * cast<CT>(valid);
                                        store_vec(cv, val_vec);
                                    }
                                }
                                for (uint32_t c = c_vec_end; c < C; ++c) {
                                    auto base = n * x_stride_n + c * x_stride_c;
                                    Var<CT> val;
                                    if constexpr (is_native) {
                                        if (x_scalar_buf) {
                                            val = buf_x->read<CT>(off_x + (base + iy_u * x_stride_h + ix_u * x_stride_w) * static_cast<uint>(sizeof(CT)));
                                        } else {
                                            val = dequant_x(x[base + iy_u * x_stride_h + ix_u * x_stride_w]);
                                        }
                                    } else {
                                        val = dequant_x(x[base + iy_u * x_stride_h + ix_u * x_stride_w]);
                                    }
                                    auto y_idx = n * y_stride_n + c * y_stride_c + oh * y_stride_h + ow * y_stride_w;
                                    if constexpr (is_native) {
                                        if (y_scalar_buf) {
                                            buf_y->write(off_y + y_idx * static_cast<uint>(sizeof(CT)), select(CT(0.0f), val, valid));
                                        } else {
                                            store_y(y_idx, select(CT(0.0f), val, valid));
                                        }
                                    } else {
                                        store_y(y_idx, select(CT(0.0f), val, valid));
                                    }
                                }
                            } else if (mode_ == "bicubic") {
                                auto ix_floor = floor(ix);
                                auto iy_floor = floor(iy);
                                auto fx = ix - ix_floor;
                                auto fy = iy - iy_floor;

                                auto cubic_weight = [&](auto t) {
                                    auto at = abs(t);
                                    auto at2 = at * at;
                                    auto at3 = at2 * at;
                                    auto a = CT(-0.75f);
                                    auto w_near = fma(a + CT(2.0f), at3, fma(-(a + CT(3.0f)), at2, one));
                                    auto w_far = fma(a, at3, fma(CT(-5.0f) * a, at2, fma(CT(8.0f) * a, at, CT(-4.0f) * a)));
                                    return select(w_far, w_near, at < one);
                                };

                                auto wx0 = cubic_weight(fx + one);
                                auto wx1 = cubic_weight(fx);
                                auto wx2 = cubic_weight(one - fx);
                                auto wx3 = cubic_weight(CT(2.0f) - fx);
                                auto wy0 = cubic_weight(fy + one);
                                auto wy1 = cubic_weight(fy);
                                auto wy2 = cubic_weight(one - fy);
                                auto wy3 = cubic_weight(CT(2.0f) - fy);

                                auto py_m1 = apply_pad(iy_floor + CT(-1.0f), iH);
                                auto py_0  = apply_pad(iy_floor + CT(0.0f),  iH);
                                auto py_p1 = apply_pad(iy_floor + CT(1.0f),  iH);
                                auto py_p2 = apply_pad(iy_floor + CT(2.0f),  iH);
                                auto px_m1 = apply_pad(ix_floor + CT(-1.0f), iW);
                                auto px_0  = apply_pad(ix_floor + CT(0.0f),  iW);
                                auto px_p1 = apply_pad(ix_floor + CT(1.0f),  iW);
                                auto px_p2 = apply_pad(ix_floor + CT(2.0f),  iW);

                                auto py_m1_u = py_m1.first.template cast<uint>();
                                auto py_0_u  = py_0.first.template cast<uint>();
                                auto py_p1_u = py_p1.first.template cast<uint>();
                                auto py_p2_u = py_p2.first.template cast<uint>();
                                auto px_m1_u = px_m1.first.template cast<uint>();
                                auto px_0_u  = px_0.first.template cast<uint>();
                                auto px_p1_u = px_p1.first.template cast<uint>();
                                auto px_p2_u = px_p2.first.template cast<uint>();

                                if (c_vec_end > 0) {
                                    for (uint32_t cv = 0; cv < c_vec_end; cv += 4u) {
                                        auto r0 = fma(wx0, gather4(cv, py_m1_u, px_m1_u), fma(wx1, gather4(cv, py_m1_u, px_0_u), fma(wx2, gather4(cv, py_m1_u, px_p1_u), wx3 * gather4(cv, py_m1_u, px_p2_u))));
                                        auto r1 = fma(wx0, gather4(cv, py_0_u,  px_m1_u), fma(wx1, gather4(cv, py_0_u,  px_0_u), fma(wx2, gather4(cv, py_0_u,  px_p1_u), wx3 * gather4(cv, py_0_u,  px_p2_u))));
                                        auto r2 = fma(wx0, gather4(cv, py_p1_u, px_m1_u), fma(wx1, gather4(cv, py_p1_u, px_0_u), fma(wx2, gather4(cv, py_p1_u, px_p1_u), wx3 * gather4(cv, py_p1_u, px_p2_u))));
                                        auto r3 = fma(wx0, gather4(cv, py_p2_u, px_m1_u), fma(wx1, gather4(cv, py_p2_u, px_0_u), fma(wx2, gather4(cv, py_p2_u, px_p1_u), wx3 * gather4(cv, py_p2_u, px_p2_u))));
                                        auto sum_vec = fma(wy0, r0, fma(wy1, r1, fma(wy2, r2, wy3 * r3)));
                                        store_vec(cv, sum_vec);
                                    }
                                }
                                for (uint32_t c = c_vec_end; c < C; ++c) {
                                    auto base = n * x_stride_n + c * x_stride_c;
                                    auto r0 = fma(wx0, safe_fetch(base, py_m1_u, px_m1_u, py_m1.second, px_m1.second),
                                           fma(wx1, safe_fetch(base, py_m1_u, px_0_u,  py_m1.second, px_0.second),
                                           fma(wx2, safe_fetch(base, py_m1_u, px_p1_u, py_m1.second, px_p1.second),
                                               wx3 * safe_fetch(base, py_m1_u, px_p2_u, py_m1.second, px_p2.second))));
                                    auto r1 = fma(wx0, safe_fetch(base, py_0_u,  px_m1_u, py_0.second,  px_m1.second),
                                           fma(wx1, safe_fetch(base, py_0_u,  px_0_u,  py_0.second,  px_0.second),
                                           fma(wx2, safe_fetch(base, py_0_u,  px_p1_u, py_0.second,  px_p1.second),
                                               wx3 * safe_fetch(base, py_0_u,  px_p2_u, py_0.second,  px_p2.second))));
                                    auto r2 = fma(wx0, safe_fetch(base, py_p1_u, px_m1_u, py_p1.second, px_m1.second),
                                           fma(wx1, safe_fetch(base, py_p1_u, px_0_u,  py_p1.second, px_0.second),
                                           fma(wx2, safe_fetch(base, py_p1_u, px_p1_u, py_p1.second, px_p1.second),
                                               wx3 * safe_fetch(base, py_p1_u, px_p2_u, py_p1.second, px_p2.second))));
                                    auto r3 = fma(wx0, safe_fetch(base, py_p2_u, px_m1_u, py_p2.second, px_m1.second),
                                           fma(wx1, safe_fetch(base, py_p2_u, px_0_u,  py_p2.second, px_0.second),
                                           fma(wx2, safe_fetch(base, py_p2_u, px_p1_u, py_p2.second, px_p1.second),
                                               wx3 * safe_fetch(base, py_p2_u, px_p2_u, py_p2.second, px_p2.second))));
                                    auto sum = fma(wy0, r0, fma(wy1, r1, fma(wy2, r2, wy3 * r3)));
                                    auto y_idx = n * y_stride_n + c * y_stride_c + oh * y_stride_h + ow * y_stride_w;
                                    if constexpr (is_native) {
                                        if (y_scalar_buf) {
                                            buf_y->write(off_y + y_idx * static_cast<uint>(sizeof(CT)), sum);
                                        } else {
                                            store_y(y_idx, sum);
                                        }
                                    } else {
                                        store_y(y_idx, sum);
                                    }
                                }
                            } else {
                                // Bilinear interpolation
                                auto ix_floor = floor(ix);
                                auto iy_floor = floor(iy);
                                auto lx = ix - ix_floor;
                                auto ly = iy - iy_floor;
                                auto hx = one - lx;
                                auto hy = one - ly;

                                auto [ix0_p, ix0_v] = apply_pad(ix_floor, iW);
                                auto [ix1_p, ix1_v] = apply_pad(ix_floor + one, iW);
                                auto [iy0_p, iy0_v] = apply_pad(iy_floor, iH);
                                auto [iy1_p, iy1_v] = apply_pad(iy_floor + one, iH);

                                auto iy0_u = iy0_p.template cast<uint>();
                                auto iy1_u = iy1_p.template cast<uint>();
                                auto ix0_u = ix0_p.template cast<uint>();
                                auto ix1_u = ix1_p.template cast<uint>();

                                if (c_vec_end > 0) {
                                    auto w00 = hy * hx;
                                    auto w01 = hy * lx;
                                    auto w10 = ly * hx;
                                    auto w11 = ly * lx;
                                    for (uint32_t cv = 0; cv < c_vec_end; cv += 4u) {
                                        auto v00_vec = gather4(cv, iy0_u, ix0_u);
                                        auto v01_vec = gather4(cv, iy0_u, ix1_u);
                                        auto v10_vec = gather4(cv, iy1_u, ix0_u);
                                        auto v11_vec = gather4(cv, iy1_u, ix1_u);
                                        auto sum_vec = fma(w00, v00_vec, fma(w01, v01_vec, fma(w10, v10_vec, w11 * v11_vec)));
                                        store_vec(cv, sum_vec);
                                    }
                                }
                                for (uint32_t c = c_vec_end; c < C; ++c) {
                                    auto base = n * x_stride_n + c * x_stride_c;
                                    auto v00 = safe_fetch(base, iy0_u, ix0_u, iy0_v, ix0_v);
                                    auto v01 = safe_fetch(base, iy0_u, ix1_u, iy0_v, ix1_v);
                                    auto v10 = safe_fetch(base, iy1_u, ix0_u, iy1_v, ix0_v);
                                    auto v11 = safe_fetch(base, iy1_u, ix1_u, iy1_v, ix1_v);
                                    auto t0 = fma(hx, v00, lx * v01);
                                    auto t1 = fma(hx, v10, lx * v11);
                                    auto result = fma(ly, t1, hy * t0);
                                    auto y_idx = n * y_stride_n + c * y_stride_c + oh * y_stride_h + ow * y_stride_w;
                                    if constexpr (is_native) {
                                        if (y_scalar_buf) {
                                            buf_y->write(off_y + y_idx * static_cast<uint>(sizeof(CT)), result);
                                        } else {
                                            store_y(y_idx, result);
                                        }
                                    } else {
                                        store_y(y_idx, result);
                                    }
                                }
                            }
                        }
                    }
                }
            });
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(GridSample) {
    int32_t align_corners = 0;
    luisa::string mode = "bilinear";
    luisa::string padding_mode = "zeros";
    if (auto p = node.try_get_attr("align_corners"))
        align_corners = p->get<onnx::AttributeType::INT>();
    if (auto p = node.try_get_attr("mode"))
        mode = p->get<onnx::AttributeType::STRING>();
    if (auto p = node.try_get_attr("padding_mode"))
        padding_mode = p->get<onnx::AttributeType::STRING>();
    return luisa::make_unique<GridSample>(align_corners, std::move(mode), std::move(padding_mode));
};

}// namespace lcml::onnx
