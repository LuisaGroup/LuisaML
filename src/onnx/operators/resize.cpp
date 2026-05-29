#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"
#include "onnx/fp_quantized.h"

namespace lcml::onnx {

// Resize: resizes the input tensor using interpolation.
// ONNX spec (opset 11+): inputs: X, roi, scales, sizes
// Attributes: coordinate_transform_mode, cubic_coeff_a, exclude_outside, extrapolation_value, mode, nearest_mode
class Resize : public Operator {
private:
    std::string mode_;                     // "nearest", "linear", "cubic"
    std::string coordinate_transform_mode_;// "half_pixel", "asymmetric", etc.
    std::string nearest_mode_;             // "round_prefer_floor", etc.
    float cubic_coeff_a_;                  // cubic coefficient (default -0.75)

public:
    Resize(std::string mode, std::string coord_transform, std::string nearest_mode, float cubic_coeff_a)
        : Operator("Resize"), mode_(std::move(mode)),
          coordinate_transform_mode_(std::move(coord_transform)),
          nearest_mode_(std::move(nearest_mode)),
          cubic_coeff_a_(cubic_coeff_a) {}

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() >= 1 && outputs.size() == 1,
                     "Resize requires >=1 input and 1 output.");
        auto &X = inputs[0].get();
        auto &Y = outputs[0].get();

        auto const &x_shape = X.shape();
        auto const &y_shape = Y.shape();
        auto ndim = x_shape.size();
#else
        auto &X = inputs[0].get();
        auto &Y = outputs[0].get();

        auto const &x_shape = X.shape();
        auto const &y_shape = Y.shape();
        auto ndim = x_shape.size();
#endif

        // Helper to safely compute max valid coordinate (defense against zero-dim underflow)
        auto max_coord_f = [&](uint32_t d) -> float {
            return (x_shape[d] > 0) ? static_cast<float>(x_shape[d] - 1) : 0.0f;
        };

        auto const &tid = X.element_type();
        bool is_quantized = (tid == typeid(FP4E2M1) || tid == typeid(FP8E4M3FN) ||
                             tid == typeid(FP8E5M2) || tid == typeid(FP16Quantized));

        if (is_quantized) {
            visit_typeid<FP4E2M1, FP8E4M3FN, FP8E5M2, FP16Quantized>(tid, [&]<typename T>() {
                using QT = half;
                auto &x = static_cast<NNTensor<T> &>(X);
                auto &y = static_cast<NNTensor<T> &>(Y);

                // Compute scale factors per dimension (output_size / input_size)
                std::vector<float> scales(ndim);
                for (size_t d = 0; d < ndim; ++d) {
                    scales[d] = static_cast<float>(y_shape[d]) / static_cast<float>(x_shape[d]);
                }

                auto dequant = [&](auto v) -> Var<QT> {
                    if constexpr (std::is_same_v<T, FP4E2M1>) {
                        static auto c = fp4e2m1_to_float();
                        return c(v.bits.cast<ushort>());
                    } else if constexpr (std::is_same_v<T, FP8E4M3FN>) {
                        static auto c = fp8e4m3_to_float();
                        return c(v.bits.cast<ushort>());
                    } else if constexpr (std::is_same_v<T, FP8E5M2>) {
                        static auto c = fp8e5m2_to_float();
                        return c(v.bits.cast<ushort>());
                    } else if constexpr (std::is_same_v<T, FP16Quantized>) {
                        return v.bits;
                    }
                };
                auto store_y = [&](auto idx, Var<QT> val) {
                    if constexpr (std::is_same_v<T, FP4E2M1>) {
                        static auto c = fp4e2m1_from_float();
                        y[idx].bits = c(val).cast<uint16_t>();
                    } else if constexpr (std::is_same_v<T, FP8E4M3FN>) {
                        static auto c = fp8e4m3_from_float();
                        y[idx].bits = c(val).cast<uint16_t>();
                    } else if constexpr (std::is_same_v<T, FP8E5M2>) {
                        static auto c = fp8e5m2_from_float();
                        y[idx].bits = c(val).cast<uint16_t>();
                    } else if constexpr (std::is_same_v<T, FP16Quantized>) {
                        y[idx].bits = val;
                    }
                };

                auto coord_transform_q = [&](auto out_coord, uint32_t d) {
                    float scale = scales[d];
                    float in_size = static_cast<float>(x_shape[d]);
                    float out_size = static_cast<float>(y_shape[d]);
                    auto out_f = out_coord.template cast<QT>();

                    if (coordinate_transform_mode_ == "asymmetric") {
                        return out_f / Var<QT>{QT(scale)};
                    } else if (coordinate_transform_mode_ == "align_corners") {
                        float ratio = (out_size > 1.0f) ? (in_size - 1.0f) / (out_size - 1.0f) : 0.0f;
                        return out_f * Var<QT>{QT(ratio)};
                    } else if (coordinate_transform_mode_ == "pytorch_half_pixel") {
                        if (out_size > 1.0f) {
                            return (out_f + Var<QT>{QT(0.5f)}) / Var<QT>{QT(scale)} - Var<QT>{QT(0.5f)};
                        } else {
                            return def(QT{0});
                        }
                    } else {
                        return (out_f + Var<QT>{QT(0.5f)}) / Var<QT>{QT(scale)} - Var<QT>{QT(0.5f)};
                    }
                };

                auto nearest_round_q = [&](auto val) {
                    if (nearest_mode_ == "floor") {
                        return floor(val);
                    } else if (nearest_mode_ == "ceil") {
                        return ceil(val);
                    } else if (nearest_mode_ == "round_prefer_ceil") {
                        return floor(val + Var<QT>{QT(0.5f)});
                    } else {
                        return ceil(val - Var<QT>{QT(0.5f)});
                    }
                };

                if (mode_ == "nearest") {
                    for (auto linear_out : dynamic_range(y.size())) {
                        auto in_linear = def(0u);
                        for_each_dim(linear_out, y.strides(), ndim, [&](uint32_t d, auto coord) {
                            auto in_f = coord_transform_q(coord, d);
                            auto in_rounded = nearest_round_q(in_f);
                            auto in_clamped = max(min(in_rounded, Var<QT>{QT(max_coord_f(d))}),
                                                  Var<QT>{QT(0.0f)});
                            in_linear += in_clamped.template cast<uint>() * x.strides()[d];
                        }, y.size());
                        store_y(linear_out, dequant(x[in_linear]));
                    }
                } else if (mode_ == "linear") {
#ifndef NDEBUG
                    LUISA_ASSERT(ndim == 4, "Resize: linear mode only supports 4D tensors.");
#endif
                    uint32_t N = x_shape[0], C = x_shape[1];
                    uint32_t iH = x_shape[2], iW = x_shape[3];
                    uint32_t oH = y_shape[2], oW = y_shape[3];
                    float iH_max = (iH > 0) ? static_cast<float>(iH - 1) : 0.0f;
                    float iW_max = (iW > 0) ? static_cast<float>(iW - 1) : 0.0f;

                    for (auto n : dynamic_range(N)) {
                        for (auto c : dynamic_range(C)) {
                            for (auto oh : dynamic_range(oH)) {
                                for (auto ow : dynamic_range(oW)) {
                                    auto iy_f = coord_transform_q(oh, 2);
                                    auto ix_f = coord_transform_q(ow, 3);

                                    auto iy_floor = floor(iy_f);
                                    auto ix_floor = floor(ix_f);

                                    auto ly = iy_f - iy_floor;
                                    auto lx = ix_f - ix_floor;
                                    auto hy = Var<QT>{QT{1}} - ly;
                                    auto hx = Var<QT>{QT{1}} - lx;

                                    auto clamp_h = [&](auto v) {
                                        return max(min(v, Var<QT>{QT(iH_max)}), Var<QT>{QT(0.0f)}).template cast<uint>();
                                    };
                                    auto clamp_w = [&](auto v) {
                                        return max(min(v, Var<QT>{QT(iW_max)}), Var<QT>{QT(0.0f)}).template cast<uint>();
                                    };

                                    auto iy0 = clamp_h(iy_floor);
                                    auto ix0 = clamp_w(ix_floor);
                                    auto iy1 = clamp_h(iy_floor + Var<QT>{QT{1}});
                                    auto ix1 = clamp_w(ix_floor + Var<QT>{QT{1}});

                                    auto base = n * x.strides()[0] + c * x.strides()[1];
                                    auto v00 = dequant(x[base + iy0 * x.strides()[2] + ix0 * x.strides()[3]]);
                                    auto v01 = dequant(x[base + iy0 * x.strides()[2] + ix1 * x.strides()[3]]);
                                    auto v10 = dequant(x[base + iy1 * x.strides()[2] + ix0 * x.strides()[3]]);
                                    auto v11 = dequant(x[base + iy1 * x.strides()[2] + ix1 * x.strides()[3]]);

                                    auto t0 = fma(hx, v00, lx * v01);
                                    auto t1 = fma(hx, v10, lx * v11);
                                    auto result = fma(hy, t0, ly * t1);
                                    store_y(n * y.strides()[0] + c * y.strides()[1] +
                                            oh * y.strides()[2] + ow * y.strides()[3], result);
                                }
                            }
                        }
                    }
                } else if (mode_ == "cubic") {
#ifndef NDEBUG
                    LUISA_ASSERT(ndim == 4, "Resize: cubic mode only supports 4D tensors.");
#endif
                    uint32_t N = x_shape[0], C = x_shape[1];
                    uint32_t iH = x_shape[2], iW = x_shape[3];
                    uint32_t oH = y_shape[2], oW = y_shape[3];
                    float iH_max = (iH > 0) ? static_cast<float>(iH - 1) : 0.0f;
                    float iW_max = (iW > 0) ? static_cast<float>(iW - 1) : 0.0f;

                    auto cubic_weight_q = [&](auto t) {
                        auto at = abs(t);
                        auto at2 = at * at;
                        auto at3 = at2 * at;
                        auto a = Var<QT>{QT(cubic_coeff_a_)};
                        auto one = Var<QT>{QT(1.0f)};
                        auto w_near = fma(a + Var<QT>{QT(2.0f)}, at3, -(a + Var<QT>{QT(3.0f)}) * at2) + one;
                        auto w_far = fma(a, at3, fma(-Var<QT>{QT(5.0f)} * a, at2, fma(Var<QT>{QT(8.0f)} * a, at, -Var<QT>{QT(4.0f)} * a)));
                        return select(w_far, w_near, at < one);
                    };

                    for (auto n : dynamic_range(N)) {
                        for (auto c : dynamic_range(C)) {
                            for (auto oh : dynamic_range(oH)) {
                                for (auto ow : dynamic_range(oW)) {
                                    auto iy_f = coord_transform_q(oh, 2);
                                    auto ix_f = coord_transform_q(ow, 3);

                                    auto iy_floor = floor(iy_f);
                                    auto ix_floor = floor(ix_f);
                                    auto fy = iy_f - iy_floor;
                                    auto fx = ix_f - ix_floor;

                                    auto one = Var<QT>{QT(1.0f)};
                                    auto two = Var<QT>{QT(2.0f)};

                                    auto wx0 = cubic_weight_q(fx + one);
                                    auto wx1 = cubic_weight_q(fx);
                                    auto wx2 = cubic_weight_q(one - fx);
                                    auto wx3 = cubic_weight_q(two - fx);

                                    auto wy0 = cubic_weight_q(fy + one);
                                    auto wy1 = cubic_weight_q(fy);
                                    auto wy2 = cubic_weight_q(one - fy);
                                    auto wy3 = cubic_weight_q(two - fy);

                                    auto base = n * x.strides()[0] + c * x.strides()[1];
                                    auto fetch = [&](int dy, int dx) {
                                        auto py = max(min(iy_floor + Var<QT>{QT(static_cast<float>(dy))},
                                                          Var<QT>{QT(iH_max)}),
                                                      Var<QT>{QT(0.0f)})
                                                      .template cast<uint>();
                                        auto px = max(min(ix_floor + Var<QT>{QT(static_cast<float>(dx))},
                                                          Var<QT>{QT(iW_max)}),
                                                      Var<QT>{QT(0.0f)})
                                                      .template cast<uint>();
                                        return dequant(x[base + py * x.strides()[2] + px * x.strides()[3]]);
                                    };

                                    auto r0 = fma(wx0, fetch(-1, -1), fma(wx1, fetch(-1, 0), fma(wx2, fetch(-1, 1), wx3 * fetch(-1, 2))));
                                    auto r1 = fma(wx0, fetch(0, -1), fma(wx1, fetch(0, 0), fma(wx2, fetch(0, 1), wx3 * fetch(0, 2))));
                                    auto r2 = fma(wx0, fetch(1, -1), fma(wx1, fetch(1, 0), fma(wx2, fetch(1, 1), wx3 * fetch(1, 2))));
                                    auto r3 = fma(wx0, fetch(2, -1), fma(wx1, fetch(2, 0), fma(wx2, fetch(2, 1), wx3 * fetch(2, 2))));

                                    auto result = fma(wy0, r0, fma(wy1, r1, fma(wy2, r2, wy3 * r3)));
                                    store_y(n * y.strides()[0] + c * y.strides()[1] +
                                            oh * y.strides()[2] + ow * y.strides()[3], result);
                                }
                            }
                        }
                    }
                } else {
#ifndef NDEBUG
                    LUISA_ASSERT(false, "Resize: mode '{}' is not supported.", mode_);
#endif
                }
            });
        } else {
            visit_typeid<NNFilteredTypeList<IsFloatingPoint>>(tid, [&]<typename T>() {
                using VT = nn_storage_type_t<T>;
                auto &x = static_cast<NNTensor<T> &>(X);
                auto &y = static_cast<NNTensor<T> &>(Y);

                // Compute scale factors per dimension (output_size / input_size)
                std::vector<float> scales(ndim);
                for (size_t d = 0; d < ndim; ++d) {
                    scales[d] = static_cast<float>(y_shape[d]) / static_cast<float>(x_shape[d]);
                }

                // ByteBuffer helpers: avoid DynamicArray::operator[] std::visit overhead
                auto is_bb = x.container().is_byte_buffer();
                Var<ByteBuffer> *buf_x = nullptr;
                uint off_x = 0u;
                if (is_bb) {
                    buf_x = x.container().get_byte_buffer();
                    off_x = static_cast<uint>(x.container().get_byte_offset());
                }
                auto read_x = [&](Var<uint> addr) -> Var<VT> {
                    if (is_bb) {
                        return buf_x->read<VT>(off_x + addr * static_cast<uint>(sizeof(VT)));
                    }
                    return x[addr];
                };
                auto write_y = [&](Var<uint> addr, Var<VT> val) {
                    if (y.container().is_byte_buffer()) {
                        auto buf_y = y.container().get_byte_buffer();
                        auto off_y = static_cast<uint>(y.container().get_byte_offset());
                        buf_y->write(off_y + addr * static_cast<uint>(sizeof(VT)), val);
                    } else {
                        y[addr] = val;
                    }
                };

                // Coordinate transform: map output coordinate to input coordinate
                // coord_transform returns the input floating-point coordinate
                auto coord_transform = [&](auto out_coord, uint32_t d) {
                    float scale = scales[d];
                    float in_size = static_cast<float>(x_shape[d]);
                    float out_size = static_cast<float>(y_shape[d]);
                    auto out_f = out_coord.template cast<VT>();

                    if (coordinate_transform_mode_ == "asymmetric") {
                        // in_coord = out_coord / scale
                        return out_f / Var<VT>{VT(scale)};
                    } else if (coordinate_transform_mode_ == "align_corners") {
                        // in_coord = out_coord * (in_size - 1) / (out_size - 1)
                        float ratio = (out_size > 1.0f) ? (in_size - 1.0f) / (out_size - 1.0f) : 0.0f;
                        return out_f * Var<VT>{VT(ratio)};
                    } else if (coordinate_transform_mode_ == "pytorch_half_pixel") {
                        // in_coord = out_size > 1 ? (out_coord + 0.5) / scale - 0.5 : 0
                        if (out_size > 1.0f) {
                            return (out_f + Var<VT>{VT(0.5f)}) / Var<VT>{VT(scale)} - Var<VT>{VT(0.5f)};
                        } else {
                            return def(VT{0});
                        }
                    } else {
                        // "half_pixel" (default) and "tf_half_pixel_for_nn"
                        // in_coord = (out_coord + 0.5) / scale - 0.5
                        return (out_f + Var<VT>{VT(0.5f)}) / Var<VT>{VT(scale)} - Var<VT>{VT(0.5f)};
                    }
                };

                // Nearest mode rounding function
                auto nearest_round = [&](auto val) {
                    if (nearest_mode_ == "floor") {
                        return floor(val);
                    } else if (nearest_mode_ == "ceil") {
                        return ceil(val);
                    } else if (nearest_mode_ == "round_prefer_ceil") {
                        return floor(val + Var<VT>{VT(0.5f)});
                    } else {
                        // "round_prefer_floor" (default)
                        return ceil(val - Var<VT>{VT(0.5f)});
                    }
                };

                if (mode_ == "nearest") {
                    for (auto linear_out : dynamic_range(y.size())) {
                        auto in_linear = def(0u);
                        for_each_dim(linear_out, y.strides(), ndim, [&](uint32_t d, auto coord) {
                            auto in_f = coord_transform(coord, d);
                            auto in_rounded = nearest_round(in_f);
                            // Clamp to valid range
                            auto in_clamped = max(min(in_rounded, Var<VT>{VT(max_coord_f(d))}),
                                                  Var<VT>{VT(0.0f)});
                            in_linear += in_clamped.template cast<uint>() * x.strides()[d];
                        }, y.size());
                        write_y(linear_out, read_x(in_linear));
                    }
                } else if (mode_ == "linear") {
                    // Bilinear interpolation for 4D (N,C,H,W)
#ifndef NDEBUG
                    LUISA_ASSERT(ndim == 4, "Resize: linear mode only supports 4D tensors.");
#endif
                    uint32_t N = x_shape[0], C = x_shape[1];
                    uint32_t iH = x_shape[2], iW = x_shape[3];
                    uint32_t oH = y_shape[2], oW = y_shape[3];
                    float iH_max = (iH > 0) ? static_cast<float>(iH - 1) : 0.0f;
                    float iW_max = (iW > 0) ? static_cast<float>(iW - 1) : 0.0f;

                    for (auto n : dynamic_range(N)) {
                        for (auto c : dynamic_range(C)) {
                            for (auto oh : dynamic_range(oH)) {
                                for (auto ow : dynamic_range(oW)) {
                                    // Map output to input coordinates using coord_transform
                                    auto iy_f = coord_transform(oh, 2);
                                    auto ix_f = coord_transform(ow, 3);

                                    // Use floor to avoid negative coordinate uint underflow
                                    auto iy_floor = floor(iy_f);
                                    auto ix_floor = floor(ix_f);

                                    auto ly = iy_f - iy_floor;
                                    auto lx = ix_f - ix_floor;
                                    auto hy = Var<VT>{VT{1}} - ly;
                                    auto hx = Var<VT>{VT{1}} - lx;

                                    // Clamp coordinates to valid range
                                    auto clamp_h = [&](auto v) {
                                        return max(min(v, Var<VT>{VT(iH_max)}), Var<VT>{VT(0.0f)}).template cast<uint>();
                                    };
                                    auto clamp_w = [&](auto v) {
                                        return max(min(v, Var<VT>{VT(iW_max)}), Var<VT>{VT(0.0f)}).template cast<uint>();
                                    };

                                    auto iy0 = clamp_h(iy_floor);
                                    auto ix0 = clamp_w(ix_floor);
                                    auto iy1 = clamp_h(iy_floor + Var<VT>{VT{1}});
                                    auto ix1 = clamp_w(ix_floor + Var<VT>{VT{1}});

                                    auto base = n * x.strides()[0] + c * x.strides()[1];
                                    auto v00 = read_x(base + iy0 * x.strides()[2] + ix0 * x.strides()[3]);
                                    auto v01 = read_x(base + iy0 * x.strides()[2] + ix1 * x.strides()[3]);
                                    auto v10 = read_x(base + iy1 * x.strides()[2] + ix0 * x.strides()[3]);
                                    auto v11 = read_x(base + iy1 * x.strides()[2] + ix1 * x.strides()[3]);

                                    auto t0 = fma(hx, v00, lx * v01);
                                    auto t1 = fma(hx, v10, lx * v11);
                                    auto result = fma(hy, t0, ly * t1);
                                    write_y(n * y.strides()[0] + c * y.strides()[1] +
                                            oh * y.strides()[2] + ow * y.strides()[3], result);
                                }
                            }
                        }
                    }
                } else if (mode_ == "cubic") {
                    // Bicubic interpolation for 4D (N,C,H,W)
#ifndef NDEBUG
                    LUISA_ASSERT(ndim == 4, "Resize: cubic mode only supports 4D tensors.");
#endif
                    uint32_t N = x_shape[0], C = x_shape[1];
                    uint32_t iH = x_shape[2], iW = x_shape[3];
                    uint32_t oH = y_shape[2], oW = y_shape[3];
                    float iH_max = (iH > 0) ? static_cast<float>(iH - 1) : 0.0f;
                    float iW_max = (iW > 0) ? static_cast<float>(iW - 1) : 0.0f;

                    // Keys cubic kernel with configurable coefficient a
                    // w(t) = (a+2)|t|^3 - (a+3)|t|^2 + 1,       0 <= |t| < 1
                    // w(t) = a|t|^3 - 5a|t|^2 + 8a|t| - 4a,     1 <= |t| < 2
                    auto cubic_weight = [&](auto t) {
                        auto at = abs(t);
                        auto at2 = at * at;
                        auto at3 = at2 * at;
                        auto a = Var<VT>{VT(cubic_coeff_a_)};
                        auto one = Var<VT>{VT(1.0f)};
                        auto w_near = fma(a + Var<VT>{VT(2.0f)}, at3, -(a + Var<VT>{VT(3.0f)}) * at2) + one;
                        auto w_far = fma(a, at3, fma(-Var<VT>{VT(5.0f)} * a, at2, fma(Var<VT>{VT(8.0f)} * a, at, -Var<VT>{VT(4.0f)} * a)));
                        return select(w_far, w_near, at < one);
                    };

                    for (auto n : dynamic_range(N)) {
                        for (auto c : dynamic_range(C)) {
                            for (auto oh : dynamic_range(oH)) {
                                for (auto ow : dynamic_range(oW)) {
                                    auto iy_f = coord_transform(oh, 2);
                                    auto ix_f = coord_transform(ow, 3);

                                    auto iy_floor = floor(iy_f);
                                    auto ix_floor = floor(ix_f);
                                    auto fy = iy_f - iy_floor;
                                    auto fx = ix_f - ix_floor;

                                    auto one = Var<VT>{VT(1.0f)};
                                    auto two = Var<VT>{VT(2.0f)};

                                    // Precompute 4 weights for x and y
                                    auto wx0 = cubic_weight(fx + one);// t = 1+fx (pixel at floor-1)
                                    auto wx1 = cubic_weight(fx);      // t = fx   (pixel at floor)
                                    auto wx2 = cubic_weight(one - fx);// t = 1-fx (pixel at floor+1)
                                    auto wx3 = cubic_weight(two - fx);// t = 2-fx (pixel at floor+2)

                                    auto wy0 = cubic_weight(fy + one);
                                    auto wy1 = cubic_weight(fy);
                                    auto wy2 = cubic_weight(one - fy);
                                    auto wy3 = cubic_weight(two - fy);

                                    // Fetch with clamped coordinates
                                    auto base = n * x.strides()[0] + c * x.strides()[1];
                                    auto result = def(VT{0});
                                    if (is_bb && x.strides()[3] == 1) {
                                        using VecT = typename detail::VecDispatch<VT>::VecT;
                                        auto ix_floor_i = cast<int>(ix_floor);
                                        auto iy_floor_i = cast<int>(iy_floor);
                                        $if (ix_floor_i >= 1 & ix_floor_i + 2 < static_cast<int>(iW)) {
                                            auto compute_py = [&](int dy) {
                                                auto py_i = max(min(iy_floor_i + dy,
                                                                    Var<int>{static_cast<int>(iH) - 1}),
                                                                Var<int>{0});
                                                return cast<uint>(py_i);
                                            };
                                            auto read_row_vec = [&](int dy) {
                                                auto py = compute_py(dy);
                                                auto addr = base + py * x.strides()[2] + cast<uint>(ix_floor_i - 1);
                                                return buf_x->read<VecT>(off_x + addr * static_cast<uint>(sizeof(VT)));
                                            };
                                            auto row0 = read_row_vec(-1);
                                            auto row1 = read_row_vec(0);
                                            auto row2 = read_row_vec(1);
                                            auto row3 = read_row_vec(2);
                                            auto r0 = fma(wx3, row0.w, fma(wx2, row0.z, fma(wx1, row0.y, wx0 * row0.x)));
                                            auto r1 = fma(wx3, row1.w, fma(wx2, row1.z, fma(wx1, row1.y, wx0 * row1.x)));
                                            auto r2 = fma(wx3, row2.w, fma(wx2, row2.z, fma(wx1, row2.y, wx0 * row2.x)));
                                            auto r3 = fma(wx3, row3.w, fma(wx2, row3.z, fma(wx1, row3.y, wx0 * row3.x)));
                                            result = fma(wy3, r3, fma(wy2, r2, fma(wy1, r1, wy0 * r0)));
                                        } $else {
                                            auto fetch_scalar = [&](int dy, int dx) {
                                                auto py = max(min(iy_floor + Var<VT>{VT(static_cast<float>(dy))},
                                                                  Var<VT>{VT(iH_max)}),
                                                              Var<VT>{VT(0.0f)})
                                                          .template cast<uint>();
                                                auto px = max(min(ix_floor + Var<VT>{VT(static_cast<float>(dx))},
                                                                  Var<VT>{VT(iW_max)}),
                                                              Var<VT>{VT(0.0f)})
                                                          .template cast<uint>();
                                                return read_x(base + py * x.strides()[2] + px * x.strides()[3]);
                                            };
                                            auto r0 = fma(wx0, fetch_scalar(-1, -1), fma(wx1, fetch_scalar(-1, 0), fma(wx2, fetch_scalar(-1, 1), wx3 * fetch_scalar(-1, 2))));
                                            auto r1 = fma(wx0, fetch_scalar(0, -1), fma(wx1, fetch_scalar(0, 0), fma(wx2, fetch_scalar(0, 1), wx3 * fetch_scalar(0, 2))));
                                            auto r2 = fma(wx0, fetch_scalar(1, -1), fma(wx1, fetch_scalar(1, 0), fma(wx2, fetch_scalar(1, 1), wx3 * fetch_scalar(1, 2))));
                                            auto r3 = fma(wx0, fetch_scalar(2, -1), fma(wx1, fetch_scalar(2, 0), fma(wx2, fetch_scalar(2, 1), wx3 * fetch_scalar(2, 2))));
                                            result = fma(wy0, r0, fma(wy1, r1, fma(wy2, r2, wy3 * r3)));
                                        };
                                    } else {
                                        auto fetch = [&](int dy, int dx) {
                                            auto py = max(min(iy_floor + Var<VT>{VT(static_cast<float>(dy))},
                                                              Var<VT>{VT(iH_max)}),
                                                          Var<VT>{VT(0.0f)})
                                                      .template cast<uint>();
                                            auto px = max(min(ix_floor + Var<VT>{VT(static_cast<float>(dx))},
                                                              Var<VT>{VT(iW_max)}),
                                                          Var<VT>{VT(0.0f)})
                                                      .template cast<uint>();
                                            return read_x(base + py * x.strides()[2] + px * x.strides()[3]);
                                        };
                                        auto r0 = fma(wx0, fetch(-1, -1), fma(wx1, fetch(-1, 0), fma(wx2, fetch(-1, 1), wx3 * fetch(-1, 2))));
                                        auto r1 = fma(wx0, fetch(0, -1), fma(wx1, fetch(0, 0), fma(wx2, fetch(0, 1), wx3 * fetch(0, 2))));
                                        auto r2 = fma(wx0, fetch(1, -1), fma(wx1, fetch(1, 0), fma(wx2, fetch(1, 1), wx3 * fetch(1, 2))));
                                        auto r3 = fma(wx0, fetch(2, -1), fma(wx1, fetch(2, 0), fma(wx2, fetch(2, 1), wx3 * fetch(2, 2))));
                                        result = fma(wy0, r0, fma(wy1, r1, fma(wy2, r2, wy3 * r3)));
                                    }
                                    write_y(n * y.strides()[0] + c * y.strides()[1] +
                                            oh * y.strides()[2] + ow * y.strides()[3], result);
                                }
                            }
                        }
                    }
                } else {
#ifndef NDEBUG
                    LUISA_ASSERT(false, "Resize: mode '{}' is not supported.", mode_);
#endif
                }
            });
        }
    }
};

REGISTER_TO_DEFAULT_OPSET(Resize) {
    std::string mode = "nearest";
    std::string coord_transform = "half_pixel";
    std::string nearest_mode = "round_prefer_floor";
    if (auto p = node.try_get_attr("mode"))
        mode = p->get<onnx::AttributeType::STRING>();
    if (auto p = node.try_get_attr("coordinate_transform_mode"))
        coord_transform = p->get<onnx::AttributeType::STRING>();
    if (auto p = node.try_get_attr("nearest_mode"))
        nearest_mode = p->get<onnx::AttributeType::STRING>();
    float cubic_coeff_a = -0.75f;
    if (auto p = node.try_get_attr("cubic_coeff_a"))
        cubic_coeff_a = p->get<onnx::AttributeType::FLOAT>();
    return std::make_unique<Resize>(std::move(mode), std::move(coord_transform),
                                    std::move(nearest_mode), cubic_coeff_a);
};

}// namespace lcml::onnx
