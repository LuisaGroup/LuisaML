#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>

namespace lcml::onnx {

// NonMaxSuppression: selects bounding boxes based on score and overlap thresholds.
// ONNX spec: inputs: boxes(num_batches, spatial, 4), scores(num_batches, num_classes, spatial),
//   max_output_boxes_per_class, iou_threshold, score_threshold
// Output: selected_indices (num_selected, 3) where each row = [batch_index, class_index, box_index]
class NonMaxSuppression : public Operator {
private:
    int32_t center_point_box_;

public:
    NonMaxSuppression(int32_t center_point_box) : Operator("NonMaxSuppression"), center_point_box_(center_point_box) {}

    void forward(luisa::span<std::reference_wrapper<ITensor>> inputs,
                 luisa::span<std::reference_wrapper<ITensor>> outputs) override {
        LUISA_ASSERT(inputs.size() >= 2 && outputs.size() == 1,
                     "NonMaxSuppression requires >=2 inputs and 1 output.");

        // Input 0: boxes - shape (num_batches, spatial_dim, 4)
        // Input 1: scores - shape (num_batches, num_classes, spatial_dim)
        // Input 2 (optional): max_output_boxes_per_class - scalar int64
        // Input 3 (optional): iou_threshold - scalar float
        // Input 4 (optional): score_threshold - scalar float
        auto &boxes_t = inputs[0].get();
        auto &scores_t = inputs[1].get();
        auto &Y = outputs[0].get();

        auto const &boxes_shape = boxes_t.shape();  // (N, S, 4)
        auto const &scores_shape = scores_t.shape();// (N, C, S)
        auto const &y_shape = Y.shape();            // (num_selected, 3)

        uint32_t num_batches = boxes_shape[0];
        uint32_t spatial_dim = boxes_shape[1];
        uint32_t num_classes = scores_shape[1];
        uint32_t max_output = y_shape[0];// total output slots

        // Compute max_output_boxes_per_class from output shape / (num_batches * num_classes)
        uint32_t max_per_class_host = (num_batches > 0 && num_classes > 0)
                                          ? max_output / (num_batches * num_classes)
                                          : max_output;
        if (max_per_class_host == 0) max_per_class_host = max_output;
        if (max_per_class_host > spatial_dim) max_per_class_host = spatial_dim;

        auto &y = static_cast<NNTensor<int> &>(Y);

#ifndef NDEBUG
        LUISA_ASSERT(boxes_t.element_type_index() == scores_t.element_type_index(),
                     "NonMaxSuppression: boxes and scores must have the same element type.");
#endif

        visit_type_index<NNTypeList>(boxes_t.element_type_index(), [&]<typename T>() {
            using VT = nn_storage_type_t<T>;
            using CT = std::conditional_t<
                std::is_same_v<T, FP4E2M1> || std::is_same_v<T, FP8E4M3FN> ||
                    std::is_same_v<T, FP8E5M2> || std::is_same_v<T, FP16Quantized>,
                half, VT>;

            if constexpr (IsFloatingPoint<T>::value || std::is_same_v<T, FP4E2M1> ||
                          std::is_same_v<T, FP8E4M3FN> || std::is_same_v<T, FP8E5M2> ||
                          std::is_same_v<T, FP16Quantized>) {
                auto &boxes = static_cast<NNTensor<T> &>(boxes_t);
                auto &scores = static_cast<NNTensor<T> &>(scores_t);

                auto dequant = [&](auto &v) -> Var<CT> {
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
                    } else {
                        return cast<CT>(v);
                    }
                };

                auto half_ct = Var<CT>{CT(0.5f)};
                auto zero_ct = Var<CT>{CT(0.0f)};
                auto neg_inf_ct = def(std::is_same_v<CT, half> ? CT(-65504.0f) : CT(-1e30f));
                auto eps_ct = Var<CT>{CT(std::is_same_v<CT, half> ? 1e-4f : 1e-10f)};

                // Read optional iou_threshold (default 0)
                auto iou_thresh = def(cast<CT>(0.0f));
                if (inputs.size() >= 4 && inputs[3].get().size() > 0) {
                    auto &iou_t = static_cast<NNTensor<float> &>(inputs[3].get());
                    iou_thresh = cast<CT>(iou_t[0u]);
                }

                // Read optional score_threshold (default -inf)
                auto score_thresh = neg_inf_ct;
                if (inputs.size() >= 5 && inputs[4].get().size() > 0) {
                    auto &st = static_cast<NNTensor<float> &>(inputs[4].get());
                    score_thresh = cast<CT>(st[0u]);
                }

                // Read optional max_output_boxes_per_class from input[2]
                auto max_per_class = def(static_cast<uint>(max_per_class_host));
                if (inputs.size() >= 3 && inputs[2].get().size() > 0) {
                    auto &mob = static_cast<NNTensor<int> &>(inputs[2].get());
                    auto mob_val = mob[0u].cast<uint>();
                    max_per_class = min(max_per_class, mob_val);
                }

                // Global write pointer into output tensor
                auto out_ptr = def(0u);

                // Initialize output to zeros
                for (auto i : dynamic_range(y.size())) {
                    y[i] = def(0);
                }

                // For each batch and class, perform greedy NMS
                for (auto batch : dynamic_range(num_batches)) {
                    for (auto cls : dynamic_range(num_classes)) {
                        auto score_base = batch * scores.strides()[0] + cls * scores.strides()[1];
                        auto selected_count = def(0u);

                        for (auto rank : dynamic_range(spatial_dim)) {
                            $if (selected_count >= max_per_class) {
                                $break;
                            };

                            auto best_box = def(0u);
                            for (auto candidate : dynamic_range(spatial_dim)) {
                                auto cand_score = dequant(scores[score_base + candidate]);
                                auto count = def(0u);
                                for (auto other : dynamic_range(spatial_dim)) {
                                    auto other_score = dequant(scores[score_base + other]);
                                    $if ((other_score > cand_score) | (other_score == cand_score & other < candidate)) {
                                        count += 1u;
                                    };
                                }
                                $if (count == rank) {
                                    best_box = candidate;
                                };
                            }

                            auto box_score = dequant(scores[score_base + best_box]);

                            $if (box_score > score_thresh) {
                                auto box_base = batch * boxes.strides()[0] + best_box * boxes.strides()[1];
                                Var<CT> c0, c1, c2, c3;
                                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
                                    if (boxes.container().is_byte_buffer() && boxes.strides()[2] == 1) {
                                        auto buf_boxes = boxes.container().get_byte_buffer();
                                        auto off_boxes = static_cast<uint>(boxes.container().get_byte_offset());
                                        using VecT = typename detail::VecDispatch<CT>::VecT;
                                        auto v = buf_boxes->read<VecT>(off_boxes + box_base * static_cast<uint>(sizeof(CT)));
                                        c0 = v.x;
                                        c1 = v.y;
                                        c2 = v.z;
                                        c3 = v.w;
                                    } else {
                                        c0 = dequant(boxes[box_base + 0u]);
                                        c1 = dequant(boxes[box_base + 1u]);
                                        c2 = dequant(boxes[box_base + 2u]);
                                        c3 = dequant(boxes[box_base + 3u]);
                                    }
                                } else {
                                    c0 = dequant(boxes[box_base + 0u]);
                                    c1 = dequant(boxes[box_base + 1u]);
                                    c2 = dequant(boxes[box_base + 2u]);
                                    c3 = dequant(boxes[box_base + 3u]);
                                }

                                Var<CT> by1, bx1, by2, bx2;
                                if (center_point_box_ == 1) {
                                    auto cx = c0, cy = c1, w = c2, h = c3;
                                    by1 = luisa::compute::fma(h, -half_ct, cy);
                                    bx1 = luisa::compute::fma(w, -half_ct, cx);
                                    by2 = luisa::compute::fma(h, half_ct, cy);
                                    bx2 = luisa::compute::fma(w, half_ct, cx);
                                } else {
                                    by1 = c0;
                                    bx1 = c1;
                                    by2 = c2;
                                    bx2 = c3;
                                }

                                auto y1 = min(by1, by2);
                                auto x1 = min(bx1, bx2);
                                auto y2 = max(by1, by2);
                                auto x2 = max(bx1, bx2);
                                auto area_cur = (y2 - y1) * (x2 - x1);

                                auto suppressed = def(false);

                                for (auto prev : dynamic_range(spatial_dim)) {
                                    $if (prev >= selected_count) {
                                        $break;
                                    };
                                    auto prev_out_row = out_ptr - selected_count + prev;
                                    auto prev_box_idx = y[prev_out_row * 3u + 2u].cast<uint>();

                                    auto prev_base = batch * boxes.strides()[0] + prev_box_idx * boxes.strides()[1];
                                    Var<CT> pc0, pc1, pc2, pc3;
                                    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
                                        if (boxes.container().is_byte_buffer() && boxes.strides()[2] == 1) {
                                            auto buf_boxes = boxes.container().get_byte_buffer();
                                            auto off_boxes = static_cast<uint>(boxes.container().get_byte_offset());
                                            using VecT = typename detail::VecDispatch<CT>::VecT;
                                            auto v = buf_boxes->read<VecT>(off_boxes + prev_base * static_cast<uint>(sizeof(CT)));
                                            pc0 = v.x;
                                            pc1 = v.y;
                                            pc2 = v.z;
                                            pc3 = v.w;
                                        } else {
                                            pc0 = dequant(boxes[prev_base + 0u]);
                                            pc1 = dequant(boxes[prev_base + 1u]);
                                            pc2 = dequant(boxes[prev_base + 2u]);
                                            pc3 = dequant(boxes[prev_base + 3u]);
                                        }
                                    } else {
                                        pc0 = dequant(boxes[prev_base + 0u]);
                                        pc1 = dequant(boxes[prev_base + 1u]);
                                        pc2 = dequant(boxes[prev_base + 2u]);
                                        pc3 = dequant(boxes[prev_base + 3u]);
                                    }

                                    Var<CT> py1, px1, py2, px2;
                                    if (center_point_box_ == 1) {
                                        auto cx = pc0, cy = pc1, w = pc2, h = pc3;
                                        py1 = luisa::compute::fma(h, -half_ct, cy);
                                        px1 = luisa::compute::fma(w, -half_ct, cx);
                                        py2 = luisa::compute::fma(h, half_ct, cy);
                                        px2 = luisa::compute::fma(w, half_ct, cx);
                                    } else {
                                        py1 = pc0;
                                        px1 = pc1;
                                        py2 = pc2;
                                        px2 = pc3;
                                    }

                                    // Fix coordinate normalization: preserve original values before min/max
                                    auto min_py = min(py1, py2);
                                    auto min_px = min(px1, px2);
                                    auto max_py = max(py1, py2);
                                    auto max_px = max(px1, px2);
                                    py1 = min_py;
                                    px1 = min_px;
                                    py2 = max_py;
                                    px2 = max_px;

                                    auto area_prev = (py2 - py1) * (px2 - px1);

                                    auto inter_y1 = max(y1, py1);
                                    auto inter_x1 = max(x1, px1);
                                    auto inter_y2 = min(y2, py2);
                                    auto inter_x2 = min(x2, px2);
                                    auto inter_area = max(inter_y2 - inter_y1, zero_ct) * max(inter_x2 - inter_x1, zero_ct);
                                    auto union_area = area_cur + area_prev - inter_area;
                                    auto iou = inter_area / max(union_area, eps_ct);

                                    $if (iou > iou_thresh) {
                                        suppressed = true;
                                    };
                                }

                                $if (!suppressed) {
                                    $if (out_ptr < max_output) {
                                        y[out_ptr * 3u + 0u] = batch.cast<int>();
                                        y[out_ptr * 3u + 1u] = cls.cast<int>();
                                        y[out_ptr * 3u + 2u] = best_box.cast<int>();
                                        out_ptr += 1u;
                                        selected_count += 1u;
                                    };
                                };
                            };
                        }
                    }
                }
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(NonMaxSuppression) {
    int32_t center_point_box = 0;
    if (auto p = node.try_get_attr("center_point_box"))
        center_point_box = p->get<onnx::AttributeType::INT>();
    return luisa::make_unique<NonMaxSuppression>(center_point_box);
};

// RoiAlign: Region of Interest Align pooling.
// ONNX spec: inputs: X(N,C,H,W), rois(num_rois, 4), batch_indices(num_rois)
// Attributes: mode ("avg"/"max"), output_height, output_width, sampling_ratio, spatial_scale
class RoiAlign : public Operator {
private:
    luisa::string mode_;
    int64_t output_height_;
    int64_t output_width_;
    int64_t sampling_ratio_;
    float spatial_scale_;

public:
    RoiAlign(luisa::string mode, int64_t oh, int64_t ow, int64_t sr, float ss)
        : Operator("RoiAlign"), mode_(std::move(mode)),
          output_height_(oh), output_width_(ow),
          sampling_ratio_(sr), spatial_scale_(ss) {}

    void forward(luisa::span<std::reference_wrapper<ITensor>> inputs,
                 luisa::span<std::reference_wrapper<ITensor>> outputs) override {
        LUISA_ASSERT(inputs.size() == 3 && outputs.size() == 1,
                     "RoiAlign requires 3 inputs and 1 output.");
        auto &X = inputs[0].get();
        auto &Y = outputs[0].get();

        auto const &x_shape = X.shape();
        uint32_t C = x_shape[1], iH = x_shape[2], iW = x_shape[3];
        uint32_t oH = static_cast<uint32_t>(output_height_);
        uint32_t oW = static_cast<uint32_t>(output_width_);

        LUISA_ASSERT(X.element_type_index() == inputs[1].get().element_type_index() && X.element_type_index() == Y.element_type_index(),
                     "RoiAlign: X, rois and Y must have the same element type.");

        // Improved adaptive sampling ratio: use feature-map scale estimate when sampling_ratio == 0
        uint32_t sr_h, sr_w;
        if (sampling_ratio_ > 0) {
            sr_h = static_cast<uint32_t>(sampling_ratio_);
            sr_w = static_cast<uint32_t>(sampling_ratio_);
        } else {
            auto ceil_positive = [](float v) -> uint32_t {
                uint32_t iv = static_cast<uint32_t>(v);
                return (v > static_cast<float>(iv)) ? iv + 1u : iv;
            };
            float est_h = static_cast<float>(iH) * spatial_scale_ / static_cast<float>(oH);
            float est_w = static_cast<float>(iW) * spatial_scale_ / static_cast<float>(oW);
            sr_h = std::max(1u, ceil_positive(est_h));
            sr_w = std::max(1u, ceil_positive(est_w));
        }

        visit_type_index<NNTypeList>(X.element_type_index(), [&]<typename T>() {
            using VT = nn_storage_type_t<T>;
            using CT = std::conditional_t<
                std::is_same_v<T, FP4E2M1> || std::is_same_v<T, FP8E4M3FN> ||
                    std::is_same_v<T, FP8E5M2> || std::is_same_v<T, FP16Quantized>,
                half, VT>;

            if constexpr (IsFloatingPoint<T>::value || std::is_same_v<T, FP4E2M1> ||
                          std::is_same_v<T, FP8E4M3FN> || std::is_same_v<T, FP8E5M2> ||
                          std::is_same_v<T, FP16Quantized>) {
                auto &x = static_cast<NNTensor<T> &>(X);
                auto &rois = static_cast<NNTensor<T> &>(inputs[1].get());
                auto &batch_idx = static_cast<NNTensor<int> &>(inputs[2].get());
                auto &y = static_cast<NNTensor<T> &>(Y);

                auto dequant = [&](auto &v) -> Var<CT> {
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
                    } else {
                        return cast<CT>(v);
                    }
                };

                auto store_y = [&](auto idx, auto val) {
                    if constexpr (std::is_same_v<T, FP4E2M1>) {
                        static auto c = fp4e2m1_from_float();
                        y[idx].bits = c(val.template cast<half>()).cast<uint16_t>();
                    } else if constexpr (std::is_same_v<T, FP8E4M3FN>) {
                        static auto c = fp8e4m3_from_float();
                        y[idx].bits = c(val.template cast<half>()).cast<uint16_t>();
                    } else if constexpr (std::is_same_v<T, FP8E5M2>) {
                        static auto c = fp8e5m2_from_float();
                        y[idx].bits = c(val.template cast<half>()).cast<uint16_t>();
                    } else if constexpr (std::is_same_v<T, FP16Quantized>) {
                        y[idx].bits = val.template cast<half>();
                    } else {
                        y[idx] = val.template cast<VT>();
                    }
                };

                auto num_rois = rois.shape()[0];
                auto scale = Var<CT>{CT(spatial_scale_)};
                auto zero = Var<CT>{CT(0.0f)};
                auto one = Var<CT>{CT(1.0f)};

                auto bilinear_sample = [&](auto &x_ref, auto base, auto cy, auto cx) {
                    auto iy_floor = floor(cy);
                    auto ix_floor = floor(cx);
                    auto ly = cy - iy_floor;
                    auto lx = cx - ix_floor;

                    auto clamp_h = [&](auto v) {
                        return max(min(v, Var<CT>{CT(static_cast<float>(iH - 1))}), zero).template cast<uint>();
                    };
                    auto clamp_w = [&](auto v) {
                        return max(min(v, Var<CT>{CT(static_cast<float>(iW - 1))}), zero).template cast<uint>();
                    };

                    auto iy0 = clamp_h(iy_floor);
                    auto ix0 = clamp_w(ix_floor);
                    auto iy1 = clamp_h(iy_floor + one);
                    auto ix1 = clamp_w(ix_floor + one);

                    Var<CT> v00, v01, v10, v11;
                    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
                        if (x_ref.container().is_byte_buffer() && x_ref.strides()[3] == 1) {
                            auto buf_x = x_ref.container().get_byte_buffer();
                            auto off_x = static_cast<uint>(x_ref.container().get_byte_offset());
                            using Vec2T = std::conditional_t<std::is_same_v<CT, float>, float2, half2>;
                            auto off_00 = off_x + (base + iy0 * x_ref.strides()[2] + ix0 * x_ref.strides()[3]) * static_cast<uint>(sizeof(CT));
                            auto off_10 = off_x + (base + iy1 * x_ref.strides()[2] + ix0 * x_ref.strides()[3]) * static_cast<uint>(sizeof(CT));
                            auto v0x = buf_x->read<Vec2T>(off_00);
                            auto v1x = buf_x->read<Vec2T>(off_10);
                            v00 = v0x.x;
                            v01 = v0x.y;
                            v10 = v1x.x;
                            v11 = v1x.y;
                        } else {
                            v00 = dequant(x_ref[base + iy0 * x_ref.strides()[2] + ix0 * x_ref.strides()[3]]);
                            v01 = dequant(x_ref[base + iy0 * x_ref.strides()[2] + ix1 * x_ref.strides()[3]]);
                            v10 = dequant(x_ref[base + iy1 * x_ref.strides()[2] + ix0 * x_ref.strides()[3]]);
                            v11 = dequant(x_ref[base + iy1 * x_ref.strides()[2] + ix1 * x_ref.strides()[3]]);
                        }
                    } else {
                        v00 = dequant(x_ref[base + iy0 * x_ref.strides()[2] + ix0 * x_ref.strides()[3]]);
                        v01 = dequant(x_ref[base + iy0 * x_ref.strides()[2] + ix1 * x_ref.strides()[3]]);
                        v10 = dequant(x_ref[base + iy1 * x_ref.strides()[2] + ix0 * x_ref.strides()[3]]);
                        v11 = dequant(x_ref[base + iy1 * x_ref.strides()[2] + ix1 * x_ref.strides()[3]]);
                    }

                    // FMA-optimized bilinear interpolation
                    auto t0 = luisa::compute::fma(lx, v01 - v00, v00);
                    auto t1 = luisa::compute::fma(lx, v11 - v10, v10);
                    return luisa::compute::fma(ly, t1 - t0, t0);
                };

                auto inv_count = Var<CT>{CT(1.0f / static_cast<float>(sr_h * sr_w))};
                auto neg_inf_ct = def(std::is_same_v<CT, half> ? CT(-65504.0f) : CT(-1e30f));

                for (auto roi_idx : dynamic_range(num_rois)) {
                    auto n = batch_idx[roi_idx].cast<uint>();

                    Var<CT> r0, r1, r2, r3;
                    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
                        if (rois.container().is_byte_buffer() && rois.strides()[1] == 1) {
                            auto buf_rois = rois.container().get_byte_buffer();
                            auto off_rois = static_cast<uint>(rois.container().get_byte_offset());
                            using VecT = typename detail::VecDispatch<CT>::VecT;
                            auto v = buf_rois->read<VecT>(off_rois + (roi_idx * 4u) * static_cast<uint>(sizeof(CT)));
                            r0 = v.x;
                            r1 = v.y;
                            r2 = v.z;
                            r3 = v.w;
                        } else {
                            r0 = dequant(rois[roi_idx * 4u + 0u]);
                            r1 = dequant(rois[roi_idx * 4u + 1u]);
                            r2 = dequant(rois[roi_idx * 4u + 2u]);
                            r3 = dequant(rois[roi_idx * 4u + 3u]);
                        }
                    } else {
                        r0 = dequant(rois[roi_idx * 4u + 0u]);
                        r1 = dequant(rois[roi_idx * 4u + 1u]);
                        r2 = dequant(rois[roi_idx * 4u + 2u]);
                        r3 = dequant(rois[roi_idx * 4u + 3u]);
                    }

                    auto x1 = r0 * scale;
                    auto y1 = r1 * scale;
                    auto x2 = r2 * scale;
                    auto y2 = r3 * scale;

                    auto roi_h = max(y2 - y1, one);
                    auto roi_w = max(x2 - x1, one);
                    auto bin_h = roi_h / Var<CT>{CT(static_cast<float>(oH))};
                    auto bin_w = roi_w / Var<CT>{CT(static_cast<float>(oW))};

                    for (auto c : dynamic_range(C)) {
                        auto base = n * x.strides()[0] + c * x.strides()[1];
                        for (auto oh : dynamic_range(oH)) {
                            for (auto ow : dynamic_range(oW)) {
                                if (mode_ == "max") {
                                    auto max_val = def(neg_inf_ct);
                                    for (uint32_t iy = 0; iy < sr_h; ++iy) {
                                        for (uint32_t ix = 0; ix < sr_w; ++ix) {
                                            auto sub_y = y1 + bin_h * (oh.cast<CT>() +
                                                                       Var<CT>{CT((static_cast<float>(iy) + 0.5f) / static_cast<float>(sr_h))});
                                            auto sub_x = x1 + bin_w * (ow.cast<CT>() +
                                                                       Var<CT>{CT((static_cast<float>(ix) + 0.5f) / static_cast<float>(sr_w))});
                                            auto val = bilinear_sample(x, base, sub_y, sub_x);
                                            max_val = max(max_val, val);
                                        }
                                    }
                                    store_y(roi_idx * y.strides()[0] + c * y.strides()[1] +
                                            oh * y.strides()[2] + ow * y.strides()[3], max_val);
                                } else {
                                    auto sum = def(CT{0});
                                    for (uint32_t iy = 0; iy < sr_h; ++iy) {
                                        for (uint32_t ix = 0; ix < sr_w; ++ix) {
                                            auto sub_y = y1 + bin_h * (oh.cast<CT>() +
                                                                       Var<CT>{CT((static_cast<float>(iy) + 0.5f) / static_cast<float>(sr_h))});
                                            auto sub_x = x1 + bin_w * (ow.cast<CT>() +
                                                                       Var<CT>{CT((static_cast<float>(ix) + 0.5f) / static_cast<float>(sr_w))});
                                            sum += bilinear_sample(x, base, sub_y, sub_x);
                                        }
                                    }
                                    store_y(roi_idx * y.strides()[0] + c * y.strides()[1] +
                                            oh * y.strides()[2] + ow * y.strides()[3], sum * inv_count);
                                }
                            }
                        }
                    }
                }
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(RoiAlign) {
    luisa::string mode = "avg";
    int64_t oh = 1, ow = 1, sr = 0;
    float ss = 1.0f;
    if (auto p = node.try_get_attr("mode")) mode = p->get<onnx::AttributeType::STRING>();
    if (auto p = node.try_get_attr("output_height")) oh = p->get<onnx::AttributeType::INT>();
    if (auto p = node.try_get_attr("output_width")) ow = p->get<onnx::AttributeType::INT>();
    if (auto p = node.try_get_attr("sampling_ratio")) sr = p->get<onnx::AttributeType::INT>();
    if (auto p = node.try_get_attr("spatial_scale")) ss = p->get<onnx::AttributeType::FLOAT>();
    return luisa::make_unique<RoiAlign>(std::move(mode), oh, ow, sr, ss);
};

}// namespace lcml::onnx
