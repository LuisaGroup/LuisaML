#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

namespace lcml::onnx {

// Slice: produces a slice of the input tensor along multiple axes.
// ONNX spec: inputs: data, starts, ends, [axes], [steps]
class Slice : public Operator {
public:
    Slice() : Operator("Slice") {}

    /// Aggressive compile-time view detection for Slice.
    /// Replicates the fast-path logic: if all slice params are constant,
    /// single-axis, step=1, we can determine at graph-build time whether
    /// the result will be a contiguous view.
    bool is_output_view([[maybe_unused]] size_t output_index,
                        [[maybe_unused]] onnx::Node const &node) const override {
        auto const &inputs = node.get_inputs();
        if (inputs.size() < 3) return false;

        auto const &data_var = inputs[0].get();
        auto const &starts_var = inputs[1].get();
        auto const &ends_var = inputs[2].get();
        auto const &data_shape = data_var.get_shape();
        auto ndim = data_shape.size();
        if (ndim == 0) return false;

        // All slice parameters must be compile-time constants (have raw_data)
        if (!starts_var.is_constant() || starts_var.get_raw_data().empty()) return false;
        if (!ends_var.is_constant() || ends_var.get_raw_data().empty()) return false;
        bool has_axes = (inputs.size() >= 4 && !inputs[3].get().get_raw_data().empty());
        bool has_steps = (inputs.size() >= 5 && !inputs[4].get().get_raw_data().empty());
        if (inputs.size() >= 4 && inputs[3].get().is_constant() && inputs[3].get().get_raw_data().empty()) {
            // axes input exists but has no data — cannot analyze
            if (!inputs[3].get().get_shape().empty() && inputs[3].get().get_shape()[0] > 0)
                return false;
        }
        if (inputs.size() >= 5 && inputs[4].get().is_constant() && inputs[4].get().get_raw_data().empty()) {
            if (!inputs[4].get().get_shape().empty() && inputs[4].get().get_shape()[0] > 0)
                return false;
        }

        // Parse constant int values from raw_data
        auto parse_ints = [](onnx::Variable const &var) -> std::vector<int> {
            auto const &raw = var.get_raw_data();
            size_t n = 1;
            for (auto d : var.get_shape()) n *= d;
            std::vector<int> result(n);
            if (var.get_dtype() == onnx::DataType::INT64) {
                auto const *src = reinterpret_cast<int64_t const *>(raw.data());
                for (size_t i = 0; i < n; ++i) result[i] = static_cast<int>(src[i]);
            } else {
                auto const *src = reinterpret_cast<int32_t const *>(raw.data());
                for (size_t i = 0; i < n; ++i) result[i] = src[i];
            }
            return result;
        };

        auto starts = parse_ints(starts_var);
        auto ends = parse_ints(ends_var);
        size_t num_slices = starts.size();

        std::vector<int> axes_vec, steps_vec;
        if (has_axes) axes_vec = parse_ints(inputs[3].get());
        if (has_steps) {
            steps_vec = parse_ints(inputs[4].get());
            // All steps must be 1
            for (auto s : steps_vec) {
                if (s != 1) return false;
            }
        }

        // Find the single sliced axis (non-full-range axis)
        int sole_axis = -1;
        int sole_start = 0;
        for (size_t s = 0; s < num_slices; ++s) {
            int ax = has_axes ? axes_vec[s] : static_cast<int>(s);
            if (ax < 0) ax += static_cast<int>(ndim);
            if (ax < 0 || ax >= static_cast<int>(ndim)) return false;

            int sv = starts[s];
            int ev = ends[s];
            int dim_size = static_cast<int>(data_shape[ax]);
            if (sv < 0) sv += dim_size;
            if (ev < 0) ev += dim_size;
            sv = std::max(0, std::min(sv, dim_size));
            ev = std::max(0, std::min(ev, dim_size));

            // If this axis covers the full range, it's a no-op → skip
            if (sv == 0 && ev >= dim_size) continue;

            // More than one axis is being sliced → not a simple view
            if (sole_axis >= 0) return false;

            sole_axis = ax;
            sole_start = sv;
        }

        // No axis actually sliced → full pass-through view
        if (sole_axis < 0) return true;

        // View only if the sliced region is contiguous: axis must be 0
        // (or equivalently, outer_count == 1, meaning all dims before axis are 1)
        size_t outer_count = 1;
        for (int d = 0; d < sole_axis; ++d)
            outer_count *= data_shape[d];
        return outer_count == 1;
    }

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() >= 3 && inputs.size() <= 5 && outputs.size() == 1,
                     "Slice requires 3-5 inputs and 1 output.");
        auto &data = inputs[0].get();
        auto &output = outputs[0].get();
        auto ndim = data.ndim();

        LUISA_ASSERT(data.element_type() == output.element_type(),
                     "Slice: data and output must have the same element type.");
        LUISA_ASSERT(inputs[1].get().element_type() == typeid(int) || inputs[1].get().element_type() == typeid(slong),
                     "Slice: starts must be int or int64 type.");
        LUISA_ASSERT(inputs[2].get().element_type() == typeid(int) || inputs[2].get().element_type() == typeid(slong),
                     "Slice: ends must be int or int64 type.");
#else
        auto &data = inputs[0].get();
        auto &output = outputs[0].get();
        auto ndim = data.ndim();
#endif

        // Read starts, ends from input tensors (int type)
        auto &starts_t = static_cast<NNTensor<int> &>(inputs[1].get());
        auto &ends_t = static_cast<NNTensor<int> &>(inputs[2].get());
        auto num_slices = starts_t.size();

        // Default axes = [0, 1, ..., num_slices-1], default steps = [1, ...]
        bool has_axes = (inputs.size() >= 4 && inputs[3].get().size() > 0);
        bool has_steps = (inputs.size() >= 5 && inputs[4].get().size() > 0);

        // Read starts, axes, steps from input tensors and build per-dimension
        // start/step arrays. The output shape is already computed by the runtime.
        // We need to map each output coordinate to the correct input position:
        //   input_coord[d] = starts[d] + output_coord[d] * steps[d]

        // Build per-dimension start and step using DSL values from input tensors
        // For dimensions not in the axes list, start=0 and step=1
        auto &axes_t = has_axes ? static_cast<NNTensor<int> &>(inputs[3].get()) : starts_t;
        auto &steps_t = has_steps ? static_cast<NNTensor<int> &>(inputs[4].get()) : starts_t;

        visit_typeid<NNTypeList>(data.element_type(), [&]<typename T>() {
            auto &in = static_cast<NNTensor<T> &>(data);
            auto &out = static_cast<NNTensor<T> &>(output);
            using ST = nn_storage_type_t<T>;

            // Fast path: all slice params are constant, single axis, step=1
            // Depending on the axis position we either use a zero-copy View
            // (when the sliced region is contiguous in memory) or a block-copy
            // that avoids the expensive per-element for_each_dim decomposition.
            auto try_const_fast_path = [&]() -> bool {
                // All slice parameters must be compile-time constants
                if (!inputs[1].get().is_constant() || !inputs[2].get().is_constant())
                    return false;
                if (has_steps && !inputs[4].get().is_constant())
                    return false;
                if (has_axes && !inputs[3].get().is_constant())
                    return false;

                auto &starts_const = static_cast<NNConstTensor<int> const &>(inputs[1].get());
                auto &ends_const = static_cast<NNConstTensor<int> const &>(inputs[2].get());
                auto num_s = starts_const.const_data().size();

                // Verify: only ONE axis is sliced, step must be 1
                int sole_axis = -1;
                int sole_start = 0;
                for (size_t s = 0; s < num_s; ++s) {
                    int ax = has_axes ?
                                 static_cast<NNConstTensor<int> const &>(inputs[3].get()).const_data()[s] :
                                 static_cast<int>(s);
                    if (ax < 0) ax += static_cast<int>(ndim);

                    if (has_steps) {
                        int step = static_cast<NNConstTensor<int> const &>(inputs[4].get()).const_data()[s];
                        if (step != 1) return false;
                    }

                    int sv = starts_const.const_data()[s];
                    int ev = ends_const.const_data()[s];
                    if (sv < 0) sv += static_cast<int>(in.shape()[ax]);
                    if (ev < 0) ev += static_cast<int>(in.shape()[ax]);
                    // Clamp
                    sv = std::max(0, std::min(sv, static_cast<int>(in.shape()[ax])));
                    ev = std::max(0, std::min(ev, static_cast<int>(in.shape()[ax])));

                    // If this axis slice covers the full dimension, skip it (no-op)
                    if (sv == 0 && ev >= static_cast<int>(in.shape()[ax]))
                        continue;

                    // More than one axis is being sliced -> fall back
                    if (sole_axis >= 0) return false;

                    sole_axis = ax;
                    sole_start = sv;
                }

                // No axis actually sliced (full pass-through) -> View of entire input
                if (sole_axis < 0) {
                    // Full pass-through: explicit element-wise copy
                    if (!in.container().shares_storage_with(out.container())) {
                        if constexpr (detail::VecDispatch<ST>::supported) {
                            if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                                detail::vectorized_unary<ST>(in, out, [](auto v) { return v; });
                            } else {
                                for (auto i : dynamic_range(static_cast<uint32_t>(out.size()))) {
                                    out[i] = in[i];
                                }
                            }
                        } else {
                            for (auto i : dynamic_range(static_cast<uint32_t>(out.size()))) {
                                out[i] = in[i];
                            }
                        }
                    }
                    return true;
                }

                auto axis_u = static_cast<uint32_t>(sole_axis);
                auto inner_size = in.strides()[axis_u];// elements per axis coordinate
                auto slice_len = out.shape()[axis_u];
                auto in_axis_size = in.shape()[axis_u];
                auto output_block_size = static_cast<size_t>(slice_len) * inner_size;
                auto input_block_stride = static_cast<size_t>(in_axis_size) * inner_size;
                auto start_offset = static_cast<size_t>(sole_start) * inner_size;

                // Compute outer_count = product of all dimensions before the sliced axis
                size_t outer_count = 1;
                for (uint32_t d = 0; d < axis_u; ++d)
                    outer_count *= in.shape()[d];

                if (outer_count == 1) {
                    // Contiguous slice: explicit element-wise copy.
                    // If already sharing storage (inplace allocated), skip.
                    if (!in.container().shares_storage_with(out.container())) {
                        auto start_off = static_cast<uint32_t>(start_offset);
                        if constexpr (detail::VecDispatch<ST>::supported) {
                            if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                                using VecT = typename detail::VecDispatch<ST>::VecT;
                                auto buf_in = in.container().get_byte_buffer();
                                auto buf_o = out.container().get_byte_buffer();
                                auto off_in = static_cast<uint>(in.container().get_byte_offset());
                                auto off_o = static_cast<uint>(out.container().get_byte_offset());
                                auto n = static_cast<uint>(out.size());
                                auto vec_n = n / detail::VecDispatch<ST>::N;
                                auto rem = n % detail::VecDispatch<ST>::N;
                                auto byte_start_off = start_off * static_cast<uint>(sizeof(ST));
                                for (auto i : dynamic_range(vec_n)) {
                                    auto byte_idx = i * static_cast<uint>(sizeof(VecT));
                                    auto v = buf_in->read<VecT>(off_in + byte_start_off + byte_idx);
                                    buf_o->write(off_o + byte_idx, v);
                                }
                                for (auto i : dynamic_range(rem)) {
                                    auto idx = vec_n * detail::VecDispatch<ST>::N + i;
                                    auto byte_idx = idx * static_cast<uint>(sizeof(ST));
                                    auto v = buf_in->read<ST>(off_in + byte_start_off + byte_idx);
                                    buf_o->write(off_o + byte_idx, v);
                                }
                            } else {
                                for (auto i : dynamic_range(static_cast<uint32_t>(out.size()))) {
                                    out[i] = in[i + start_off];
                                }
                            }
                        } else {
                            for (auto i : dynamic_range(static_cast<uint32_t>(out.size()))) {
                                out[i] = in[i + start_off];
                            }
                        }
                    }
                } else {
                    // axis != 0 with multiple outer positions: block copy
                    // Each outer position has a contiguous block of output_block_size elements
                    if constexpr (detail::VecDispatch<ST>::supported) {
                        if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                            using VecT = typename detail::VecDispatch<ST>::VecT;
                            auto buf_in = in.container().get_byte_buffer();
                            auto buf_o = out.container().get_byte_buffer();
                            auto off_in = static_cast<uint>(in.container().get_byte_offset());
                            auto off_o = static_cast<uint>(out.container().get_byte_offset());
                            auto n = static_cast<uint>(output_block_size);
                            auto vec_n = n / detail::VecDispatch<ST>::N;
                            auto rem = n % detail::VecDispatch<ST>::N;
                            auto elem_size = static_cast<uint>(sizeof(ST));
                            for (size_t o = 0; o < outer_count; ++o) {
                                auto in_base = static_cast<uint32_t>(o * input_block_stride + start_offset);
                                auto out_base = static_cast<uint32_t>(o * output_block_size);
                                for (auto i : dynamic_range(vec_n)) {
                                    auto byte_idx = i * static_cast<uint>(sizeof(VecT));
                                    auto v = buf_in->read<VecT>(off_in + in_base * elem_size + byte_idx);
                                    buf_o->write(off_o + out_base * elem_size + byte_idx, v);
                                }
                                for (auto i : dynamic_range(rem)) {
                                    auto idx = vec_n * detail::VecDispatch<ST>::N + i;
                                    auto byte_idx = idx * elem_size;
                                    auto v = buf_in->read<ST>(off_in + in_base * elem_size + byte_idx);
                                    buf_o->write(off_o + out_base * elem_size + byte_idx, v);
                                }
                            }
                        } else {
                            for (size_t o = 0; o < outer_count; ++o) {
                                auto in_base = static_cast<uint32_t>(o * input_block_stride + start_offset);
                                auto out_base = static_cast<uint32_t>(o * output_block_size);
                                for (auto i : dynamic_range(static_cast<uint32_t>(output_block_size))) {
                                    out[i + out_base] = in[i + in_base];
                                }
                            }
                        }
                    } else {
                        for (size_t o = 0; o < outer_count; ++o) {
                            auto in_base = static_cast<uint32_t>(o * input_block_stride + start_offset);
                            auto out_base = static_cast<uint32_t>(o * output_block_size);
                            for (auto i : dynamic_range(static_cast<uint32_t>(output_block_size))) {
                                out[i + out_base] = in[i + in_base];
                            }
                        }
                    }
                }
                return true;
            };

            if (try_const_fast_path()) return;

            // Constant optimized slow path: pre-compute dim_starts/dim_steps on CPU
            // to eliminate DSL $if branches, and vectorize inner dimension when possible.
            auto try_const_optimized_path = [&]() -> bool {
                if (!inputs[1].get().is_constant() || !inputs[2].get().is_constant())
                    return false;
                if (has_steps && !inputs[4].get().is_constant())
                    return false;
                if (has_axes && !inputs[3].get().is_constant())
                    return false;

                auto &starts_const = static_cast<NNConstTensor<int> const &>(inputs[1].get());
                auto num_s = starts_const.const_data().size();

                std::vector<int> cpu_starts(ndim, 0);
                std::vector<int> cpu_steps(ndim, 1);

                for (size_t s = 0; s < num_s; ++s) {
                    int ax = has_axes ?
                                 static_cast<NNConstTensor<int> const &>(inputs[3].get()).const_data()[s] :
                                 static_cast<int>(s);
                    if (ax < 0) ax += static_cast<int>(ndim);
                    if (ax < 0 || ax >= static_cast<int>(ndim)) continue;

                    int sv = starts_const.const_data()[s];
                    int dim_size = static_cast<int>(in.shape()[ax]);
                    if (sv < 0) sv += dim_size;
                    sv = std::max(0, std::min(sv, dim_size));

                    int step = has_steps ?
                                   static_cast<NNConstTensor<int> const &>(inputs[4].get()).const_data()[s] :
                                   1;

                    cpu_starts[ax] = sv;
                    cpu_steps[ax] = step;
                }

                DynamicArray<int> dim_starts(ndim);
                DynamicArray<int> dim_steps(ndim);
                for (uint32_t d = 0; d < ndim; ++d) {
                    dim_starts[d] = cpu_starts[d];
                    dim_steps[d] = cpu_steps[d];
                }

                if (in.container().shares_storage_with(out.container())) return true;

                if constexpr (detail::VecDispatch<ST>::supported) {
                    bool inner_vec = (ndim > 0 &&
                                      cpu_steps[ndim - 1] == 1 &&
                                      in.strides()[ndim - 1] == 1 &&
                                      out.strides()[ndim - 1] == 1 &&
                                      in.container().is_byte_buffer() &&
                                      out.container().is_byte_buffer());
                    if (inner_vec && out.shape()[ndim - 1] >= detail::VecDispatch<ST>::N) {
                        uint32_t inner_size = out.shape()[ndim - 1];
                        uint32_t outer_count = out.size() / inner_size;
                        using VecT = typename detail::VecDispatch<ST>::VecT;
                        auto buf_in = in.container().get_byte_buffer();
                        auto buf_o = out.container().get_byte_buffer();
                        auto off_in = static_cast<uint>(in.container().get_byte_offset());
                        auto off_o = static_cast<uint>(out.container().get_byte_offset());
                        auto elem_size = static_cast<uint>(sizeof(ST));
                        for (uint32_t block = 0; block < outer_count; ++block) {
                            auto block_start = block * inner_size;
                            auto in_base = def(0u);
                            for_each_dim(block_start, out.strides(), out.shape(), ndim,
                                         [&](uint32_t d, auto coord) {
                                             auto in_coord = (dim_starts[d] + coord.cast<int>() * dim_steps[d]).cast<uint>();
                                             in_base += in_coord * in.strides()[d];
                                         });
                            auto vec_n = inner_size / detail::VecDispatch<ST>::N;
                            auto rem = inner_size % detail::VecDispatch<ST>::N;
                            auto in_byte_base = in_base * elem_size;
                            auto out_byte_base = block_start * elem_size;
                            for (auto i : dynamic_range(vec_n)) {
                                auto byte_idx = i * static_cast<uint>(sizeof(VecT));
                                auto v = buf_in->read<VecT>(off_in + in_byte_base + byte_idx);
                                buf_o->write(off_o + out_byte_base + byte_idx, v);
                            }
                            for (auto i : dynamic_range(rem)) {
                                auto idx = vec_n * detail::VecDispatch<ST>::N + i;
                                auto byte_idx = idx * elem_size;
                                auto v = buf_in->read<ST>(off_in + in_byte_base + byte_idx);
                                buf_o->write(off_o + out_byte_base + byte_idx, v);
                            }
                        }
                        return true;
                    }
                }

                if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                    auto buf_in = in.container().get_byte_buffer();
                    auto buf_o = out.container().get_byte_buffer();
                    auto off_in = static_cast<uint>(in.container().get_byte_offset());
                    auto off_o = static_cast<uint>(out.container().get_byte_offset());
                    auto elem_size = static_cast<uint>(sizeof(ST));
                    for (auto linear_out : dynamic_range(out.size())) {
                        auto in_linear = def(0u);
                        for_each_dim(linear_out, out.strides(), out.shape(), ndim,
                                     [&](uint32_t d, auto coord) {
                                         auto in_coord = (dim_starts[d] + coord.cast<int>() * dim_steps[d]).cast<uint>();
                                         in_linear += in_coord * in.strides()[d];
                                     });
                        auto v = buf_in->read<ST>(off_in + in_linear * elem_size);
                        buf_o->write(off_o + linear_out * elem_size, v);
                    }
                } else {
                    for (auto linear_out : dynamic_range(out.size())) {
                        auto in_linear = def(0u);
                        for_each_dim(linear_out, out.strides(), out.shape(), ndim,
                                     [&](uint32_t d, auto coord) {
                                         auto in_coord = (dim_starts[d] + coord.cast<int>() * dim_steps[d]).cast<uint>();
                                         in_linear += in_coord * in.strides()[d];
                                     });
                        out[linear_out] = in[in_linear];
                    }
                }
                return true;
            };

            if (try_const_optimized_path()) return;

            // Dynamic slow path: build dim_starts/dim_steps with DSL branches
            DynamicArray<int> dim_starts(ndim);
            DynamicArray<int> dim_steps(ndim);
            for (uint32_t d = 0; d < ndim; ++d) {
                dim_starts[d] = 0;
                dim_steps[d] = 1;
            }

            for (uint32_t s = 0; s < num_slices; ++s) {
                Int axis_val = has_axes ? axes_t[s] : Int{static_cast<int>(s)};
                Int start_val = starts_t[s];
                Int step_val = has_steps ? steps_t[s] : Int{1};

                if (has_axes) {
                    for (uint32_t d = 0; d < ndim; ++d) {
                        $if ((axis_val == Int{static_cast<int>(d)}) |
                             (axis_val == Int{static_cast<int>(d) - static_cast<int>(ndim)})) {
                            $if (start_val < 0) {
                                dim_starts[d] = start_val + Int{static_cast<int>(in.shape()[d])};
                            } $else {
                                dim_starts[d] = start_val;
                            };
                            dim_steps[d] = step_val;
                        };
                    }
                } else {
                    if (s < ndim) {
                        $if (start_val < 0) {
                            dim_starts[s] = start_val + Int{static_cast<int>(in.shape()[s])};
                        } $else {
                            dim_starts[s] = start_val;
                        };
                        dim_steps[s] = step_val;
                    }
                }
            }

            if (in.container().shares_storage_with(out.container())) return;

            if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                auto buf_in = in.container().get_byte_buffer();
                auto buf_o = out.container().get_byte_buffer();
                auto off_in = static_cast<uint>(in.container().get_byte_offset());
                auto off_o = static_cast<uint>(out.container().get_byte_offset());
                auto elem_size = static_cast<uint>(sizeof(ST));
                for (auto linear_out : dynamic_range(out.size())) {
                    auto in_linear = def(0u);
                    for_each_dim(linear_out, out.strides(), out.shape(), ndim,
                                 [&](uint32_t d, auto coord) {
                                     auto in_coord = (dim_starts[d] + coord.cast<int>() * dim_steps[d]).cast<uint>();
                                     in_linear += in_coord * in.strides()[d];
                                 });
                    auto v = buf_in->read<ST>(off_in + in_linear * elem_size);
                    buf_o->write(off_o + linear_out * elem_size, v);
                }
            } else {
                for (auto linear_out : dynamic_range(out.size())) {
                    auto in_linear = def(0u);
                    for_each_dim(linear_out, out.strides(), out.shape(), ndim,
                                 [&](uint32_t d, auto coord) {
                                     auto in_coord = (dim_starts[d] + coord.cast<int>() * dim_steps[d]).cast<uint>();
                                     in_linear += in_coord * in.strides()[d];
                                 });
                    out[linear_out] = in[in_linear];
                }
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(Slice) {
    return std::make_unique<Slice>();
};

// DynamicSlice: same as Slice but with dynamic inputs
// (This is actually a deprecated op in ONNX; Slice with input tensors superseded it)
class DynamicSlice : public Operator {
public:
    DynamicSlice() : Operator("DynamicSlice") {}

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() >= 3 && outputs.size() == 1,
                     "DynamicSlice requires >=3 inputs and 1 output.");
        auto &data = inputs[0].get();
        auto &output = outputs[0].get();
        LUISA_ASSERT(data.element_type() == output.element_type(),
                     "DynamicSlice: data and output must have the same element type.");
#else
        auto &data = inputs[0].get();
        auto &output = outputs[0].get();
#endif
        auto ndim = data.ndim();

        auto &starts_t = static_cast<NNTensor<int> &>(inputs[1].get());
        auto &ends_t = static_cast<NNTensor<int> &>(inputs[2].get());
        auto num_slices = starts_t.size();
        bool has_axes = (inputs.size() >= 4 && inputs[3].get().size() > 0);

        visit_typeid<NNTypeList>(data.element_type(), [&]<typename T>() {
            auto &in = static_cast<NNTensor<T> &>(data);
            auto &out = static_cast<NNTensor<T> &>(output);
            using ST = nn_storage_type_t<T>;

            if (in.container().shares_storage_with(out.container())) return;

            DynamicArray<int> dim_starts(ndim);
            DynamicArray<int> dim_steps(ndim);
            for (uint32_t d = 0; d < ndim; ++d) {
                dim_starts[d] = 0;
                dim_steps[d] = 1;
            }

            for (uint32_t s = 0; s < num_slices; ++s) {
                Int axis_val = has_axes ?
                                   static_cast<NNTensor<int> &>(inputs[3].get())[s] :
                                   Int{static_cast<int>(s)};
                Int start_val = starts_t[s];

                for (uint32_t d = 0; d < ndim; ++d) {
                    $if ((axis_val == Int{static_cast<int>(d)}) |
                         (axis_val == Int{static_cast<int>(d) - static_cast<int>(ndim)})) {
                        $if (start_val < 0) {
                            dim_starts[d] = start_val + Int{static_cast<int>(in.shape()[d])};
                        } $else {
                            dim_starts[d] = start_val;
                        };
                    };
                }
            }

            if (in.container().is_byte_buffer() && out.container().is_byte_buffer()) {
                auto buf_in = in.container().get_byte_buffer();
                auto buf_o = out.container().get_byte_buffer();
                auto off_in = static_cast<uint>(in.container().get_byte_offset());
                auto off_o = static_cast<uint>(out.container().get_byte_offset());
                auto elem_size = static_cast<uint>(sizeof(ST));
                for (auto linear_out : dynamic_range(out.size())) {
                    auto in_linear = def(0u);
                    for_each_dim(linear_out, out.strides(), out.shape(), ndim,
                                 [&](uint32_t d, auto coord) {
                                     auto in_coord = (dim_starts[d] + coord.cast<int>() * dim_steps[d]).cast<uint>();
                                     in_linear += in_coord * in.strides()[d];
                                 });
                    auto v = buf_in->read<ST>(off_in + in_linear * elem_size);
                    buf_o->write(off_o + linear_out * elem_size, v);
                }
            } else {
                for (auto linear_out : dynamic_range(out.size())) {
                    auto in_linear = def(0u);
                    for_each_dim(linear_out, out.strides(), out.shape(), ndim,
                                 [&](uint32_t d, auto coord) {
                                     auto in_coord = (dim_starts[d] + coord.cast<int>() * dim_steps[d]).cast<uint>();
                                     in_linear += in_coord * in.strides()[d];
                                 });
                    out[linear_out] = in[in_linear];
                }
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(DynamicSlice) {
    return std::make_unique<DynamicSlice>();
};

}// namespace lcml::onnx
