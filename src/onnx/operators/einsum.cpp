#include "onnx/operator.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"
#include <sstream>
#include <algorithm>

namespace lcml::onnx {

// ============================================================================
// Einsum: evaluates a general Einstein summation expression.
// ONNX spec: attribute equation (string)
//
// Supports the full einsum syntax:
//   - Explicit labels: "ij,jk->ik" (matrix multiply)
//   - Implicit output:  "ij,jk" (auto-deduced output labels)
//   - Ellipsis:        "...ij,...jk->...ik" (batched matmul)
//   - Transpose:       "...ji->...ij"
//   - Trace:           "...ii->...i"
//   - Outer product:   "i,j->ij"
//   - Summation:       "ij->" (sum all elements)
//   - Mixed:           "bij,bjk->bik"
//
// Algorithm:
//   1. Parse equation, expand ellipsis into concrete internal labels.
//   2. Collect all unique labels and map them to dimension sizes.
//   3. Separate labels into output dims and contraction (reduction) dims.
//   4. For each output element, iterate over contraction space,
//      compute product of all inputs, and accumulate.
// ============================================================================
class Einsum : public Operator {
private:
    std::string equation_;

    // Internal label alphabet for ellipsis expansion.
    // We use uppercase letters (A-Z) which won't conflict with user labels (a-z).
    static char ellipsis_label(int idx) {
        LUISA_ASSERT(idx < 26, "Einsum: ellipsis expansion exceeds 26 dimensions");
        return static_cast<char>('A' + idx);
    }

    // Remove all spaces from string
    static std::string strip(const std::string &s) {
        std::string r;
        r.reserve(s.size());
        for (char c : s)
            if (c != ' ') r += c;
        return r;
    }

    // Split a string by delimiter
    static std::vector<std::string> split(const std::string &s, char delim) {
        std::vector<std::string> parts;
        std::istringstream ss(s);
        std::string token;
        while (std::getline(ss, token, delim))
            parts.push_back(token);
        return parts;
    }

    // Count explicit (non-ellipsis) labels in a subscript
    static int count_explicit_labels(const std::string &sub) {
        int n = 0;
        for (size_t i = 0; i < sub.size(); ++i) {
            if (sub[i] == '.') {
                // Skip the entire "..."
                LUISA_ASSERT(i + 2 < sub.size() && sub[i + 1] == '.' && sub[i + 2] == '.',
                             "Einsum: invalid ellipsis notation in '{}'", sub);
                i += 2;// skip 3 dots total
            } else {
                ++n;
            }
        }
        return n;
    }

    // Check if subscript contains ellipsis
    static bool has_ellipsis(const std::string &sub) {
        return sub.find("...") != std::string::npos;
    }

    // Expand a subscript's ellipsis with the given labels string,
    // returning a new subscript with "..." replaced by those labels.
    static std::string expand_ellipsis(const std::string &sub,
                                       const std::string &ellipsis_labels) {
        std::string result;
        for (size_t i = 0; i < sub.size(); ++i) {
            if (sub[i] == '.' && i + 2 < sub.size() && sub[i + 1] == '.' && sub[i + 2] == '.') {
                result += ellipsis_labels;
                i += 2;// skip 3 dots
            } else {
                result += sub[i];
            }
        }
        return result;
    }

    struct ParsedEquation {
        std::vector<std::string> input_subs;// expanded (no ellipsis)
        std::string output_sub;             // expanded (no ellipsis)
    };

    // Full equation parser with ellipsis support.
    static ParsedEquation parse_equation(
        const std::string &eq,
        const std::vector<ITensor::shape_type> &input_shapes) {

        std::string clean = strip(eq);
        std::string lhs, rhs;
        bool has_arrow = false;

        auto arrow = clean.find("->");
        if (arrow != std::string::npos) {
            lhs = clean.substr(0, arrow);
            rhs = clean.substr(arrow + 2);
            has_arrow = true;
        } else {
            lhs = clean;
        }

        // Split lhs into raw input subscripts
        auto raw_subs = split(lhs, ',');
        LUISA_ASSERT(raw_subs.size() == input_shapes.size(),
                     "Einsum: equation has {} inputs but got {}",
                     raw_subs.size(), input_shapes.size());

        // Determine if any subscript uses ellipsis
        bool any_ellipsis = false;
        for (auto &s : raw_subs)
            if (has_ellipsis(s)) any_ellipsis = true;
        if (has_arrow && has_ellipsis(rhs)) any_ellipsis = true;

        // Compute ellipsis expansion
        std::string ellipsis_labels;
        if (any_ellipsis) {
            // For each input with ellipsis, compute how many dims the ellipsis covers
            // ellipsis_ndim = tensor_ndim - num_explicit_labels
            // All inputs must agree on broadcast-compatible ellipsis shapes.
            int max_ellipsis_ndim = 0;
            std::vector<int> ellipsis_ndims(raw_subs.size(), 0);

            for (size_t t = 0; t < raw_subs.size(); ++t) {
                if (has_ellipsis(raw_subs[t])) {
                    int explicit_n = count_explicit_labels(raw_subs[t]);
                    int ndim = static_cast<int>(input_shapes[t].size());
                    ellipsis_ndims[t] = ndim - explicit_n;
                    LUISA_ASSERT(ellipsis_ndims[t] >= 0,
                                 "Einsum: subscript '{}' has more explicit labels ({}) than tensor dims ({})",
                                 raw_subs[t], explicit_n, ndim);
                    max_ellipsis_ndim = std::max(max_ellipsis_ndim, ellipsis_ndims[t]);
                }
            }

            // Broadcast ellipsis dimensions across all inputs
            // ellipsis_shape[k] = broadcast of all inputs' ellipsis dim k
            std::vector<uint32_t> ellipsis_shape(max_ellipsis_ndim, 1u);

            for (size_t t = 0; t < raw_subs.size(); ++t) {
                if (!has_ellipsis(raw_subs[t])) continue;
                int e_ndim = ellipsis_ndims[t];
                // Find where ellipsis starts in the tensor shape
                // by finding position of "..." in the subscript
                auto dot_pos = raw_subs[t].find("...");
                int labels_before = static_cast<int>(dot_pos);

                // Ellipsis dims are right-aligned for broadcasting
                int offset = max_ellipsis_ndim - e_ndim;
                for (int k = 0; k < e_ndim; ++k) {
                    uint32_t dim_size = input_shapes[t][labels_before + k];
                    int bk = offset + k;// broadcast-aligned index
                    if (ellipsis_shape[bk] == 1u) {
                        ellipsis_shape[bk] = dim_size;
                    } else {
                        LUISA_ASSERT(dim_size == 1u || dim_size == ellipsis_shape[bk],
                                     "Einsum: ellipsis dimension {} not broadcast-compatible: {} vs {}",
                                     k, ellipsis_shape[bk], dim_size);
                        if (dim_size > 1u) ellipsis_shape[bk] = dim_size;
                    }
                }
            }

            // Generate internal labels for ellipsis dims
            for (int k = 0; k < max_ellipsis_ndim; ++k) {
                ellipsis_labels += ellipsis_label(k);
            }
        }

        // Expand all subscripts
        ParsedEquation result;
        for (auto &s : raw_subs) {
            if (has_ellipsis(s)) {
                // For inputs with fewer ellipsis dims than max, pad with size-1 dims
                // by inserting extra labels at the front of the ellipsis expansion
                int e_ndim_this = static_cast<int>(input_shapes[&s - &raw_subs[0]].size()) - count_explicit_labels(s);
                int pad = static_cast<int>(ellipsis_labels.size()) - e_ndim_this;
                // The full ellipsis_labels covers all broadcast dims.
                // For this input, the first 'pad' labels are broadcast (dim=1).
                result.input_subs.push_back(expand_ellipsis(s, ellipsis_labels.substr(pad)));
            } else {
                result.input_subs.push_back(s);
            }
        }

        if (has_arrow) {
            if (any_ellipsis && has_ellipsis(rhs)) {
                result.output_sub = expand_ellipsis(rhs, ellipsis_labels);
            } else {
                result.output_sub = rhs;
            }
        } else {
            // Implicit output mode: labels appearing exactly once, sorted.
            // Ellipsis labels go first (leftmost), then sorted free labels.
            std::unordered_map<char, int> count;
            for (auto &s : result.input_subs)
                for (char c : s) count[c]++;

            if (any_ellipsis) {
                // Ellipsis labels always appear in output
                result.output_sub = ellipsis_labels;
            }
            // Free labels (appear exactly once) in sorted order
            std::string free_labels;
            for (auto &[c, n] : count) {
                if (n == 1 && ellipsis_labels.find(c) == std::string::npos) {
                    free_labels += c;
                }
            }
            std::sort(free_labels.begin(), free_labels.end());
            result.output_sub += free_labels;
        }

        return result;
    }

public:
    Einsum(std::string equation) : Operator("Einsum"), equation_(std::move(equation)) {}

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
        LUISA_ASSERT(!inputs.empty() && outputs.size() == 1,
                     "Einsum requires >=1 inputs and 1 output.");
        auto &Y = outputs[0].get();

        // Collect input shapes
        std::vector<ITensor::shape_type> input_shapes;
        input_shapes.reserve(inputs.size());
        for (auto &inp : inputs)
            input_shapes.push_back(inp.get().shape());

        // Parse equation with ellipsis expansion
        auto parsed = parse_equation(equation_, input_shapes);
        auto &input_subs = parsed.input_subs;
        auto &output_sub = parsed.output_sub;

        LUISA_ASSERT(input_subs.size() == inputs.size(),
                     "Einsum: equation has {} inputs but got {}",
                     input_subs.size(), inputs.size());

        // Collect all unique labels and determine their dimension sizes
        std::string all_labels;// ordered unique labels
        std::unordered_map<char, uint32_t> label_size;

        for (size_t t = 0; t < inputs.size(); ++t) {
            auto &sub = input_subs[t];
            auto &shape = input_shapes[t];
            LUISA_ASSERT(sub.size() == shape.size(),
                         "Einsum: expanded subscript '{}' has {} dims but tensor has {} dims",
                         sub, sub.size(), shape.size());
            for (size_t d = 0; d < sub.size(); ++d) {
                char c = sub[d];
                if (label_size.find(c) == label_size.end()) {
                    label_size[c] = shape[d];
                    all_labels += c;
                } else {
                    // For broadcast-expanded ellipsis dims, allow size-1 vs size-N
                    if (label_size[c] != shape[d]) {
                        if (shape[d] == 1u) {
                            // This input broadcasts on this label — keep existing size
                        } else if (label_size[c] == 1u) {
                            label_size[c] = shape[d];
                        } else {
                            LUISA_ASSERT(false,
                                         "Einsum: label '{}' has inconsistent sizes {} vs {}",
                                         c, label_size[c], shape[d]);
                        }
                    }
                }
            }
        }

        // Separate into output labels and contraction labels
        std::string contract_labels;
        for (char c : all_labels) {
            if (output_sub.find(c) == std::string::npos) {
                contract_labels += c;
            }
        }

        // Build output label sizes and strides (row-major) for coordinate decomposition
        uint32_t out_ndim = static_cast<uint32_t>(output_sub.size());
        std::vector<uint32_t> out_label_sizes(out_ndim);
        for (uint32_t i = 0; i < out_ndim; ++i) {
            out_label_sizes[i] = label_size[output_sub[i]];
        }
        std::vector<uint32_t> out_strides(out_ndim, 1u);
        for (int i = static_cast<int>(out_ndim) - 2; i >= 0; --i) {
            out_strides[i] = out_strides[i + 1] * out_label_sizes[i + 1];
        }

        // Build contraction label sizes and strides
        uint32_t con_ndim = static_cast<uint32_t>(contract_labels.size());
        std::vector<uint32_t> con_sizes(con_ndim);
        uint32_t con_total = 1;
        for (uint32_t i = 0; i < con_ndim; ++i) {
            con_sizes[i] = label_size[contract_labels[i]];
            con_total *= con_sizes[i];
        }
        std::vector<uint32_t> con_strides(con_ndim, 1u);
        for (int i = static_cast<int>(con_ndim) - 2; i >= 0; --i) {
            con_strides[i] = con_strides[i + 1] * con_sizes[i + 1];
        }

        // Combined labels: output labels + contraction labels
        auto combined_labels = output_sub + contract_labels;
        uint32_t total_labels = static_cast<uint32_t>(combined_labels.size());

        // For each input tensor, precompute:
        //   input_label_strides[t][l] = stride in input t for combined_label[l]
        //   If the label doesn't appear => stride = 0 (broadcast / absent).
        //   If the label appears but input's dim size is 1 => stride = 0 (broadcast).
        std::vector<std::vector<uint32_t>> input_label_strides(inputs.size());
        for (size_t t = 0; t < inputs.size(); ++t) {
            auto &sub = input_subs[t];
            auto &strides = inputs[t].get().strides();
            auto &shape = input_shapes[t];
            input_label_strides[t].resize(total_labels, 0u);
            for (uint32_t l = 0; l < total_labels; ++l) {
                char lbl = combined_labels[l];
                auto pos = sub.find(lbl);
                if (pos != std::string::npos) {
                    // If this input's dim is 1, it's broadcast — use stride 0
                    if (shape[pos] == 1u && label_size[lbl] > 1u) {
                        input_label_strides[t][l] = 0u;
                    } else {
                        input_label_strides[t][l] = strides[pos];
                    }
                }
            }
        }

        visit_typeid<NNFilteredTypeList<IsFloatingPoint>>(Y.element_type(), [&]<typename T>() {
            using VT = nn_storage_type_t<T>;
            auto &y = static_cast<NNTensor<T> &>(Y);

            // Prepare typed input tensor references
            std::vector<NNTensor<T> *> typed_inputs(inputs.size());
            for (size_t t = 0; t < inputs.size(); ++t) {
                typed_inputs[t] = &static_cast<NNTensor<T> &>(inputs[t].get());
            }

            // Helper: scalar read via ByteBuffer when possible, else DynamicArray
            auto read_scalar = [&](auto *tensor, Var<uint> idx) -> Var<VT> {
                if (tensor->container().is_byte_buffer()) {
                    auto buf = tensor->container().get_byte_buffer();
                    auto off = static_cast<uint>(tensor->container().get_byte_offset());
                    return buf->read<VT>(off + idx * static_cast<uint>(sizeof(VT)));
                }
                return (*tensor)[idx];
            };

            // Helper: scalar write via ByteBuffer when possible, else DynamicArray
            auto write_scalar = [&](Var<uint> idx, Var<VT> val) {
                if (y.container().is_byte_buffer()) {
                    auto buf = y.container().get_byte_buffer();
                    auto off = static_cast<uint>(y.container().get_byte_offset());
                    buf->write(off + idx * static_cast<uint>(sizeof(VT)), val);
                } else {
                    y[idx] = val;
                }
            };

            // Determine vectorization eligibility for common 2-input paths
            bool use_vec_2in_nocontract = false;
            bool use_vec_2in_contract = false;
            if (typed_inputs.size() == 2 && detail::VecDispatch<VT>::supported) {
                // No-contraction: innermost output dim must be contiguous (stride 1) or broadcast (stride 0)
                if (con_total == 1 && con_ndim == 0 && out_ndim > 0 &&
                    y.container().is_byte_buffer() &&
                    typed_inputs[0]->container().is_byte_buffer() &&
                    typed_inputs[1]->container().is_byte_buffer() &&
                    input_label_strides[0][out_ndim - 1] <= 1 &&
                    input_label_strides[1][out_ndim - 1] <= 1) {
                    use_vec_2in_nocontract = true;
                }
                // Contraction: innermost contraction dim must be contiguous (stride 1) in both inputs
                if (con_ndim > 0 &&
                    typed_inputs[0]->container().is_byte_buffer() &&
                    typed_inputs[1]->container().is_byte_buffer() &&
                    con_sizes.back() >= 4 &&
                    input_label_strides[0][out_ndim + con_ndim - 1] == 1 &&
                    input_label_strides[1][out_ndim + con_ndim - 1] == 1) {
                    use_vec_2in_contract = true;
                }
            }

            // Fast path: 2-input no-contraction with inner-dim vectorization
            if (use_vec_2in_nocontract) {
                detail::vectorized_broadcast_2in<VT>(
                    *typed_inputs[0], *typed_inputs[1], y,
                    out_label_sizes, out_ndim,
                    input_label_strides[0], input_label_strides[1],
                    [&](auto va, auto vb) { return va * vb; });
            } else {
                for (auto out_linear : dynamic_range(y.size())) {
                    // Decompose output linear index into label coordinates
                    DynamicArray<RemoveVarT<decltype(out_linear)>> label_coords(total_labels);

                    // First out_ndim entries = output label coords
                    for_each_dim(out_linear, out_strides, out_label_sizes, out_ndim,
                                 [&](uint32_t d, auto coord) {
                                     label_coords[d] = coord;
                                 });

                    auto acc = def(VT{0});

                    if (con_total == 1 && con_ndim == 0) {
                        // No contraction: just element-wise product
                        auto prod = def(VT{1});
                        for (size_t t = 0; t < typed_inputs.size(); ++t) {
                            auto idx = def(0u);
                            for (uint32_t l = 0; l < out_ndim; ++l) {
                                auto s = input_label_strides[t][l];
                                if (s == 1) {
                                    idx += label_coords[l];
                                } else if (s > 1) {
                                    idx += label_coords[l] * s;
                                }
                            }
                            prod *= read_scalar(typed_inputs[t], idx);
                        }
                        acc = prod;
                    } else {
                        // With contraction: sum over contraction dimensions
                        if (use_vec_2in_contract) {
                            using VecT = typename detail::VecDispatch<VT>::VecT;
                            auto buf0 = typed_inputs[0]->container().get_byte_buffer();
                            auto buf1 = typed_inputs[1]->container().get_byte_buffer();
                            auto off0 = static_cast<uint>(typed_inputs[0]->container().get_byte_offset());
                            auto off1 = static_cast<uint>(typed_inputs[1]->container().get_byte_offset());

                            uint32_t inner_size = con_sizes.back();
                            uint32_t outer_con_total = con_total / inner_size;
                            uint32_t vec_n = inner_size / 4;
                            uint32_t rem = inner_size % 4;
                            auto last_con_label_idx = out_ndim + con_ndim - 1;

                            for (auto outer : dynamic_range(outer_con_total)) {
                                // Decompose outer into contraction coords for dims 0..con_ndim-2
                                if (con_ndim > 1) {
                                    auto outer_remaining = def(cast<uint>(outer));
                                    for (uint32_t d = 0; d < con_ndim - 1; ++d) {
                                        if (con_strides[d] == 1) {
                                            label_coords[out_ndim + d] = outer_remaining;
                                        } else {
                                            auto coord = outer_remaining / con_strides[d];
                                            outer_remaining = outer_remaining % con_strides[d];
                                            label_coords[out_ndim + d] = coord;
                                        }
                                    }
                                }

                                // Compute base indices excluding the vectorized last contraction dim
                                auto base_idx0 = def(0u);
                                auto base_idx1 = def(0u);
                                for (uint32_t l = 0; l < total_labels; ++l) {
                                    if (l == last_con_label_idx) continue;
                                    auto s0 = input_label_strides[0][l];
                                    if (s0 == 1) {
                                        base_idx0 += label_coords[l];
                                    } else if (s0 > 1) {
                                        base_idx0 += label_coords[l] * s0;
                                    }
                                    auto s1 = input_label_strides[1][l];
                                    if (s1 == 1) {
                                        base_idx1 += label_coords[l];
                                    } else if (s1 > 1) {
                                        base_idx1 += label_coords[l] * s1;
                                    }
                                }

                                auto vec_acc = detail::VecDispatch<VT>::broadcast(def(VT{0}));
                                for (auto v : dynamic_range(vec_n)) {
                                    auto k = v * 4u;
                                    auto va = buf0->read<VecT>(off0 + (base_idx0 + k) * static_cast<uint>(sizeof(VT)));
                                    auto vb = buf1->read<VecT>(off1 + (base_idx1 + k) * static_cast<uint>(sizeof(VT)));
                                    vec_acc = luisa::compute::fma(va, vb, vec_acc);
                                }

                                acc += vec_acc.x + vec_acc.y + vec_acc.z + vec_acc.w;

                                for (auto r : dynamic_range(rem)) {
                                    auto k = vec_n * 4u + r;
                                    auto va = buf0->read<VT>(off0 + (base_idx0 + k) * static_cast<uint>(sizeof(VT)));
                                    auto vb = buf1->read<VT>(off1 + (base_idx1 + k) * static_cast<uint>(sizeof(VT)));
                                    acc = luisa::compute::fma(va, vb, acc);
                                }
                            }
                        } else {
                            for (auto con_linear : dynamic_range(con_total)) {
                                // Decompose contraction linear index into contraction label coords
                                if (con_ndim > 0) {
                                    auto con_remaining = def(cast<uint>(con_linear));
                                    for (uint32_t d = 0; d < con_ndim; ++d) {
                                        if (con_strides[d] == 1) {
                                            label_coords[out_ndim + d] = con_remaining;
                                        } else {
                                            auto coord = con_remaining / con_strides[d];
                                            con_remaining = con_remaining % con_strides[d];
                                            label_coords[out_ndim + d] = coord;
                                        }
                                    }
                                }

                                // Compute product of all inputs at current label coordinates
                                if (typed_inputs.size() == 2) {
                                    auto idx0 = def(0u);
                                    for (uint32_t l = 0; l < total_labels; ++l) {
                                        auto s = input_label_strides[0][l];
                                        if (s == 1) {
                                            idx0 += label_coords[l];
                                        } else if (s > 1) {
                                            idx0 += label_coords[l] * s;
                                        }
                                    }
                                    auto idx1 = def(0u);
                                    for (uint32_t l = 0; l < total_labels; ++l) {
                                        auto s = input_label_strides[1][l];
                                        if (s == 1) {
                                            idx1 += label_coords[l];
                                        } else if (s > 1) {
                                            idx1 += label_coords[l] * s;
                                        }
                                    }
                                    acc = luisa::compute::fma(read_scalar(typed_inputs[0], idx0), read_scalar(typed_inputs[1], idx1), acc);
                                } else {
                                    auto prod = def(VT{1});
                                    for (size_t t = 0; t < typed_inputs.size(); ++t) {
                                        auto idx = def(0u);
                                        for (uint32_t l = 0; l < total_labels; ++l) {
                                            auto s = input_label_strides[t][l];
                                            if (s == 1) {
                                                idx += label_coords[l];
                                            } else if (s > 1) {
                                                idx += label_coords[l] * s;
                                            }
                                        }
                                        prod *= read_scalar(typed_inputs[t], idx);
                                    }
                                    acc += prod;
                                }
                            }
                        }
                    }

                    write_scalar(out_linear, acc);
                }
            }
        });
    }
};

REGISTER_TO_DEFAULT_OPSET(Einsum) {
    std::string equation;
    if (auto p = node.try_get_attr("equation"))
        equation = p->get<onnx::AttributeType::STRING>();
    return std::make_unique<Einsum>(std::move(equation));
};

}// namespace lcml::onnx
