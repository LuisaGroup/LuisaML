#include "onnx/network_instance.h"
#include "onnx/register_allocator/register_allocator.h"
#include "luisa/ast/function_builder.h"
#include <unordered_set>

namespace lcml::onnx {
using namespace luisa::compute;

static ITensor::shape_type to_tensor_shape(std::vector<size_t> const &shape) {
    ITensor::shape_type result;
    result.reserve(shape.size());
    for (auto s : shape) { result.push_back(static_cast<ITensor::size_type>(s)); }
    return result;
}

size_t NetworkInstance::compute_num_elements(std::vector<size_t> const &shape) {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>{});
}

void NetworkInstance::validate_and_bind(
    TensorTable &tensor_table,
    std::string_view role,
    onnx::Variable const &var,
    ITensor &ext_tensor) {

    auto const &name = var.get_name();

    // Validate element type (typeid) matches
    auto const &expected_typeid = onnx_dtype_to_typeid(var.get_dtype());
    LUISA_ASSERT(ext_tensor.element_type() == expected_typeid,
                 "{} '{}' element type mismatch: expected {}, got {}",
                 role, name, expected_typeid.name(), ext_tensor.element_type().name());

    // Validate shape matches
    auto const &expected_shape = var.get_shape();
    auto const &actual_shape = ext_tensor.shape();
    LUISA_ASSERT(actual_shape.size() == expected_shape.size(),
                 "{} '{}' shape rank mismatch: expected {}, got {}",
                 role, name, expected_shape.size(), actual_shape.size());
    for (size_t d = 0; d < expected_shape.size(); ++d) {
        LUISA_ASSERT(actual_shape[d] == expected_shape[d],
                     "{} '{}' shape mismatch at dim {}: expected {}, got {}",
                     role, name, d, expected_shape[d], actual_shape[d]);
    }
    ext_tensor.set_name(std::string{role} + "_" + name);

    // Bind as borrowed reference in tensor table
    tensor_table.bind(name, ext_tensor);
}

void NetworkInstance::bind_external_tensors(
    TensorTable &tensor_table,
    onnx::Graph const &graph) {

    for (auto const &var_ref : graph.get_inputs()) {
        auto const &var = var_ref.get();
        if (auto it = inputs_.find(var.get_name()); it != inputs_.end()) {
            validate_and_bind(tensor_table, "Input", var, it->second.get());
        }
    }

    for (auto const &var_ref : graph.get_outputs()) {
        auto const &var = var_ref.get();
        if (auto it = outputs_.find(var.get_name()); it != outputs_.end()) {
            validate_and_bind(tensor_table, "Output", var, it->second.get());
        }
    }
}
inline std::string_view onnx_dtype_to_string(onnx::DataType dt) {
    return magic_enum::enum_name(dt);
}
static std::string format_node_info_markdown(
    const onnx::Graph &graph,
    size_t total_nodes) {
    std::ostringstream oss;
    auto const &nodes = graph.get_nodes();

    // Header
    oss << "\n";
    oss << "# Network Execution Log\n\n";
    oss << "## Summary\n\n";
    oss << "| Metric | Value |\n";
    oss << "|--------|-------|\n";
    oss << "| Total Nodes | " << total_nodes << " |\n";
    oss << "| Graph Inputs | " << graph.get_inputs().size() << " |\n";
    oss << "| Graph Outputs | " << graph.get_outputs().size() << " |\n";
    oss << "| Variables | " << graph.get_variables().size() << " |\n\n";

    // Nodes detailed table
    oss << "## Nodes Detail\n\n";
    oss << "| Index | Op Type | Inputs | Outputs |\n";
    oss << "|-------|---------|--------|---------|\n";

    for (size_t i = 0; i < nodes.size(); ++i) {
        auto const &node = nodes[i];

        // Format inputs
        std::string inputs_str;
        bool first_input = true;
        for (auto const &in_var : node.get_inputs()) {
            if (!first_input) inputs_str += ", ";
            inputs_str += std::string(in_var.get().get_name());
            first_input = false;
        }
        if (inputs_str.empty()) inputs_str = "(none)";

        // Format outputs
        std::string outputs_str;
        bool first_output = true;
        for (auto const &out_var : node.get_outputs()) {
            if (!first_output) outputs_str += ", ";
            outputs_str += std::string(out_var.get().get_name());
            first_output = false;
        }
        if (outputs_str.empty()) outputs_str = "(none)";

        // Escape pipe characters for markdown table
        auto escape_pipes = [](std::string &s) {
            size_t pos = 0;
            while ((pos = s.find('|', pos)) != std::string::npos) {
                s.replace(pos, 1, "\\|");
                pos += 2;
            }
        };
        escape_pipes(inputs_str);
        escape_pipes(outputs_str);

        oss << "| " << i << " | " << node.get_op_type() << " | "
            << inputs_str << " | " << outputs_str << " |\n";
    }

    oss << "\n";
    return oss.str();
}

static std::string format_node_execution_markdown(
    const onnx::Node &node,
    size_t node_idx,
    size_t total_nodes) {
    std::ostringstream oss;

    oss << "\n";
    oss << "### Node [" << node_idx << "/" << (total_nodes - 1) << "]: "
        << node.get_op_type() << "\n\n";

    // Inputs
    oss << "**Inputs:**\n";
    for (auto const &in_var : node.get_inputs()) {
        auto const &var = in_var.get();
        oss << "- `" << var.get_name() << "` : ";
        oss << "shape=[";
        auto const &shape = var.get_shape();
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << shape[i];
        }
        oss << "], dtype=" << onnx_dtype_to_string(var.get_dtype()) << "\n";
    }
    if (node.get_inputs().empty()) {
        oss << "- (none)\n";
    }

    // Outputs
    oss << "\n**Outputs:**\n";
    for (auto const &out_var : node.get_outputs()) {
        auto const &var = out_var.get();
        oss << "- `" << var.get_name() << "` : ";
        oss << "shape=[";
        auto const &shape = var.get_shape();
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << shape[i];
        }
        oss << "], dtype=" << onnx_dtype_to_string(var.get_dtype()) << "\n";
    }
    if (node.get_outputs().empty()) {
        oss << "- (none)\n";
    }

    oss << "\n---\n";
    return oss.str();
}

void NetworkInstance::create_tensor_for_var(
    TensorTable &tensor_table,
    onnx::Variable const &var) {

    auto const &name = var.get_name();
    auto const &shape = var.get_shape();
    auto num_elements = compute_num_elements(shape);
    auto tensor_shape = to_tensor_shape(shape);

    visit_onnx_dtype(var.get_dtype(), [&]<typename T>() {
        std::unique_ptr<NNTensor<T>> tensor;

        // Fill weight data from ByteBuffer (external safetensors file)
        if (var.is_trainable_weight()) {
            LUISA_ASSERT(weight_buffer_ != nullptr,
                         "Weight buffer not set but required for variable: {}", name);
            auto [buf_start, buf_end] = var.get_buffer_range();
            if constexpr (std::is_same_v<T, FP4E2M1>) {
                LUISA_ASSERT(buf_end - buf_start == (num_elements + 1) / 2,
                             "Weight buffer size mismatch for variable: {}", name);
            } else if constexpr (std::is_same_v<T, FP8E4M3FN> || std::is_same_v<T, FP8E5M2>) {
                LUISA_ASSERT(buf_end - buf_start == ((num_elements + 3) / 4) * 4,
                             "Weight buffer size mismatch for variable: {}", name);
            } else {
                LUISA_ASSERT(buf_end - buf_start == num_elements * sizeof(T),
                             "Weight buffer size mismatch for variable: {}", name);
            }
            tensor = std::make_unique<NNTensor<T>>(
                std::move(tensor_shape),
                typename NNTensor<T>::container_type{num_elements, weight_buffer_, buf_start});
        }
        // Fill weight data from raw_data (base64-decoded, embedded in JSON)
        else {
            // Build CPU-side vector for constant tensors
            using BkT = typename NNTensor<T>::value_type;
            using DtT = typename NNConstTensor<T>::storage_type;
            std::vector<DtT> cpu_vec;
            bool has_raw = !var.get_raw_data().empty();
            if (has_raw) {
                cpu_vec.resize(num_elements);
                auto const &raw = var.get_raw_data();
                if constexpr (std::is_same_v<T, FP4E2M1>) {
                    for (size_t i = 0; i < num_elements; ++i) {
                        uint8_t byte = raw[i / 2];
                        uint8_t nibble = (i % 2 == 0) ? (byte >> 4) : (byte & 0x0f);
                        cpu_vec[i].bits = nibble;
                    }
                } else if constexpr (std::is_same_v<T, FP8E4M3FN> || std::is_same_v<T, FP8E5M2>) {
                    // FP8 raw data is densely packed as 1 byte per element,
                    // but FP8 structs have uint16_t bits (2 bytes).
                    for (size_t i = 0; i < num_elements; ++i) {
                        cpu_vec[i].bits = static_cast<uint16_t>(raw[i]);
                    }
                } else {
                    auto const *src = reinterpret_cast<T const *>(raw.data());
                    for (size_t i = 0; i < num_elements; ++i) {
                        if constexpr (std::is_same_v<T, BkT>) {
                            cpu_vec[i] = static_cast<DtT>(src[i]);
                        } else {
                            cpu_vec[i] = static_cast<DtT>(static_cast<BkT>(src[i]));
                        }
                    }
                }
            }
            // Detect arithmetic sequence patterns in constant data:
            // - all_equal (delta == 0) → Scalar mode
            // - arithmetic sequence (constant delta) → Linear mode
            enum class SeqKind { None,
                                 Scalar,
                                 Linear };
            auto detect_sequence = [&]() -> std::pair<SeqKind, BkT /*delta*/> {
                if constexpr (!std::is_arithmetic_v<BkT>) {
                    return {SeqKind::None, BkT{}};
                } else {
                    if (num_elements <= 1u)
                        return {SeqKind::Scalar, BkT{}};
                    // Check if all elements are equal → Scalar
                    bool equal = true;
                    for (size_t i = 1; i < num_elements; ++i) {
                        if (cpu_vec[i] != cpu_vec[0]) {
                            equal = false;
                            break;
                        }
                    }
                    if (equal) return {SeqKind::Scalar, BkT{}};
                    // Check arithmetic sequence: delta = v[1] - v[0], verify all
                    auto d = static_cast<BkT>(cpu_vec[1]) - static_cast<BkT>(cpu_vec[0]);
                    for (size_t i = 2; i < num_elements; ++i) {
                        auto expected = static_cast<BkT>(cpu_vec[0]) + d * static_cast<BkT>(i);
                        if (static_cast<BkT>(cpu_vec[i]) != expected)
                            return {SeqKind::None, BkT{}};
                    }
                    return {SeqKind::Linear, d};
                }
            };
            auto [seq_kind, seq_delta] = var.is_constant() && has_raw ? detect_sequence() : std::pair{SeqKind::None, BkT{}};
            if (seq_kind == SeqKind::Scalar) {
                BkT scalar_val = num_elements >= 1u ? static_cast<BkT>(cpu_vec[0]) : BkT{};
                auto container = typename NNTensor<T>::container_type{
                    typename NNTensor<T>::container_type::scalar_tag_t{}, scalar_val, num_elements};
                tensor = std::make_unique<NNConstTensor<T>>(
                    std::move(tensor_shape), std::move(container), std::move(cpu_vec));
            } else if (seq_kind == SeqKind::Linear) {
                BkT start_val = static_cast<BkT>(cpu_vec[0]);
                auto container = typename NNTensor<T>::container_type{
                    typename NNTensor<T>::container_type::linear_tag_t{}, start_val, seq_delta, num_elements};
                tensor = std::make_unique<NNConstTensor<T>>(
                    std::move(tensor_shape), std::move(container), std::move(cpu_vec));
            } else {
                auto container = typename NNTensor<T>::container_type{num_elements};
                // Write to GPU container
                if (has_raw) {
                    for (uint i = 0; i < (uint)num_elements; ++i) {
                        if constexpr (std::is_same_v<BkT, FP16Quantized>) {
                            container[i].bits = def(cpu_vec[i].bits);
                        } else if constexpr (requires(BkT x) { x.bits; }) {
                            container[i].bits = def(static_cast<uint16_t>(cpu_vec[i].bits));
                        } else {
                            container[i] = static_cast<BkT>(cpu_vec[i]);
                        }
                    }
                }
                if (var.is_constant() && has_raw) {
                    tensor = std::make_unique<NNConstTensor<T>>(
                        std::move(tensor_shape), std::move(container), std::move(cpu_vec));
                } else {
                    tensor = std::make_unique<NNTensor<T>>(
                        std::move(tensor_shape), std::move(container));
                }
            }
        }
        if (var.is_constant()) {
            tensor->set_name("CONST_" + name);
        } else {
            tensor->set_name(name);
        }

        // Transfer ownership to tensor table
        tensor_table.own(name, std::move(tensor));
    });
}

void NetworkInstance::create_intermediate_tensors(
    TensorTable &tensor_table,
    onnx::Graph const &graph) {

    for (auto const &[name, var] : graph.get_variables()) {
        if (tensor_table.contains(name)) continue;
        create_tensor_for_var(tensor_table, var);
    }
}

// ==================== Operator Pre-creation ====================

NetworkInstance::OperatorList NetworkInstance::create_all_operators(
    onnx::Graph const &graph,
    onnx::OperatorSet &opset) {
    OperatorList ops;
    ops.reserve(graph.get_nodes().size());
    for (auto const &node : graph.get_nodes()) {
        ops.emplace_back(opset.create_operator(node.get_op_type(), node));
    }
    return ops;
}

// ==================== Last-Use Analysis ====================

LastUseMap NetworkInstance::build_last_use_map(
    onnx::Graph const &graph,
    TensorTable const &tensor_table,
    OperatorList const &ops) {

    LastUseMap last_use;
    auto const &nodes = graph.get_nodes();

    // Pass 1: record last node index where each variable is used as input
    for (size_t i = 0; i < nodes.size(); ++i) {
        for (auto const &in_var : nodes[i].get_inputs()) {
            auto const &vname = in_var.get().get_name();
            auto [it, inserted] = last_use.emplace(vname, i);
            if (!inserted && i > it->second) {
                it->second = i;
            }
        }
        // Also record outputs as produced at this index
        for (auto const &out_var : nodes[i].get_outputs()) {
            auto const &vname = out_var.get().get_name();
            last_use.emplace(vname, i);
        }
    }

    // Pass 1.5: Extend live ranges for variables referenced inside subgraphs.
    // If a node (e.g. If) contains subgraph attributes (then_branch / else_branch),
    // those subgraphs may reference variables from the parent graph.  The register
    // allocator only sees main-graph node inputs, so it would consider those
    // variables "dead" before the If node and recycle their memory.
    // We recursively collect all variable names referenced as node inputs inside
    // subgraphs that are NOT locally defined in those subgraphs, then extend
    // their last_use to the node index that owns the subgraph.
    {
        // Recursively collect external variable references from a subgraph.
        std::function<void(onnx::Graph const &, std::unordered_set<std::string> &)> collect_external_refs;
        collect_external_refs = [&](onnx::Graph const &sub_graph,
                                    std::unordered_set<std::string> &ext_refs) {
            auto const &local_vars = sub_graph.get_variables();
            for (auto const &sub_node : sub_graph.get_nodes()) {
                for (auto const &in_var : sub_node.get_inputs()) {
                    auto const &vname = in_var.get().get_name();
                    // If not locally defined in the subgraph, it's an external reference
                    if (local_vars.find(vname) == local_vars.end()) {
                        ext_refs.insert(vname);
                    }
                }
                // Recurse into nested subgraphs (e.g. nested If)
                for (auto const &[attr_name, attr] : sub_node.get_attributes()) {
                    auto attr_type = onnx::Attribute::type_of(attr);
                    if (attr_type == onnx::AttributeType::GRAPH) {
                        collect_external_refs(*attr.get<onnx::AttributeType::GRAPH>(), ext_refs);
                    } else if (attr_type == onnx::AttributeType::GRAPHS) {
                        for (auto const &g : attr.get<onnx::AttributeType::GRAPHS>()) {
                            collect_external_refs(*g, ext_refs);
                        }
                    }
                }
            }
        };

        for (size_t i = 0; i < nodes.size(); ++i) {
            for (auto const &[attr_name, attr] : nodes[i].get_attributes()) {
                auto attr_type = onnx::Attribute::type_of(attr);
                std::unordered_set<std::string> ext_refs;
                if (attr_type == onnx::AttributeType::GRAPH) {
                    collect_external_refs(*attr.get<onnx::AttributeType::GRAPH>(), ext_refs);
                } else if (attr_type == onnx::AttributeType::GRAPHS) {
                    for (auto const &g : attr.get<onnx::AttributeType::GRAPHS>()) {
                        collect_external_refs(*g, ext_refs);
                    }
                }
                // Extend last_use of all external references to this node's index
                for (auto const &vname : ext_refs) {
                    auto [it, inserted] = last_use.emplace(vname, i);
                    if (!inserted && i > it->second) {
                        it->second = i;
                    }
                }
            }
        }
    }

    // Pass 2: View dependency propagation REMOVED.
    // View operators (Slice/Gather/Split) no longer create views; they
    // perform explicit gather/copy. The input's live range does NOT need
    // to extend to cover view outputs' consumers.

    // Pass 3: mark graph outputs as never reclaimable
    for (auto const &var_ref : graph.get_outputs()) {
        last_use[var_ref.get().get_name()] = SIZE_MAX;
    }

    // Pass 4: mark constants and borrowed tensors as never reclaimable
    for (auto const &[vname, var] : graph.get_variables()) {
        if (var.is_constant()) {
            last_use[vname] = SIZE_MAX;
        }
    }
    for (auto const &[name, entry] : tensor_table) {
        if (entry.is_borrowed()) {
            last_use[name] = SIZE_MAX;
        }
    }

    return last_use;
}

AllocationPlan NetworkInstance::create_intermediate_tensors_pooled(
    TensorTable &tensor_table,
    onnx::Graph const &graph,
    LastUseMap const &last_use_map,
    OperatorList const &ops,
    std::vector<ExternalSlot> external_slots,
    std::vector<ITensor *> external_storages) {

    // First, create all constants and weights (they are never pooled)
    for (auto const &[name, var] : graph.get_variables()) {
        if (tensor_table.contains(name)) continue;
        if (var.is_constant() || var.is_trainable_weight() || !var.get_raw_data().empty()) {
            create_tensor_for_var(tensor_table, var);
        }
    }

    // Build the set of already-bound tensor names (externals, constants, weights)
    onnx::StringNodeMap<bool> already_bound;
    for (auto const &[name, entry] : tensor_table) {
        already_bound[name] = true;
    }

    // Phase 1: Run the graph-coloring register allocator with optional external slots.
    RegisterAllocator allocator(graph, last_use_map, ops, already_bound, {}, std::move(external_slots));
    auto plan = allocator.run();

    // Phase 2: Create physical storage arrays and all tensor variables.
    // For each color slot, we either borrow from an external storage (parent graph)
    // or create a new "phantom" Local DynamicArray with size_class capacity.
    // ALL members get their NNTensor with a View into the storage at offset 0.

    struct PhantomStorage {
        ITensor *storage_ptr = nullptr;        // Points to the actual storage (owned or borrowed)
        std::unique_ptr<ITensor> owned_storage;// Non-null only when we own the storage
    };
    std::vector<PhantomStorage> phantom_storages(plan.color_slots.size());

    for (size_t si = 0; si < plan.color_slots.size(); ++si) {
        auto const &slot = plan.color_slots[si];
        if (slot.members.empty()) continue;

        auto const &first_var = graph.get_var(slot.members[0]);
        auto slot_size = slot.size_class;

        // Check if this slot borrows from a parent-graph external storage
        if (slot.borrowed_slot_index != SIZE_MAX &&
            slot.borrowed_slot_index < external_storages.size() &&
            external_storages[slot.borrowed_slot_index] != nullptr) {
            // Borrow: just point to the parent's storage without taking ownership
            phantom_storages[si].storage_ptr = external_storages[slot.borrowed_slot_index];
        } else {
            // Own: create a new phantom storage tensor with slot_size capacity
            auto phantom_shape = typename ITensor::shape_type{static_cast<ITensor::size_type>(slot_size)};
            visit_onnx_dtype(first_var.get_dtype(), [&]<typename T>() {
                auto container = typename NNTensor<T>::container_type{slot_size};
                auto tensor = std::make_unique<NNTensor<T>>(
                    std::move(phantom_shape), std::move(container));
                tensor->set_name("_storage_" + std::to_string(si));
                phantom_storages[si].storage_ptr = tensor.get();
                phantom_storages[si].owned_storage = std::move(tensor);
            });
        }
    }

    // Now create all member tensors as Views into their slot's storage.
    for (size_t si = 0; si < plan.color_slots.size(); ++si) {
        auto const &slot = plan.color_slots[si];
        if (slot.members.empty()) continue;
        if (!phantom_storages[si].storage_ptr) continue;

        for (auto const &name : slot.members) {
            if (tensor_table.contains(name)) continue;

            auto const &var = graph.get_var(name);
            auto const &shape = var.get_shape();
            auto num_elements = compute_num_elements(shape);
            auto tensor_shape = to_tensor_shape(shape);

            visit_onnx_dtype(var.get_dtype(), [&]<typename T>() {
                auto &storage = static_cast<NNTensor<T> &>(*phantom_storages[si].storage_ptr);
                auto view_container = typename NNTensor<T>::container_type(
                    num_elements, storage.container());
                auto tensor = std::make_unique<NNTensor<T>>(
                    std::move(tensor_shape), std::move(view_container));
                tensor->set_name(name);
                tensor_table.own(name, std::move(tensor));
            });
        }
    }

    // Phase 2b: Transfer ownership of newly-created phantom storages into
    // tensor_table so the underlying DynamicArray objects remain alive.
    // Borrowed storages are NOT transferred (they live in the parent table).
    for (size_t si = 0; si < phantom_storages.size(); ++si) {
        if (!phantom_storages[si].owned_storage) continue;
        auto storage_name = "_phantom_storage_" + std::to_string(si);
        tensor_table.own(std::move(storage_name), std::move(phantom_storages[si].owned_storage));
    }

    // Phase 3: Create any remaining variables not covered by the allocation plan
    for (auto const &[name, var] : graph.get_variables()) {
        if (tensor_table.contains(name)) continue;
        create_tensor_for_var(tensor_table, var);
    }

    return plan;
}

void NetworkInstance::execute_operators(
    TensorTable &tensor_table,
    onnx::Graph const &graph,
    OperatorList &ops,
    LastUseMap const &last_use_map,
    Schedule const &schedule,
    AllocationPlan const *plan) {

    auto const &nodes = graph.get_nodes();
    size_t N = nodes.size();
    if (enable_logging_) {
        auto markdown = format_node_info_markdown(graph, N);
        // Create stats.md if it doesn't exist
        static bool file_created = false;
        if (!file_created && !std::filesystem::exists("./stats.md")) {
            std::ofstream create_file("./stats.md", std::ios::out);
            if (create_file) {
                create_file.close();
                file_created = true;
            }
        }
        std::ofstream stats_file("./stats.md", std::ios::app);
        if (stats_file) {
            stats_file << markdown;
            stats_file.close();
        }
    }
    // Save the parent execution context (in case of nested graphs)
    auto saved_ctx = exec_ctx_;

    // Set up execution context for this graph so that subgraph operators
    // (e.g. If) can inspect our allocation state.
    exec_ctx_.tensor_table = &tensor_table;
    exec_ctx_.plan = plan;
    exec_ctx_.last_use_map = &last_use_map;

    // If a virtual schedule is provided, execute in that order;
    // otherwise fall back to the original topological order.
    // TODO print more info
    for (size_t vi = 0; vi < N; ++vi) {
        size_t node_idx = (!schedule.empty() && schedule.size() == N) ? schedule[vi] : vi;
        auto const &node = nodes[node_idx];
        if (enable_logging_) {
            auto node_markdown = format_node_execution_markdown(node, node_idx, N);
            std::ofstream stats_file("./stats.md", std::ios::app);
            if (stats_file) {
                stats_file << node_markdown;
                stats_file.close();
                // LUISA_INFO("stats.md writed.");
            }
        }
        // Update current node index in execution context
        exec_ctx_.current_node_idx = node_idx;

        std::vector<std::reference_wrapper<ITensor>> op_inputs;
        std::vector<std::reference_wrapper<ITensor>> op_outputs;

        for (auto const &in_var : node.get_inputs()) {
            op_inputs.emplace_back(tensor_table.at(in_var.get().get_name()));
        }
        for (auto const &out_var : node.get_outputs()) {
            op_outputs.emplace_back(tensor_table.at(out_var.get().get_name()));
        }

        auto &op = ops[node_idx];
        op->set_environment(*this, tensor_table);
        if (op->need_outline()) {
            $outline_with_name(op->get_name()) {
                op->forward(op_inputs, op_outputs);
            };
        } else {
            op->forward(op_inputs, op_outputs);
        }
    }

    // Restore parent execution context
    exec_ctx_ = saved_ctx;
}

void NetworkInstance::forward_graph(
    TensorTable &tensor_table,
    onnx::Graph const &graph) {
    // Phase 0: Pre-create all operator instances.
    auto ops = create_all_operators(graph, model_.get_opset());

    // Phase 1: Build last-use map from the graph structure.
    auto last_use_map = build_last_use_map(graph, tensor_table, ops);

    // Phase 2: Create all tensors using pool-aware allocation.
    //          Returns the full allocation plan (with schedule and slot info).
    auto plan = create_intermediate_tensors_pooled(tensor_table, graph, last_use_map, ops);

    if (enable_logging_) {
        LUISA_INFO("=== Allocation Plan ===");
        LUISA_INFO("Total slots: {}", plan.color_slots.size());
        for (size_t si = 0; si < plan.color_slots.size(); ++si) {
            auto const &slot = plan.color_slots[si];
            LUISA_INFO("Slot {} (size={}, type={}): members = [{}]",
                       si, slot.size_class, slot.type_idx.name(),
                       [&]() {
                           std::string s;
                           for (size_t i = 0; i < slot.members.size(); ++i) {
                               if (i > 0) s += ", ";
                               s += slot.members[i];
                           }
                           return s;
                       }());
        }
        LUISA_INFO("Schedule: [{}]",
                   [&]() {
                       std::string s;
                       for (size_t i = 0; i < plan.schedule.size(); ++i) {
                           if (i > 0) s += ", ";
                           s += std::to_string(plan.schedule[i]);
                       }
                       return s;
                   }());
    }

    // Phase 3: Execute operators in the optimized schedule order.
    //          Pass the plan so that subgraph operators (e.g. If) can inspect
    //          parent-graph allocation state for safe storage reuse.
    execute_operators(tensor_table, graph, ops, last_use_map, plan.schedule, &plan);
}

void NetworkInstance::forward() {
    tensor_table_.clear();
    bind_external_tensors(tensor_table_, model_.get_graph());
    forward_graph(tensor_table_, model_.get_graph());
}

// ==================== Split Prepare / Execute ====================

PreparedGraph NetworkInstance::prepare_graph(
    TensorTable *parent_table,
    onnx::Graph const &graph,
    std::span<std::reference_wrapper<ITensor>> bound_outputs) {

    PreparedGraph pg;
    pg.tensor_table.set_parent(parent_table);

    // Bind output tensors
    auto const &graph_outs = graph.get_outputs();
    for (size_t i = 0; i < bound_outputs.size() && i < graph_outs.size(); ++i) {
        pg.tensor_table.bind(graph_outs[i].get().get_name(), bound_outputs[i].get());
    }

    // Phase 0: Pre-create all operator instances
    pg.ops = create_all_operators(graph, model_.get_opset());

    // Phase 1: Build last-use map
    pg.last_use_map = build_last_use_map(graph, pg.tensor_table, pg.ops);

    // Phase 2: Create all tensors using pool-aware allocation
    pg.plan = create_intermediate_tensors_pooled(pg.tensor_table, graph, pg.last_use_map, pg.ops);
    pg.schedule = pg.plan.schedule;

    return pg;
}

void NetworkInstance::execute_prepared_graph(
    PreparedGraph &prepared,
    onnx::Graph const &graph) {

    execute_operators(prepared.tensor_table, graph, prepared.ops,
                      prepared.last_use_map, prepared.schedule, &prepared.plan);
}

std::vector<PreparedGraph> NetworkInstance::prepare_exclusive_graphs(
    TensorTable *parent_table,
    std::vector<std::pair<onnx::Graph const *, std::span<std::reference_wrapper<ITensor>>>> const &branch_infos,
    std::vector<ITensor *> const &reusable_parent_storages) {

    std::vector<PreparedGraph> results;
    results.reserve(branch_infos.size());

    // ---- Build external slots from caller-provided reusable parent storages ----
    // The caller (e.g. If operator via collect_dead_parent_storages) has already
    // verified that these storages are dead at the current execution point and
    // safe for subgraph reuse.  We simply describe them as ExternalSlots.
    std::vector<ExternalSlot> parent_ext_slots;
    std::vector<ITensor *> parent_ext_storages;

    for (size_t i = 0; i < reusable_parent_storages.size(); ++i) {
        auto *storage = reusable_parent_storages[i];
        if (!storage) continue;
        ExternalSlot es;
        es.type_idx = storage->element_type();
        es.capacity = storage->size();
        es.external_index = parent_ext_storages.size();
        parent_ext_slots.push_back(es);
        parent_ext_storages.push_back(storage);
    }

    // For sibling branches (then/else), we also collect phantom storages
    // created by earlier branches so that later branches can reuse them.
    // These are safe to share because branches execute exclusively ($if/$else).
    std::vector<ExternalSlot> sibling_ext_slots;
    std::vector<ITensor *> sibling_ext_storages;

    for (size_t bi = 0; bi < branch_infos.size(); ++bi) {
        auto const &[graph_ptr, bound_outputs] = branch_infos[bi];
        auto const &graph = *graph_ptr;

        PreparedGraph pg;
        pg.tensor_table.set_parent(parent_table);

        // Bind output tensors
        auto const &graph_outs = graph.get_outputs();
        for (size_t i = 0; i < bound_outputs.size() && i < graph_outs.size(); ++i) {
            pg.tensor_table.bind(graph_outs[i].get().get_name(), bound_outputs[i].get());
        }

        // Phase 0: Pre-create all operator instances
        pg.ops = create_all_operators(graph, model_.get_opset());

        // Phase 1: Build last-use map
        pg.last_use_map = build_last_use_map(graph, pg.tensor_table, pg.ops);

        // Phase 2: Merge parent + sibling external slots, pass to allocator
        std::vector<ExternalSlot> combined_ext_slots;
        std::vector<ITensor *> combined_ext_storages;
        combined_ext_slots.reserve(parent_ext_slots.size() + sibling_ext_slots.size());
        combined_ext_storages.reserve(parent_ext_storages.size() + sibling_ext_storages.size());

        for (auto &ps : parent_ext_slots) {
            combined_ext_slots.push_back(ps);
        }
        for (auto *ptr : parent_ext_storages) {
            combined_ext_storages.push_back(ptr);
        }
        for (auto &ss : sibling_ext_slots) {
            combined_ext_slots.push_back(ss);
        }
        for (auto *ptr : sibling_ext_storages) {
            combined_ext_storages.push_back(ptr);
        }

        pg.plan = create_intermediate_tensors_pooled(
            pg.tensor_table, graph, pg.last_use_map, pg.ops,
            std::move(combined_ext_slots), std::move(combined_ext_storages));
        pg.schedule = pg.plan.schedule;

        // Collect newly-created (owned) phantom storages from this branch
        // and add them as sibling external slots for subsequent branches.
        for (auto &[name, entry] : pg.tensor_table) {
            if (name.find("_phantom_storage_") == 0 && entry.is_owned()) {
                auto &storage = entry.get();
                size_t ext_idx = parent_ext_storages.size() + sibling_ext_storages.size();
                ExternalSlot es;
                es.type_idx = storage.element_type();
                es.capacity = storage.size();
                es.external_index = ext_idx;
                sibling_ext_slots.push_back(es);
                sibling_ext_storages.push_back(&storage);
            }
        }

        results.push_back(std::move(pg));
    }

    return results;
}

std::vector<ITensor *> NetworkInstance::collect_dead_parent_storages(
    TensorTable &tensor_table,
    AllocationPlan const &plan,
    LastUseMap const &last_use_map,
    size_t node_idx) {

    std::vector<ITensor *> dead_storages;

    // For each color slot in the allocation plan, check if ALL of its member
    // tensors have last_use < node_idx (i.e. they are "dead" at the current If node).
    // If so, the phantom storage backing this slot can be safely lent to subgraphs.
    for (size_t si = 0; si < plan.color_slots.size(); ++si) {
        auto const &slot = plan.color_slots[si];
        if (slot.members.empty()) continue;
        // Skip slots that are themselves borrowed from an outer scope
        if (slot.borrowed_slot_index != SIZE_MAX) continue;

        bool all_dead = true;
        for (auto const &member : slot.members) {
            auto lu_it = last_use_map.find(member);
            if (lu_it == last_use_map.end()) continue;
            // SIZE_MAX means "never reclaimable" (graph output, constant, etc.)
            if (lu_it->second == SIZE_MAX || lu_it->second >= node_idx) {
                all_dead = false;
                break;
            }
        }

        if (all_dead) {
            // Find the corresponding phantom storage in the tensor table
            auto storage_name = "_phantom_storage_" + std::to_string(si);
            if (tensor_table.contains_local(storage_name)) {
                auto &entry = tensor_table.entry(storage_name);
                if (entry.is_owned()) {
                    dead_storages.push_back(&entry.get());
                }
            }
        }
    }

    return dead_storages;
}

}// namespace lcml::onnx
