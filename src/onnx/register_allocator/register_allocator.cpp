#include "onnx/register_allocator/register_allocator.h"
#include "onnx/operators/common.h"

#include <algorithm>
#include <numeric>
#include <cassert>
#include <queue>
#include <luisa/core/logging.h>

namespace lcml::onnx {

RegisterAllocator::RegisterAllocator(
    onnx::Graph const &graph,
    LastUseMap const &last_use_map,
    OperatorList const &ops,
    onnx::StringNodeMap<bool> const &already_bound,
    RegAllocConfig const &config,
    std::vector<ExternalSlot> external_slots)
    : graph_(graph), orig_last_use_map_(last_use_map),
      ops_(ops), already_bound_(already_bound), config_(config),
      external_slots_(std::move(external_slots)) {}

AllocationPlan RegisterAllocator::run() {
    compute_virtual_order();
    init_units();
    apply_inplace_coalescing();
    compute_live_ranges();
    normalize_size_classes();
    build_interference_graph();
    greedy_color();
    return build_plan();
}

// ================================================================
// Helpers
// ================================================================

bool RegisterAllocator::is_skip_var(std::string const &vn) const {
    if (already_bound_.find(vn) != already_bound_.end()) return true;
    auto it = orig_last_use_map_.find(vn);
    if (it == orig_last_use_map_.end()) return true;
    if (it->second == SIZE_MAX) return true;
    return false;
}

ElementCount RegisterAllocator::compute_num_elements(std::vector<size_t> const &shape) {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), ElementCount{1}, std::multiplies<>{});
}

// ================================================================
// Phase 0: Compute virtual node ordering to minimize register pressure.
//
// Topological sort with "earliest deadline first" heuristic:
// prioritize nodes that free the most memory (size-weighted).
// ================================================================

// Recursively collect external variable references from subgraphs.
void RegisterAllocator::collect_external_refs_recursive(
    onnx::Graph const &sub_graph,
    std::unordered_set<std::string> &ext_refs) {
    auto const &local_vars = sub_graph.get_variables();
    for (auto const &sub_node : sub_graph.get_nodes()) {
        for (auto const &in_var : sub_node.get_inputs()) {
            auto const &vname = in_var.get().get_name();
            if (local_vars.find(vname) == local_vars.end()) {
                ext_refs.insert(vname);
            }
        }
        for (auto const &[attr_name, attr] : sub_node.get_attributes()) {
            auto attr_type = onnx::Attribute::type_of(attr);
            if (attr_type == onnx::AttributeType::GRAPH) {
                collect_external_refs_recursive(*attr.get<onnx::AttributeType::GRAPH>(), ext_refs);
            } else if (attr_type == onnx::AttributeType::GRAPHS) {
                for (auto const &g : attr.get<onnx::AttributeType::GRAPHS>()) {
                    collect_external_refs_recursive(*g, ext_refs);
                }
            }
        }
    }
}

void RegisterAllocator::build_producer_map(SchedulingContext &ctx) const {
    auto const &nodes = graph_.get_nodes();
    for (size_t i = 0; i < ctx.N; ++i) {
        for (auto const &out_var : nodes[i].get_outputs()) {
            ctx.producer[out_var.get().get_name()] = i;
        }
    }
}

void RegisterAllocator::build_ext_input_names(SchedulingContext &ctx) const {
    auto const &nodes = graph_.get_nodes();
    ctx.ext_input_names.resize(ctx.N);

    for (size_t i = 0; i < ctx.N; ++i) {
        std::unordered_set<std::string> ext_refs;
        for (auto const &[attr_name, attr] : nodes[i].get_attributes()) {
            auto attr_type = onnx::Attribute::type_of(attr);
            if (attr_type == onnx::AttributeType::GRAPH) {
                collect_external_refs_recursive(*attr.get<onnx::AttributeType::GRAPH>(), ext_refs);
            } else if (attr_type == onnx::AttributeType::GRAPHS) {
                for (auto const &g : attr.get<onnx::AttributeType::GRAPHS>()) {
                    collect_external_refs_recursive(*g, ext_refs);
                }
            }
        }
        // Remove names that are direct inputs (already handled)
        for (auto const &in_var : nodes[i].get_inputs()) {
            ext_refs.erase(in_var.get().get_name());
        }
        ctx.ext_input_names[i].assign(ext_refs.begin(), ext_refs.end());
    }
}

void RegisterAllocator::build_dag_edges(SchedulingContext &ctx) const {
    auto const &nodes = graph_.get_nodes();
    ctx.successors.resize(ctx.N);
    ctx.in_degree.assign(ctx.N, 0);

    for (size_t i = 0; i < ctx.N; ++i) {
        for (auto const &in_var : nodes[i].get_inputs()) {
            auto pit = ctx.producer.find(in_var.get().get_name());
            if (pit != ctx.producer.end()) {
                size_t from = pit->second;
                if (from != i && ctx.successors[from].insert(i).second) {
                    ctx.in_degree[i]++;
                }
            }
        }
        // Add implicit dependencies from subgraph external references
        for (auto const &vname : ctx.ext_input_names[i]) {
            auto pit = ctx.producer.find(vname);
            if (pit != ctx.producer.end()) {
                size_t from = pit->second;
                if (from != i && ctx.successors[from].insert(i).second) {
                    ctx.in_degree[i]++;
                }
            }
        }
    }
}

void RegisterAllocator::count_remaining_consumers(SchedulingContext &ctx) const {
    auto const &nodes = graph_.get_nodes();
    for (size_t i = 0; i < ctx.N; ++i) {
        for (auto const &in_var : nodes[i].get_inputs()) {
            auto const &vn = in_var.get().get_name();
            if (!is_skip_var(vn)) {
                ctx.remaining_consumers[vn]++;
            }
        }
        for (auto const &vn : ctx.ext_input_names[i]) {
            if (!is_skip_var(vn)) {
                ctx.remaining_consumers[vn]++;
            }
        }
    }
}

void RegisterAllocator::build_var_size_map(SchedulingContext &ctx) const {
    for (auto const &[vname, var] : graph_.get_variables()) {
        auto const &shape = var.get_shape();
        if (!shape.empty()) {
            ctx.var_size[vname] = std::accumulate(shape.begin(), shape.end(),
                                                  size_t{1}, std::multiplies<>{});
        }
    }
}

void RegisterAllocator::compute_node_heights(SchedulingContext &ctx) const {
    size_t N = ctx.N;
    ctx.height.assign(N, 0);

    std::vector<size_t> out_degree(N, 0);
    std::vector<std::unordered_set<size_t>> predecessors(N);
    for (size_t i = 0; i < N; ++i) {
        for (auto s : ctx.successors[i]) {
            predecessors[s].insert(i);
            out_degree[i]++;
        }
    }

    std::queue<size_t> q;
    for (size_t i = 0; i < N; ++i) {
        if (out_degree[i] == 0) q.push(i);
    }
    auto rev_out = out_degree;
    while (!q.empty()) {
        size_t n = q.front();
        q.pop();
        for (auto p : predecessors[n]) {
            ctx.height[p] = std::max(ctx.height[p], ctx.height[n] + 1);
            if (--rev_out[p] == 0) q.push(p);
        }
    }
}

// Compute freed memory size for a single variable being consumed as last use.
static int64_t freed_memory_for_var(
    std::string const &vn,
    bool skip,
    onnx::StringNodeMap<size_t> const &remaining_consumers,
    onnx::StringNodeMap<size_t> const &var_size) {
    if (skip) return 0;
    auto rc_it = remaining_consumers.find(vn);
    if (rc_it != remaining_consumers.end() && rc_it->second == 1) {
        auto sz_it = var_size.find(vn);
        return static_cast<int64_t>(sz_it != var_size.end() ? sz_it->second : 1);
    }
    return 0;
}

void RegisterAllocator::schedule_nodes(SchedulingContext &ctx) {
    auto const &nodes = graph_.get_nodes();
    size_t N = ctx.N;

    std::vector<size_t> ready;
    for (size_t i = 0; i < N; ++i) {
        if (ctx.in_degree[i] == 0) ready.push_back(i);
    }

    schedule_.clear();
    schedule_.reserve(N);
    std::vector<bool> scheduled(N, false);

    while (!ready.empty()) {
        size_t best_idx = 0;
        int64_t best_net_free = std::numeric_limits<int64_t>::min();
        size_t best_height = 0;

        for (size_t ri = 0; ri < ready.size(); ++ri) {
            size_t ni = ready[ri];
            auto const &node = nodes[ni];

            // Compute freed memory when this node consumes its last-use inputs
            int64_t free_size = 0;
            for (auto const &in_var : node.get_inputs()) {
                free_size += freed_memory_for_var(
                    in_var.get().get_name(), is_skip_var(in_var.get().get_name()),
                    ctx.remaining_consumers, ctx.var_size);
            }
            for (auto const &vn : ctx.ext_input_names[ni]) {
                free_size += freed_memory_for_var(
                    vn, is_skip_var(vn), ctx.remaining_consumers, ctx.var_size);
            }

            // Compute new memory from outputs
            int64_t new_size = 0;
            for (auto const &out_var : node.get_outputs()) {
                auto const &vn = out_var.get().get_name();
                if (!is_skip_var(vn)) {
                    auto sz_it = ctx.var_size.find(vn);
                    new_size += static_cast<int64_t>(sz_it != ctx.var_size.end() ? sz_it->second : 1);
                }
            }

            int64_t net_free = free_size - new_size;
            size_t h = ctx.height[ni];
            if (net_free > best_net_free ||
                (net_free == best_net_free && h > best_height)) {
                best_net_free = net_free;
                best_height = h;
                best_idx = ri;
            }
        }

        size_t chosen = ready[best_idx];
        ready[best_idx] = ready.back();
        ready.pop_back();

        schedule_.push_back(chosen);
        scheduled[chosen] = true;

        // Update remaining consumers (including subgraph external refs)
        for (auto const &in_var : nodes[chosen].get_inputs()) {
            auto rc_it = ctx.remaining_consumers.find(in_var.get().get_name());
            if (rc_it != ctx.remaining_consumers.end() && rc_it->second > 0) {
                rc_it->second--;
            }
        }
        for (auto const &vn : ctx.ext_input_names[chosen]) {
            auto rc_it = ctx.remaining_consumers.find(vn);
            if (rc_it != ctx.remaining_consumers.end() && rc_it->second > 0) {
                rc_it->second--;
            }
        }

        // Unlock successors
        for (auto s : ctx.successors[chosen]) {
            if (--ctx.in_degree[s] == 0) ready.push_back(s);
        }
    }

    // Append any unscheduled nodes (safety)
    for (size_t i = 0; i < N; ++i) {
        if (!scheduled[i]) schedule_.push_back(i);
    }

    // Build reverse mapping
    orig_to_virtual_.resize(N);
    for (size_t vi = 0; vi < N; ++vi) {
        orig_to_virtual_[schedule_[vi]] = vi;
    }
}

void RegisterAllocator::remap_last_use_map(SchedulingContext const &ctx) {
    auto const &nodes = graph_.get_nodes();
    size_t N = ctx.N;

    vlast_use_map_.clear();
    for (size_t orig_i = 0; orig_i < N; ++orig_i) {
        size_t vi = orig_to_virtual_[orig_i];
        auto const &node = nodes[orig_i];

        auto emplace_or_update = [](LastUseMap &m, std::string const &vn, size_t vi) {
            auto [it, inserted] = m.emplace(vn, vi);
            if (!inserted && vi > it->second) it->second = vi;
        };

        for (auto const &in_var : node.get_inputs()) {
            emplace_or_update(vlast_use_map_, in_var.get().get_name(), vi);
        }
        for (auto const &vn : ctx.ext_input_names[orig_i]) {
            emplace_or_update(vlast_use_map_, vn, vi);
        }
        for (auto const &out_var : node.get_outputs()) {
            auto const &vn = out_var.get().get_name();
            vlast_use_map_.emplace(vn, vi);
        }
    }
}

void RegisterAllocator::mark_permanent_variables() {
    // Mark graph outputs and constants as SIZE_MAX (never reclaimable)
    for (auto const &var_ref : graph_.get_outputs()) {
        vlast_use_map_[var_ref.get().get_name()] = SIZE_MAX;
    }
    for (auto const &[vname, var] : graph_.get_variables()) {
        if (var.is_constant()) vlast_use_map_[vname] = SIZE_MAX;
    }
    for (auto const &[name, _] : already_bound_) {
        auto oit = orig_last_use_map_.find(name);
        if (oit != orig_last_use_map_.end() && oit->second == SIZE_MAX) {
            vlast_use_map_[name] = SIZE_MAX;
        }
    }
}

void RegisterAllocator::compute_virtual_order() {
    auto const &nodes = graph_.get_nodes();
    size_t N = nodes.size();

    if (N == 0) {
        schedule_.clear();
        orig_to_virtual_.clear();
        return;
    }

    SchedulingContext ctx;
    ctx.N = N;

    build_producer_map(ctx);
    build_ext_input_names(ctx);
    build_dag_edges(ctx);
    count_remaining_consumers(ctx);
    build_var_size_map(ctx);
    compute_node_heights(ctx);
    schedule_nodes(ctx);
    remap_last_use_map(ctx);
    mark_permanent_variables();
}

// ================================================================
// Phase 0.5: Initialize allocation units using virtual indices
// ================================================================

void RegisterAllocator::init_units() {
    auto const &nodes = graph_.get_nodes();

    // Record virtual def_node for each variable
    onnx::StringNodeMap<NodeIndex> def_map;
    for (size_t orig_i = 0; orig_i < nodes.size(); ++orig_i) {
        size_t vi = orig_to_virtual_.empty() ? orig_i : orig_to_virtual_[orig_i];
        for (auto const &out_var : nodes[orig_i].get_outputs()) {
            auto const &vname = out_var.get().get_name();
            auto [it, inserted] = def_map.emplace(vname, vi);
            if (!inserted && vi < it->second) it->second = vi;
        }
    }

    for (auto const &[name, var] : graph_.get_variables()) {
        auto const &elem_typeid = onnx_dtype_to_typeid(var.get_dtype());
        auto num_elements = compute_num_elements(var.get_shape());

        NodeIndex def_node = 0;
        if (auto it = def_map.find(name); it != def_map.end()) def_node = it->second;

        NodeIndex last_use = 0;
        if (auto it = vlast_use_map_.find(name); it != vlast_use_map_.end()) last_use = it->second;

        AllocationUnit unit(name, std::type_index(elem_typeid), num_elements, def_node, last_use);

        if (var.is_constant() || var.is_trainable_weight() || !var.get_raw_data().empty()) {
            unit.skip_allocation = true;
        }
        if (already_bound_.find(name) != already_bound_.end()) {
            unit.skip_allocation = true;
        }
        if (last_use == SIZE_MAX) {
            unit.skip_allocation = true;
        }

        units_[name] = std::move(unit);
        uf_.init(name);
    }
}

// ================================================================
// Phase A: Universal inplace coalescing
//
// Requirements for merging output into input's storage:
//   1. input last_use <= current virtual node
//   2. all UF group members' last_use <= current virtual node
//   3. input num_elements >= output num_elements
//   4. merged range doesn't exceed threshold
//   5. cost-benefit: inflation doesn't outweigh savings
// ================================================================

bool RegisterAllocator::check_view_eligibility(
    onnx::Node const &node, Operator const &op) const {
    for (size_t oi = 0; oi < node.get_outputs().size(); ++oi) {
        if (op.is_output_view(oi, node) && !node.get_inputs().empty()) {
            auto const &oname = node.get_outputs()[oi].get().get_name();
            auto const &iname = node.get_inputs()[0].get().get_name();
            auto oit = units_.find(oname);
            auto iit = units_.find(iname);
            if (oit != units_.end() && iit != units_.end() &&
                oit->second.num_elements == iit->second.num_elements) {
                return true;
            }
        }
    }
    return false;
}

RegisterAllocator::InplaceCandidateResult RegisterAllocator::evaluate_inplace_candidate(
    onnx::Node const &node, Operator const &op,
    size_t vi, size_t out_idx) const {

    InplaceCandidateResult result;

    auto const &out_name = node.get_outputs()[out_idx].get().get_name();
    auto out_it = units_.find(out_name);
    if (out_it == units_.end()) return result;
    auto const &out_unit = out_it->second;
    if (out_unit.skip_allocation) return result;

    auto out_elems = out_unit.num_elements;

    for (size_t ii = 0; ii < node.get_inputs().size(); ++ii) {
        auto const &in_name = node.get_inputs()[ii].get().get_name();
        auto in_it = units_.find(in_name);
        if (in_it == units_.end()) continue;
        auto const &in_unit = in_it->second;
        if (in_unit.skip_allocation || in_unit.type_idx != out_unit.type_idx) continue;

        auto in_elems = in_unit.num_elements;
        if (in_elems < out_elems) continue;

        bool is_view_out = op.is_output_view(out_idx, node);
        if (is_view_out) {
            if (ii != 0 || in_elems != out_elems) continue;
        }

        auto in_root = uf_.find(in_name);
        auto out_root = uf_.find(out_name);
        if (in_root == out_root) continue;

        // Cost-benefit analysis: scan UF group for root stats
        ElementCount current_root_max = 0;
        NodeIndex root_def = SIZE_MAX;
        NodeIndex root_last_use = 0;
        for (auto const &[uname, uunit] : units_) {
            if (uunit.skip_allocation || uf_.find(uname) != in_root) continue;
            current_root_max = std::max(current_root_max, uunit.num_elements);
            root_def = std::min(root_def, uunit.def_node);
            auto lu_it = vlast_use_map_.find(uname);
            if (lu_it != vlast_use_map_.end() && lu_it->second != SIZE_MAX) {
                root_last_use = std::max(root_last_use, lu_it->second);
            }
        }

        ElementCount new_root_max = std::max(current_root_max, out_elems);
        int64_t inflation_cost = static_cast<int64_t>(new_root_max) -
                                   static_cast<int64_t>(current_root_max);
        int64_t save_benefit = static_cast<int64_t>(out_elems);
        int64_t net_benefit = save_benefit - inflation_cost;
        int64_t inflation_tolerance =
            (out_elems <= config_.small_tensor_threshold) ? config_.small_tensor_inflation_tolerance : 0;
        if (net_benefit + inflation_tolerance <= 0) continue;

        // Range extension limit for large roots
        {
            auto out_lu_it = vlast_use_map_.find(out_name);
            NodeIndex out_last = (out_lu_it != vlast_use_map_.end()) ? out_lu_it->second : vi;
            NodeIndex merged_def = std::min(root_def, out_unit.def_node);
            NodeIndex merged_last = std::max(root_last_use, out_last);
            NodeIndex merged_range = (merged_last >= merged_def) ? (merged_last - merged_def + 1) : 1;
            NodeIndex current_range = (root_last_use >= root_def) ? (root_last_use - root_def + 1) : 1;

            if (new_root_max >= config_.large_root_threshold) {
                ElementCount max_range = std::max(config_.large_root_min_range,
                                                  config_.large_root_range_budget / new_root_max);
                if (merged_range > max_range && merged_range > current_range) continue;
            }
        }

        // Safety: input's last_use must be <= vi (skip for views)
        if (!is_view_out) {
            auto self_lu_it = vlast_use_map_.find(in_name);
            if (self_lu_it == vlast_use_map_.end() || self_lu_it->second > vi) continue;

            bool group_safe = true;
            for (auto const &[uname, uunit] : units_) {
                if (uunit.skip_allocation || uf_.find(uname) != in_root) continue;
                auto mlu_it = vlast_use_map_.find(uname);
                if (mlu_it != vlast_use_map_.end() && mlu_it->second > vi) {
                    group_safe = false;
                    break;
                }
            }
            if (!group_safe) continue;
        }

        if (in_elems > result.best_elems ||
            (in_elems == result.best_elems && net_benefit > result.best_net_benefit)) {
            result.best_elems = in_elems;
            result.best_input = in_name;
            result.best_net_benefit = net_benefit;
            result.accepted = true;
        }
    }

    return result;
}

void RegisterAllocator::apply_inplace_coalescing() {
    auto const &nodes = graph_.get_nodes();

    for (size_t vi = 0; vi < schedule_.size(); ++vi) {
        size_t ni = schedule_[vi];
        auto const &node = nodes[ni];
        auto const &op = *ops_[ni];

        if (node.get_inputs().empty() || node.get_outputs().empty()) continue;

        bool is_view_eligible = check_view_eligibility(node, op);
        if (!op.can_operate_inplace() && !is_view_eligible) continue;

        // Try to inplace each output to one of the inputs
        for (size_t out_idx = 0; out_idx < node.get_outputs().size(); ++out_idx) {
            auto const &out_name = node.get_outputs()[out_idx].get().get_name();
            auto out_it = units_.find(out_name);
            if (out_it == units_.end()) continue;
            auto &out_unit = out_it->second;
            if (out_unit.skip_allocation) continue;

            auto candidate = evaluate_inplace_candidate(node, op, vi, out_idx);
            if (candidate.accepted) {
                out_unit.is_inplace = true;
                inplace_source_map_[out_name] = candidate.best_input;
                uf_.unite(candidate.best_input, out_name);
            }
        }
    }
}

// ================================================================
// Phase B: Compute live ranges for each union-find root
// ================================================================

void RegisterAllocator::compute_live_ranges() {
    root_infos_.clear();

    for (auto &[name, unit] : units_) {
        if (unit.skip_allocation) continue;

        auto root = uf_.find(name);
        auto it = root_infos_.find(root);
        if (it == root_infos_.end()) {
            RootInfo ri;
            ri.type_idx = unit.type_idx;
            ri.max_elements = unit.num_elements;
            ri.merged_def = unit.def_node;
            ri.merged_last_use = unit.last_use_node;
            ri.members.push_back(name);
            root_infos_[std::string{root}] = std::move(ri);
        } else {
            auto &ri = it->second;
            ri.max_elements = std::max(ri.max_elements, unit.num_elements);
            ri.merged_def = std::min(ri.merged_def, unit.def_node);
            auto lu_it = vlast_use_map_.find(name);
            size_t lu = (lu_it != vlast_use_map_.end()) ? lu_it->second : unit.last_use_node;
            if (lu == SIZE_MAX) {
                ri.skip = true;
            } else {
                ri.merged_last_use = std::max(ri.merged_last_use, lu);
            }
            ri.members.push_back(name);
        }
    }

    // Update merged_last_use from vlast_use_map for all members
    for (auto &[root, ri] : root_infos_) {
        for (auto const &member : ri.members) {
            auto lu_it = vlast_use_map_.find(member);
            if (lu_it != vlast_use_map_.end()) {
                if (lu_it->second == SIZE_MAX) {
                    ri.skip = true;
                } else {
                    ri.merged_last_use = std::max(ri.merged_last_use, lu_it->second);
                }
            }
        }
    }
}

// ================================================================
// Phase C: Use raw max_elements as size_class (no rounding)
// ================================================================

void RegisterAllocator::normalize_size_classes() {
    for (auto &[root, ri] : root_infos_) {
        ri.size_class = ri.max_elements;
    }
}

// ================================================================
// Phase D: Build interference graph
// ================================================================

void RegisterAllocator::build_interference_graph() {
    adj_.clear();

    std::vector<std::string> active_roots;
    for (auto const &[root, ri] : root_infos_) {
        if (ri.skip) continue;
        active_roots.push_back(root);
        adj_[root];
    }

    // Debug: print live ranges
    LUISA_INFO("=== Live Ranges ===");
    for (auto const &root : active_roots) {
        auto const &ri = root_infos_[root];
        LUISA_INFO("  {}: def={}, last_use={}, size={}", root, ri.merged_def, ri.merged_last_use, ri.max_elements);
    }

    for (size_t i = 0; i < active_roots.size(); ++i) {
        auto const &ri = root_infos_[active_roots[i]];
        for (size_t j = i + 1; j < active_roots.size(); ++j) {
            auto const &rj = root_infos_[active_roots[j]];
            if (ri.type_idx != rj.type_idx) continue;
            bool interfere = ri.merged_def <= rj.merged_last_use &&
                             rj.merged_def <= ri.merged_last_use;
            if (interfere) {
                LUISA_INFO("  Interference: {} <-> {}", active_roots[i], active_roots[j]);
                adj_[active_roots[i]].insert(active_roots[j]);
                adj_[active_roots[j]].insert(active_roots[i]);
            }
        }
    }
}

// ================================================================
// Phase E: Size-aware greedy graph coloring
//
// 1. Process roots in descending size order (large first)
// 2. For each root, pick color minimizing storage increase (best-fit)
// 3. Among zero-cost colors: large roots prefer smallest color ID
//    (interval-optimal), small roots prefer largest slot (pack into gaps)
// 4. Local search: single moves, pairwise swaps, dissolution, absorption
// ================================================================

bool RegisterAllocator::can_place_root_in_color(
    std::string const &root, ColorId color) const {
    auto adj_it = adj_.find(root);
    if (adj_it != adj_.end()) {
        for (auto const &nb : adj_it->second) {
            auto cit = color_map_.find(nb);
            if (cit != color_map_.end() && cit->second == color) return false;
        }
    }
    return true;
}

void RegisterAllocator::recompute_color_slot_size(
    ColoringContext &ctx, ColorId color) const {
    ElementCount mx = 0;
    for (auto const &r : ctx.color_members[color]) {
        mx = std::max(mx, root_infos_.at(r).size_class);
    }
    ctx.color_slot_size[color] = mx;
}

void RegisterAllocator::initial_greedy_coloring(ColoringContext &ctx) {
    // Sort: size descending, then def ascending, then last_use ascending
    auto const &infos = root_infos_;
    std::sort(ctx.active_roots.begin(), ctx.active_roots.end(),
              [&infos](std::string const &a, std::string const &b) {
                  auto sa = infos.at(a).size_class, sb = infos.at(b).size_class;
                  if (sa != sb) return sa > sb;
                  auto da = infos.at(a).merged_def, db = infos.at(b).merged_def;
                  if (da != db) return da < db;
                  return infos.at(a).merged_last_use < infos.at(b).merged_last_use;
              });

    for (auto const &root : ctx.active_roots) {
        ElementCount my_size = root_infos_[root].size_class;

        std::unordered_set<ColorId> forbidden;
        if (adj_.count(root)) {
            for (auto const &nb : adj_.at(root)) {
                auto cit = color_map_.find(nb);
                if (cit != color_map_.end()) forbidden.insert(cit->second);
            }
        }

        ColorId best_color = SIZE_MAX;
        ElementCount best_cost = SIZE_MAX, best_slot_size = 0;

        for (ColorId c = 0; c < num_colors_; ++c) {
            if (forbidden.count(c)) continue;
            // For external slots, enforce type matching (owned slots inherit
            // their type from the first root placed in them, but external
            // slots already have a fixed type from the parent graph).
            if (ctx.color_external_index[c] != SIZE_MAX) {
                auto const &ri = root_infos_[root];
                // Find the external slot's type from external_slots_
                bool type_ok = false;
                for (auto const &ext : external_slots_) {
                    if (ext.external_index == ctx.color_external_index[c]) {
                        type_ok = (ext.type_idx == ri.type_idx);
                        break;
                    }
                }
                if (!type_ok) continue;
            }
            ElementCount cur = ctx.color_slot_size[c];
            ElementCount cost = (my_size > cur) ? (my_size - cur) : 0;

            bool better = false;
            if (cost < best_cost) {
                better = true;
            } else if (cost == best_cost && cost == 0) {
                better = (my_size >= config_.color_large_size_threshold) ? (c < best_color) : (cur > best_slot_size);
            }
            if (better) {
                best_cost = cost;
                best_slot_size = cur;
                best_color = c;
            }
        }

        if (best_color == SIZE_MAX || best_cost > my_size) {
            best_color = num_colors_++;
            ctx.color_slot_size.push_back(my_size);
            ctx.color_members.emplace_back();
            ctx.color_external_index.push_back(SIZE_MAX);// owned slot
        } else {
            // Only grow the slot if it is not an external slot.
            // External slots have fixed capacity; we never enlarge them.
            if (ctx.color_external_index[best_color] == SIZE_MAX) {
                ctx.color_slot_size[best_color] = std::max(ctx.color_slot_size[best_color], my_size);
            }
            // (For external slots the slot_size is already >= my_size because
            //  the cost check above ensures my_size <= ext.capacity.)
        }
        color_map_[root] = best_color;
        ctx.color_members[best_color].insert(root);
    }
}

bool RegisterAllocator::try_single_root_moves(ColoringContext &ctx) {
    bool improved = false;

    for (auto const &root : ctx.active_roots) {
        ColorId cur_color = color_map_[root];
        ElementCount my_size = root_infos_[root].size_class;

        ElementCount slot_without_me = 0;
        for (auto const &other : ctx.color_members[cur_color]) {
            if (other != root)
                slot_without_me = std::max(slot_without_me, root_infos_[other].size_class);
        }
        int64_t remove_savings = static_cast<int64_t>(ctx.color_slot_size[cur_color]) -
                                   static_cast<int64_t>(slot_without_me);

        ColorId best_target = cur_color;
        int64_t best_net = 0;

        for (ColorId c = 0; c < num_colors_; ++c) {
            if (c == cur_color || !can_place_root_in_color(root, c)) continue;
            // Do not move into an external slot if it would need to grow
            if (ctx.color_external_index[c] != SIZE_MAX && my_size > ctx.color_slot_size[c]) continue;
            int64_t add_cost = static_cast<int64_t>(std::max(ctx.color_slot_size[c], my_size)) -
                                 static_cast<int64_t>(ctx.color_slot_size[c]);
            int64_t net = remove_savings - add_cost;
            if (net > best_net) {
                best_net = net;
                best_target = c;
            }
        }

        if (best_target != cur_color && best_net > 0) {
            ctx.color_members[cur_color].erase(root);
            ctx.color_slot_size[cur_color] = slot_without_me;
            color_map_[root] = best_target;
            ctx.color_members[best_target].insert(root);
            // Only grow owned slots; external slots have fixed capacity
            if (ctx.color_external_index[best_target] == SIZE_MAX) {
                ctx.color_slot_size[best_target] = std::max(ctx.color_slot_size[best_target], my_size);
            }
            improved = true;
        }
    }

    return improved;
}

bool RegisterAllocator::try_pairwise_swaps(ColoringContext &ctx) {
    for (size_t i = 0; i < ctx.active_roots.size(); ++i) {
        auto const &r1 = ctx.active_roots[i];
        ColorId c1 = color_map_[r1];
        ElementCount s1 = root_infos_[r1].size_class;

        for (size_t j = i + 1; j < ctx.active_roots.size(); ++j) {
            auto const &r2 = ctx.active_roots[j];
            ColorId c2 = color_map_[r2];
            if (c1 == c2) continue;
            if (root_infos_[r1].type_idx != root_infos_[r2].type_idx) continue;
            ElementCount s2 = root_infos_[r2].size_class;

            // Check placement feasibility
            bool ok1 = true, ok2 = true;
            if (adj_.count(r1)) {
                for (auto const &nb : adj_.at(r1)) {
                    if (nb != r2 && color_map_.count(nb) && color_map_[nb] == c2) {
                        ok1 = false;
                        break;
                    }
                }
            }
            if (!ok1) continue;
            if (adj_.count(r2)) {
                for (auto const &nb : adj_.at(r2)) {
                    if (nb != r1 && color_map_.count(nb) && color_map_[nb] == c1) {
                        ok2 = false;
                        break;
                    }
                }
            }
            if (!ok2) continue;

            ElementCount c1_without = 0, c2_without = 0;
            for (auto const &m : ctx.color_members[c1])
                if (m != r1) c1_without = std::max(c1_without, root_infos_[m].size_class);
            for (auto const &m : ctx.color_members[c2])
                if (m != r2) c2_without = std::max(c2_without, root_infos_[m].size_class);

            ElementCount old_total = ctx.color_slot_size[c1] + ctx.color_slot_size[c2];
            ElementCount new_total = std::max(c1_without, s2) + std::max(c2_without, s1);

            if (new_total < old_total) {
                // Do not swap if it would grow an external slot
                ElementCount new_c1 = std::max(c1_without, s2);
                ElementCount new_c2 = std::max(c2_without, s1);
                if (ctx.color_external_index[c1] != SIZE_MAX && new_c1 > ctx.color_slot_size[c1]) continue;
                if (ctx.color_external_index[c2] != SIZE_MAX && new_c2 > ctx.color_slot_size[c2]) continue;

                ctx.color_members[c1].erase(r1);
                ctx.color_members[c2].erase(r2);
                ctx.color_members[c1].insert(r2);
                ctx.color_members[c2].insert(r1);
                color_map_[r1] = c2;
                color_map_[r2] = c1;
                ctx.color_slot_size[c1] = new_c1;
                ctx.color_slot_size[c2] = new_c2;
                return true;
            }
        }
    }
    return false;
}

void RegisterAllocator::try_color_dissolution(ColoringContext &ctx) {
    std::vector<ColorId> color_order;
    for (ColorId c = 0; c < num_colors_; ++c) {
        if (!ctx.color_members[c].empty()) color_order.push_back(c);
    }
    auto const &slot_sizes = ctx.color_slot_size;
    std::sort(color_order.begin(), color_order.end(),
              [&slot_sizes](size_t a, size_t b) {
                  return slot_sizes[a] < slot_sizes[b];
              });

    for (ColorId c : color_order) {
        if (ctx.color_members[c].empty() ||
            ctx.color_members[c].size() > config_.dissolution_max_members) continue;
        // Do not dissolve external (borrowed) color slots
        if (ctx.color_external_index[c] != SIZE_MAX) continue;

        std::vector<std::string> members(ctx.color_members[c].begin(),
                                         ctx.color_members[c].end());
        std::sort(members.begin(), members.end(),
                  [this](std::string const &a, std::string const &b) {
                      return root_infos_.at(a).size_class > root_infos_.at(b).size_class;
                  });

        auto saved_members = ctx.color_members[c];
        auto saved_slot = ctx.color_slot_size[c];
        ctx.color_members[c].clear();
        ctx.color_slot_size[c] = 0;
        for (auto const &m : members) color_map_[m] = SIZE_MAX;

        std::vector<std::pair<std::string, ColorId>> reassignment;
        bool feasible = true;
        ElementCount added_cost = 0;

        for (auto const &m : members) {
            ElementCount msz = root_infos_[m].size_class;
            ColorId best_c = SIZE_MAX;
            ElementCount best_cost = SIZE_MAX;

            for (ColorId tc = 0; tc < num_colors_; ++tc) {
                if (tc == c || ctx.color_members[tc].empty()) continue;
                if (!can_place_root_in_color(m, tc)) continue;
                // Do not grow external slot beyond its fixed capacity
                if (ctx.color_external_index[tc] != SIZE_MAX && msz > ctx.color_slot_size[tc]) continue;
                ElementCount cost = (msz > ctx.color_slot_size[tc]) ? (msz - ctx.color_slot_size[tc]) : 0;
                if (cost < best_cost) {
                    best_cost = cost;
                    best_c = tc;
                }
            }

            if (best_c == SIZE_MAX) {
                feasible = false;
                break;
            }
            reassignment.emplace_back(m, best_c);
            added_cost += best_cost;
            color_map_[m] = best_c;
            ctx.color_members[best_c].insert(m);
            ctx.color_slot_size[best_c] = std::max(ctx.color_slot_size[best_c], msz);
        }

        if (!feasible || added_cost >= saved_slot) {
            // Rollback
            std::unordered_set<ColorId> affected;
            for (auto const &[m, tc] : reassignment) {
                ctx.color_members[tc].erase(m);
                affected.insert(tc);
            }
            for (auto ac : affected) recompute_color_slot_size(ctx, ac);
            ctx.color_members[c] = saved_members;
            ctx.color_slot_size[c] = saved_slot;
            for (auto const &m : members) color_map_[m] = c;
        }
    }
}

void RegisterAllocator::try_slot_absorption(ColoringContext &ctx) {
    bool absorbed = true;
    while (absorbed) {
        absorbed = false;
        for (ColorId c1 = 0; c1 < num_colors_ && !absorbed; ++c1) {
            if (ctx.color_members[c1].empty()) continue;
            for (ColorId c2 = c1 + 1; c2 < num_colors_ && !absorbed; ++c2) {
                if (ctx.color_members[c2].empty()) continue;

                ColorId big_c = (ctx.color_slot_size[c1] >= ctx.color_slot_size[c2]) ? c1 : c2;
                ColorId small_c = (big_c == c1) ? c2 : c1;
                ElementCount new_slot = std::max(ctx.color_slot_size[big_c],
                                                 ctx.color_slot_size[small_c]);
                if (new_slot >= ctx.color_slot_size[big_c] + ctx.color_slot_size[small_c])
                    continue;

                bool compatible = true;
                for (auto const &mb : ctx.color_members[big_c]) {
                    if (!compatible) break;
                    for (auto const &ms : ctx.color_members[small_c]) {
                        if (root_infos_[mb].type_idx != root_infos_[ms].type_idx) {
                            compatible = false;
                            break;
                        }
                        if (adj_.count(mb) && adj_.at(mb).count(ms)) {
                            compatible = false;
                            break;
                        }
                    }
                }

                if (compatible) {
                    // Do not merge two external slots together,
                    // and do not grow an external slot's capacity.
                    bool big_is_ext = (ctx.color_external_index[big_c] != SIZE_MAX);
                    bool small_is_ext = (ctx.color_external_index[small_c] != SIZE_MAX);
                    if (big_is_ext && small_is_ext) continue;
                    if (big_is_ext && new_slot > ctx.color_slot_size[big_c]) continue;
                    if (small_is_ext) {
                        // Absorb external into owned: swap so external is big_c
                        // (only if the ext capacity >= new_slot)
                        if (ctx.color_slot_size[small_c] >= new_slot) {
                            std::swap(big_c, small_c);
                        } else {
                            continue;
                        }
                    }

                    for (auto const &m : ctx.color_members[small_c]) {
                        color_map_[m] = big_c;
                        ctx.color_members[big_c].insert(m);
                    }
                    ctx.color_members[small_c].clear();
                    ctx.color_slot_size[big_c] = new_slot;
                    ctx.color_slot_size[small_c] = 0;
                    absorbed = true;
                }
            }
        }
    }
}

void RegisterAllocator::compact_colors(ColoringContext &ctx) {
    std::vector<ColorId> old_to_new(num_colors_, SIZE_MAX);
    ColorId new_num = 0;
    for (ColorId c = 0; c < num_colors_; ++c) {
        if (!ctx.color_members[c].empty()) old_to_new[c] = new_num++;
    }
    for (auto &[root, color] : color_map_) {
        color = old_to_new[color];
    }
    // Compact color_external_index in sync
    std::vector<size_t> new_ext_index(new_num, SIZE_MAX);
    for (ColorId c = 0; c < num_colors_; ++c) {
        if (old_to_new[c] != SIZE_MAX) {
            new_ext_index[old_to_new[c]] = ctx.color_external_index[c];
        }
    }
    ctx.color_external_index = std::move(new_ext_index);
    num_colors_ = new_num;
}

// Seed the coloring context with pre-existing external slots so that
// the greedy coloring can reuse parent-graph storage when beneficial.
void RegisterAllocator::seed_external_slots(ColoringContext &ctx) {
    for (auto const &ext : external_slots_) {
        ColorId c = num_colors_++;
        ctx.color_slot_size.push_back(ext.capacity);
        ctx.color_members.emplace_back();
        ctx.color_external_index.push_back(ext.external_index);
    }
}

void RegisterAllocator::greedy_color() {
    color_map_.clear();
    num_colors_ = 0;

    ColoringContext ctx;
    for (auto const &[root, ri] : root_infos_) {
        if (ri.skip) continue;
        ctx.active_roots.push_back(root);
    }
    if (ctx.active_roots.empty()) return;

    // Phase 0: Seed with external (parent-graph) slots so greedy coloring
    // can reuse them for free instead of allocating new storage.
    seed_external_slots(ctx);

    // Phase 1: Initial greedy coloring
    initial_greedy_coloring(ctx);

    // Phase 2a & 2b: Single-root moves and pairwise swaps
    bool improved = true;
    for (int iter = 0; improved && iter < config_.local_search_max_iterations; ++iter) {
        improved = try_single_root_moves(ctx);
        if (!improved) {
            improved = try_pairwise_swaps(ctx);
        }
    }

    // Phase 2c: Color dissolution
    try_color_dissolution(ctx);

    // Phase 2d: Slot absorption (merge compatible color pairs)
    try_slot_absorption(ctx);

    // Compact: eliminate empty colors
    compact_colors(ctx);

    // Persist the external-index mapping for build_plan
    color_external_index_ = std::move(ctx.color_external_index);
}

// ================================================================
// Phase F: Build and return the allocation plan
// ================================================================

AllocationPlan RegisterAllocator::build_plan() {
    AllocationPlan plan;

    struct SlotKey {
        std::type_index type_idx;
        ColorId color;
        bool operator==(SlotKey const &o) const { return type_idx == o.type_idx && color == o.color; }
    };
    struct SlotKeyHash {
        size_t operator()(SlotKey const &k) const {
            return k.type_idx.hash_code() ^ (std::hash<ColorId>{}(k.color) << 1);
        }
    };

    std::unordered_map<SlotKey, ColorId, SlotKeyHash> slot_index_map;

    for (auto const &[root, ri] : root_infos_) {
        if (ri.skip) continue;
        auto color_it = color_map_.find(root);
        if (color_it == color_map_.end()) continue;

        auto key = SlotKey{ri.type_idx, color_it->second};
        auto sit = slot_index_map.find(key);
        ColorId slot_idx;
        if (sit == slot_index_map.end()) {
            slot_idx = plan.color_slots.size();
            slot_index_map[key] = slot_idx;
            ColorSlot cs;
            cs.color_id = slot_idx;
            cs.type_idx = ri.type_idx;
            cs.size_class = ri.size_class;
            // Propagate borrowed status from graph coloring
            if (color_it->second < color_external_index_.size()) {
                cs.borrowed_slot_index = color_external_index_[color_it->second];
            }
            plan.color_slots.push_back(std::move(cs));
        } else {
            slot_idx = sit->second;
            plan.color_slots[slot_idx].size_class =
                std::max(plan.color_slots[slot_idx].size_class, ri.size_class);
        }

        for (auto const &member : ri.members) {
            plan.color_slots[slot_idx].members.push_back(member);

            TensorMapping tm;
            tm.color_id = slot_idx;
            auto &unit = units_[member];
            tm.is_view = unit.is_view;
            tm.is_inplace = unit.is_inplace;
            if (unit.is_inplace && inplace_source_map_.count(member)) {
                tm.view_source = inplace_source_map_[member];
            }
            tm.root_name = root;
            plan.tensor_map[member] = std::move(tm);
        }
    }

    plan.total_physical_slots = plan.color_slots.size();
    plan.schedule = schedule_;
    plan.total_intermediates = 0;
    plan.view_merges = 0;
    plan.inplace_merges = 0;

    for (auto const &[name, unit] : units_) {
        if (!unit.skip_allocation) plan.total_intermediates++;
        if (unit.is_view) plan.view_merges++;
        if (unit.is_inplace) plan.inplace_merges++;
    }

    return plan;
}

}// namespace lcml::onnx
