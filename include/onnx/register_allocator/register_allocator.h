#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <typeindex>
#include <limits>
#include <memory>

#include "onnx/onnx.h"
#include "onnx/operator.h"
#include "union_find.h"
#include "allocation_types.h"

namespace lcml::onnx {

// Tunable constants for the register allocator.
struct RegAllocConfig {
    // Inplace coalescing: elements threshold below which small-tensor tolerance applies
    ElementCount small_tensor_threshold = 8;
    // Inplace coalescing: tolerance added to net benefit for small tensors
    ptrdiff_t small_tensor_inflation_tolerance = 7;
    // Inplace coalescing: root size threshold for range extension limiting
    ElementCount large_root_threshold = 32;
    // Inplace coalescing: minimum allowed range for large roots
    ElementCount large_root_min_range = 30;
    // Inplace coalescing: budget divisor for large root range limit
    ElementCount large_root_range_budget = 5000;
    // Greedy coloring: size threshold for preferring smallest color ID
    ElementCount color_large_size_threshold = 32;
    // Local search: maximum iterations for single-move / swap improvement
    int local_search_max_iterations = 80;
    // Color dissolution: maximum members in a color to attempt dissolution
    ColorId dissolution_max_members = 20;
    // Enable diagnostic output summarizing allocation statistics
    bool verbose = true;
};

// Graph-coloring-based tensor register allocator.
//
// Optimization pipeline:
//   Phase 0: Virtual node ordering (instruction scheduling) to minimize pressure
//   Phase A: Universal inplace coalescing
//   Phase B: Compute live ranges per union-find root
//   Phase C: Normalize size classes
//   Phase D: Build interference graph
//   Phase E: Size-aware greedy graph coloring with local search
//   Phase F: Build allocation plan
class RegisterAllocator {
public:
    using OperatorList = std::vector<std::unique_ptr<Operator>>;

    RegisterAllocator(onnx::Graph const &graph,
                      LastUseMap const &last_use_map,
                      OperatorList const &ops,
                      onnx::StringNodeMap<bool> const &already_bound,
                      RegAllocConfig const &config = {},
                      std::vector<ExternalSlot> external_slots = {});

    AllocationPlan run();

private:
    onnx::Graph const &graph_;
    LastUseMap const &orig_last_use_map_;
    OperatorList const &ops_;
    onnx::StringNodeMap<bool> const &already_bound_;
    RegAllocConfig config_;
    std::vector<ExternalSlot> external_slots_;

    // Virtual ordering: orig_node_idx -> virtual_idx and vice versa
    Schedule schedule_;                     // schedule_[virtual_idx] = orig_node_idx
    std::vector<NodeIndex> orig_to_virtual_;// orig_to_virtual_[orig_idx] = virtual_idx

    // Remapped last-use map using virtual indices
    LastUseMap vlast_use_map_;

    // Per-tensor data (uses virtual indices)
    onnx::StringNodeMap<AllocationUnit> units_;
    UnionFind uf_;

    onnx::StringNodeMap<std::string> inplace_source_map_;

    struct RootInfo {
        std::type_index type_idx = std::type_index(typeid(void));
        ElementCount max_elements = 0;
        NodeIndex merged_def = SIZE_MAX;
        NodeIndex merged_last_use = 0;
        ElementCount size_class = 0;
        std::vector<std::string> members;
        bool skip = false;
    };
    onnx::StringNodeMap<RootInfo> root_infos_;

    onnx::StringNodeMap<std::unordered_set<std::string>> adj_;
    onnx::StringNodeMap<ColorId> color_map_;
    ColorId num_colors_ = 0;
    std::vector<size_t> color_external_index_;// Per-color external slot index (SIZE_MAX if owned)

    // ---- Helpers ----
    bool is_skip_var(std::string const &vn) const;
    static ElementCount compute_num_elements(std::vector<size_t> const &shape);

    // ---- Phase 0 sub-routines ----
    // Scheduling context passed between Phase 0 sub-routines
    struct SchedulingContext {
        size_t N = 0;
        onnx::StringNodeMap<NodeIndex> producer;
        std::vector<std::vector<std::string>> ext_input_names;
        std::vector<std::unordered_set<size_t>> successors;
        std::vector<size_t> in_degree;
        onnx::StringNodeMap<size_t> remaining_consumers;
        onnx::StringNodeMap<size_t> var_size;
        std::vector<size_t> height;
    };

    static void collect_external_refs_recursive(
        onnx::Graph const &sub_graph,
        std::unordered_set<std::string> &ext_refs);

    void build_producer_map(SchedulingContext &ctx) const;
    void build_ext_input_names(SchedulingContext &ctx) const;
    void build_dag_edges(SchedulingContext &ctx) const;
    void count_remaining_consumers(SchedulingContext &ctx) const;
    void build_var_size_map(SchedulingContext &ctx) const;
    void compute_node_heights(SchedulingContext &ctx) const;
    void schedule_nodes(SchedulingContext &ctx);
    void remap_last_use_map(SchedulingContext const &ctx);
    void mark_permanent_variables();

    // ---- Phase 0 pipeline ----
    void compute_virtual_order();// Phase 0

    // ---- Phase 0.5 ----
    void init_units();

    // ---- Phase A sub-routines ----
    bool check_view_eligibility(onnx::Node const &node, Operator const &op) const;

    struct InplaceCandidateResult {
        bool accepted = false;
        std::string best_input;
        ElementCount best_elems = 0;
        int64_t best_net_benefit = 0;
    };
    InplaceCandidateResult evaluate_inplace_candidate(
        onnx::Node const &node, Operator const &op,
        size_t vi, size_t out_idx) const;

    void apply_inplace_coalescing();// Phase A

    // ---- Phase B-D ----
    void compute_live_ranges();     // Phase B
    void normalize_size_classes();  // Phase C
    void build_interference_graph();// Phase D

    // ---- Phase E sub-routines ----
    // Coloring context shared among Phase E sub-routines
    struct ColoringContext {
        std::vector<std::string> active_roots;
        std::vector<ElementCount> color_slot_size;
        std::vector<std::unordered_set<std::string>> color_members;
        // Map from internal color ID -> ExternalSlot::external_index (SIZE_MAX if owned)
        std::vector<size_t> color_external_index;
    };

    bool can_place_root_in_color(std::string const &root, ColorId color) const;
    void recompute_color_slot_size(ColoringContext &ctx, ColorId color) const;
    void seed_external_slots(ColoringContext &ctx);
    void initial_greedy_coloring(ColoringContext &ctx);
    bool try_single_root_moves(ColoringContext &ctx);
    bool try_pairwise_swaps(ColoringContext &ctx);
    void try_color_dissolution(ColoringContext &ctx);
    void try_slot_absorption(ColoringContext &ctx);
    void compact_colors(ColoringContext &ctx);
    void greedy_color();// Phase E

    // ---- Phase F ----
    AllocationPlan build_plan();// Phase F
};

}// namespace lcml::onnx
