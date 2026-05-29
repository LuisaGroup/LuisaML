#pragma once

#include <memory>
#include <functional>
#include <span>
#include <luisa/runtime/byte_buffer.h>
#include "onnx.h"
#include "operators/common.h"
#include "tensor_table.h"
#include "register_allocator/allocation_types.h"

namespace lcml::onnx {

/**
 * @brief Holds the prepared state of a graph (operators, tensor table,
 *        and execution schedule) so that allocation (compile-time) and
 *        execution (GPU code generation) can be separated.
 */
struct PreparedGraph {
    std::vector<std::unique_ptr<Operator>> ops;
    TensorTable tensor_table;
    LastUseMap last_use_map;
    Schedule schedule;
    AllocationPlan plan;///< Full allocation plan (needed for parent-storage reuse in subgraphs)
};

/**
 * @brief NetworkInstance manages a neural network model instance.
 * 
 * It holds the ONNX model, input/output tensor references, and
 * intermediate tensor storage. The forward() method executes the
 * model graph using luisa DSL.
 */
class NetworkInstance {
public:
    using TensorRef = std::reference_wrapper<ITensor>;
    using ExternalTensorTable = onnx::StringNodeMap<TensorRef>;

private:
    onnx::Model model_;
    ExternalTensorTable inputs_;
    ExternalTensorTable outputs_;
    TensorTable tensor_table_;
    luisa::compute::Var<luisa::compute::ByteBuffer> *weight_buffer_ = nullptr;

    /// @brief Warp size for warp-level intrinsics optimization (warp_read_lane, warp_active_sum, etc.).
    ///        Set to <= 1 to disable warp optimization (e.g. single-thread test dispatch
    ///        where inactive lanes have undefined values).
    ///        Default: 32 (standard warp width for multi-thread render dispatches).
    uint warp_size_ = 32;

    /// @brief Whether to use CooperativeVector hardware acceleration for matrix multiplication.
    ///        When enabled, Gemm replaces the scalar/warp loop with a single cooperative_mul_add
    ///        hardware instruction. Default: false (requires hardware support, enable explicitly).
    bool use_cooperative_vector_ = false;
    bool enable_logging_ = false;

    /// @brief Execution context for the currently running graph.
    ///        Set by execute_operators() before each operator runs;
    ///        enables subgraph operators (e.g. If) to inspect parent-graph
    ///        allocation state for safe storage reuse.
    struct ExecutionContext {
        TensorTable *tensor_table = nullptr;
        AllocationPlan const *plan = nullptr;
        LastUseMap const *last_use_map = nullptr;
        size_t current_node_idx = 0;///< Current original node index (matches last_use_map indices)
    };
    ExecutionContext exec_ctx_;

    /// @brief Pre-created operators for each graph node.
    using OperatorList = std::vector<std::unique_ptr<Operator>>;

    /**
     * @brief Pre-create all operator instances for the graph nodes.
     */
    static OperatorList create_all_operators(
        onnx::Graph const &graph,
        onnx::OperatorSet &opset);

    /**
     * @brief Build a map of variable name -> last node index where it is used.
     *
     * Performs static analysis on the graph to determine the last consumer
     * of each intermediate tensor. Graph outputs and constants are marked
     * with SIZE_MAX (never reclaimable). View dependencies are propagated
     * so that source tensors live at least as long as their views.
     */
    static LastUseMap build_last_use_map(
        onnx::Graph const &graph,
        TensorTable const &tensor_table,
        OperatorList const &ops);

    /**
     * @brief Compute total number of elements from a shape vector.
     */
    static size_t compute_num_elements(std::vector<size_t> const &shape);

    /**
     * @brief Validate shape and element type of an external tensor against a graph variable,
     *        then bind it into the tensor table.
     */
    void validate_and_bind(
        TensorTable &tensor_table,
        std::string_view role,
        onnx::Variable const &var,
        ITensor &ext_tensor);

    /**
     * @brief Bind external input/output tensors into the tensor table.
     */
    void bind_external_tensors(
        TensorTable &tensor_table,
        onnx::Graph const &graph);

    /**
     * @brief Create an NNTensor for a single graph variable and add it as owned entry.
     */
    void create_tensor_for_var(
        TensorTable &tensor_table,
        onnx::Variable const &var);

    /**
     * @brief Create NNTensor for each graph variable not already bound,
     *        and fill weight data from weight_buffer or embedded raw_data.
     */
    void create_intermediate_tensors(
        TensorTable &tensor_table,
        onnx::Graph const &graph);

    /**
     * @brief Create intermediate tensors with graph-coloring register allocation.
     *        Uses RegisterAllocator to compute optimal storage reuse via
     *        interference graph coloring, then creates tensors with shared storage.
     * @param external_slots  Optional external slots from parent graph that the
     *        allocator can borrow for free instead of creating new storage.
     * @param external_storages  When external_slots is non-empty, this provides
     *        the actual ITensor storage pointers indexed by ExternalSlot::external_index.
     * @return The full allocation plan (includes execution schedule, slot info, etc.).
     */
    AllocationPlan create_intermediate_tensors_pooled(
        TensorTable &tensor_table,
        onnx::Graph const &graph,
        LastUseMap const &last_use_map,
        OperatorList const &ops,
        std::vector<ExternalSlot> external_slots = {},
        std::vector<ITensor *> external_storages = {});

    /**
     * @brief Execute all operators in the given schedule order,
     *        reading/writing through the given tensor_table.
     * @param schedule  If non-empty, specifies the execution order
     *        (schedule[vi] = original node index). If empty, uses
     *        the original topological order.
     * @param plan  Optional allocation plan pointer. When provided,
     *        stored in exec_ctx_ so that subgraph operators (e.g. If)
     *        can inspect parent-graph allocation state.
     */
    void execute_operators(
        TensorTable &tensor_table,
        onnx::Graph const &graph,
        OperatorList &ops,
        LastUseMap const &last_use_map,
        Schedule const &schedule,
        AllocationPlan const *plan = nullptr);

public:
    NetworkInstance() = default;
    ~NetworkInstance() = default;

    // Non-copyable due to unique_ptr members
    NetworkInstance(const NetworkInstance &) = delete;
    NetworkInstance &operator=(const NetworkInstance &) = delete;
    NetworkInstance(NetworkInstance &&) noexcept = default;
    NetworkInstance &operator=(NetworkInstance &&) noexcept = default;

    // ==================== Setup Interface ====================

    /**
     * @brief Set the ONNX model for this instance.
     */
    void set_model(onnx::Model model) {
        model_ = std::move(model);
    }

    /**
     * @brief Bind an external tensor as a named input.
     */
    void set_input(std::string_view name, ITensor &tensor) {
        inputs_.insert_or_assign(std::string{name}, std::ref(tensor));
    }

    /**
     * @brief Bind an external tensor as a named output.
     */
    void set_output(std::string_view name, ITensor &tensor) {
        outputs_.insert_or_assign(std::string{name}, std::ref(tensor));
    }

    /**
     * @brief Bind a weight buffer (ByteBuffer) for loading external weights.
     */
    void set_weight_buffer(luisa::compute::Var<luisa::compute::ByteBuffer> &buffer) {
        weight_buffer_ = &buffer;
    }

    /**
     * @brief Set the warp size for warp-level intrinsics optimization.
     *        Set to <= 1 to disable warp optimization (e.g. single-thread test).
     *        Default: 32.
     */
    void set_warp_size(uint size) { warp_size_ = size; }
    uint warp_size() const { return warp_size_; }
    bool use_warp_optimization() const { return warp_size_ > 1; }

    /**
     * @brief Enable or disable CooperativeVector hardware acceleration for Gemm.
     *        Default: true.
     */
    void set_use_cooperative_vector(bool enable) { use_cooperative_vector_ = enable; }
    bool use_cooperative_vector() const { return use_cooperative_vector_; }

    void enable_logging(bool enable) { enable_logging_ = enable; }

    onnx::Model const &get_model() const { return model_; }
    ExternalTensorTable const &get_inputs() const { return inputs_; }
    ExternalTensorTable const &get_outputs() const { return outputs_; }
    TensorTable const &get_tensor_table() const { return tensor_table_; }

    /**
     * @brief Execute a forward pass on a specific graph.
     *        @param tensor_table  The tensor table for variable lookup and storage.
     *        @param graph  The graph to execute.
     *        This enables subgraph computation with isolated temporary variables.
     */
    void forward_graph(TensorTable &tensor_table, onnx::Graph const &graph);

    /**
     * @brief Prepare a graph for execution: create operators, analyze lifetimes,
     *        allocate tensors (phase 0-2) without executing any GPU code.
     *        The returned PreparedGraph can later be executed with execute_prepared_graph.
     */
    PreparedGraph prepare_graph(TensorTable *parent_table, onnx::Graph const &graph,
                                std::span<std::reference_wrapper<ITensor>> bound_outputs);

    /**
     * @brief Execute a previously prepared graph (phase 3 only).
     */
    void execute_prepared_graph(PreparedGraph &prepared, onnx::Graph const &graph);

    /**
     * @brief Prepare multiple mutually-exclusive subgraphs (e.g. If then/else branches)
     *        with shared phantom storage, then return the prepared states.
     *        Since branches execute exclusively, their intermediate tensors can reuse
     *        the same storage arrays, reducing total register pressure.
     * @param parent_table  Tensor table of the parent graph (for fallback lookups).
     * @param branch_infos  Each branch's graph and its bound output tensor refs.
     * @param reusable_parent_storages  Parent phantom storages confirmed dead at the
     *        current execution point (safe for subgraph reuse).  Pass empty to disable
     *        parent-storage reuse.  Caller is responsible for liveness checking.
     */
    std::vector<PreparedGraph> prepare_exclusive_graphs(
        TensorTable *parent_table,
        std::vector<std::pair<onnx::Graph const *, std::span<std::reference_wrapper<ITensor>>>> const &branch_infos,
        std::vector<ITensor *> const &reusable_parent_storages = {});

    /**
     * @brief Execute the neural network forward pass on the model's main graph.
     */
    void forward();

    /**
     * @brief Collect phantom storages from the parent graph that are confirmed dead
     *        at the given virtual execution index.
     *
     * A phantom storage is "dead" when ALL of its member tensors' last-use indices
     * are strictly less than the given virtual index `vi`.  Such storages can be
     * safely lent to subgraphs without corrupting parent-graph data.
     *
     * @param tensor_table  The parent graph's tensor table (contains phantom storages).
     * @param plan          The parent graph's allocation plan (maps slots to members).
     * @param last_use_map  The parent graph's last-use map (per-tensor last virtual index).
     * @param node_idx  The current original node index of the If node.
     *                   (matches the index space used in build_last_use_map).
     * @return A vector of ITensor* pointing to dead parent phantom storages.
     */
    static std::vector<ITensor *> collect_dead_parent_storages(
        TensorTable &tensor_table,
        AllocationPlan const &plan,
        LastUseMap const &last_use_map,
        size_t node_idx);

    /// @brief Get the current execution context of the parent graph.
    ///        Valid only during operator execution (inside execute_operators).
    ExecutionContext const &execution_context() const noexcept { return exec_ctx_; }
};
}// namespace lcml::onnx
