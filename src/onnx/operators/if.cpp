#include "onnx/operator.h"
#include "onnx/network_instance.h"
#include "onnx/operators/common.h"
#include "onnx/onnx.h"

namespace lcml::onnx {

class If : public Operator {
private:
    std::shared_ptr<onnx::Graph> then_branch_;
    std::shared_ptr<onnx::Graph> else_branch_;
    NetworkInstance *env_ = nullptr;
    TensorTable *curr_table = nullptr;

public:
    If(std::shared_ptr<onnx::Graph> then_branch,
       std::shared_ptr<onnx::Graph> else_branch)
        : Operator("If"),
          then_branch_(std::move(then_branch)),
          else_branch_(std::move(else_branch)) {}

    bool need_outline() const override { return false; }

    void set_environment(NetworkInstance &env, TensorTable &table) override {
        env_ = &env;
        curr_table = &table;
    }

    void forward(std::span<std::reference_wrapper<ITensor>> inputs,
                 std::span<std::reference_wrapper<ITensor>> outputs) override {
        auto &cond_tensor = inputs[0].get();
#ifndef NDEBUG
        LUISA_ASSERT(inputs.size() == 1, "If operator requires exactly 1 input (cond).");
        LUISA_ASSERT(env_ != nullptr, "If operator: NetworkInstance environment not set.");
        LUISA_ASSERT(cond_tensor.element_type() == typeid(bool), "If operator: cond must be a bool tensor.");
        LUISA_ASSERT(cond_tensor.size() == 1, "If operator: cond must be a scalar (single element).");

        // Validate both branches have the correct output count
        LUISA_ASSERT(then_branch_->get_outputs().size() == outputs.size() &&
                         else_branch_->get_outputs().size() == outputs.size(),
                     "If operator: branch output count mismatch with operator outputs ({}).",
                     outputs.size());
#endif

        // Collect dead parent phantom storages that can be safely reused by subgraphs.
        // A phantom storage is "dead" when ALL its member tensors' last-use indices
        // are < the current virtual execution index (i.e. no longer needed by parent).
        std::vector<ITensor *> reusable_storages;
        auto const &ctx = env_->execution_context();
        if (ctx.tensor_table && ctx.plan && ctx.last_use_map) {
            reusable_storages = NetworkInstance::collect_dead_parent_storages(
                *ctx.tensor_table, *ctx.plan, *ctx.last_use_map, ctx.current_node_idx);
        }

        // Prepare both branches with shared phantom storage.
        // Since then_branch and else_branch are mutually exclusive ($if/$else),
        // their intermediate tensors can safely reuse the same storage arrays,
        // reducing total register pressure.
        std::vector<std::reference_wrapper<ITensor>> out_refs(outputs.begin(), outputs.end());
        std::vector<std::pair<onnx::Graph const *, std::span<std::reference_wrapper<ITensor>>>> branch_infos{
            {then_branch_.get(), std::span{out_refs}},
            {else_branch_.get(), std::span{out_refs}},
        };
        auto prepared = env_->prepare_exclusive_graphs(curr_table, branch_infos, reusable_storages);

        // Evaluate condition (DSL Var<bool>)
        auto &cond = static_cast<NNTensor<bool> &>(cond_tensor);
        auto cond_val = cond[0u];

        // Use DSL $if/$else to emit both branches into the shader.
        // Only execution (phase 3) happens inside $if/$else; tensor allocation
        // was already done above at compile time.
        $if (cond_val) {
            env_->execute_prepared_graph(prepared[0], *then_branch_);
        }
        $else {
            env_->execute_prepared_graph(prepared[1], *else_branch_);
        };
    }
};

REGISTER_TO_DEFAULT_OPSET(If) {
    auto &then_attr = node.get_attribute("then_branch");
    auto &else_attr = node.get_attribute("else_branch");

    auto then_graph = then_attr.get<onnx::AttributeType::GRAPH>();
    auto else_graph = else_attr.get<onnx::AttributeType::GRAPH>();

    return std::make_unique<If>(std::move(then_graph), std::move(else_graph));
};

}// namespace lcml::onnx
