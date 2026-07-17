#pragma once

#include "luisa_ml_config.h"

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/memory.h>

#include "tensor.h"

namespace lcml::onnx {

class Node;
class NetworkInstance;
class TensorTable;

class LUISA_ONNX_API Operator {
private:
    luisa::string _name;
public:
    Operator() = delete;
    virtual ~Operator() = default;
    Operator(const Operator &other) = default;
    Operator(Operator &&other) noexcept = default;
    Operator &operator=(const Operator &other) = default;
    Operator &operator=(Operator &&other) noexcept = default;

    Operator(luisa::string name) : _name(std::move(name)) {}
    luisa::string const &get_name() const { return _name; }

    virtual void forward(luisa::span<std::reference_wrapper<ITensor>> inputs, luisa::span<std::reference_wrapper<ITensor>> outputs) = 0;

    /// Query whether the given output will be a zero-copy view of an input.
    /// The Node reference provides access to input/output Variable shapes and
    /// constant raw_data, enabling precise compile-time view detection.
    /// Subclasses should override this and return true only for output indices
    /// that are guaranteed to be views given the graph's static information.
    virtual bool is_output_view([[maybe_unused]] size_t output_index,
                                [[maybe_unused]] onnx::Node const &node) const { return false; }

    /// Query whether this operator can safely operate in-place, i.e. its
    /// output[0] can reuse input[0]'s memory.  This is safe for:
    ///  - Element-wise ops (Add/Mul/Sub/Relu/...): output[i] = f(input[i], ...)
    ///  - Scatter ops that copy input[0] to output[0] then mutate
    /// The allocator will reuse input[0]'s slot for output[0] when input[0]
    /// reaches its last use at this node and the element counts match.
    /// NOT safe for MatMul/Gemm/Conv/Reduce/Softmax/Normalization.
    virtual bool can_operate_inplace() const { return false; }

    virtual bool need_outline() const { return true; }

    virtual void set_environment(NetworkInstance &, TensorTable &) {}

    virtual void set_warp_size(uint32_t) {}
};

}// namespace lcml::onnx
