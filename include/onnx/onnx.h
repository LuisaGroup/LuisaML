#pragma once

#include "luisa_ml_config.h"

#include <array>
#include <numeric>
#include <initializer_list>
#include <type_traits>
#include <algorithm>
#include <utility>
#include <functional>

#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/variant.h>
#include <luisa/core/stl/memory.h>

#include "operator.h"

#include "magic_enum/magic_enum_all.hpp"

namespace yyjson::reader {
class const_value_ref;
}// namespace yyjson::reader

namespace lcml::onnx {

using json_cvalue = yyjson::reader::const_value_ref;

enum class DataType {
    UNDEFINED = 0,
    FLOAT = 1,
    UINT8 = 2,
    INT8 = 3,
    UINT16 = 4,
    INT16 = 5,
    INT32 = 6,
    INT64 = 7,
    STRING = 8,
    BOOL = 9,
    FLOAT16 = 10,
    DOUBLE = 11,
    UINT32 = 12,
    UINT64 = 13,
    COMPLEX64 = 14,
    COMPLEX128 = 15,
    BFLOAT16 = 16,
    FLOAT8E4M3FN = 17,
    FLOAT8E4M3FNUZ = 18,
    FLOAT8E5M2 = 19,
    FLOAT8E5M2FNUZ = 20,
    UINT4 = 21,
    INT4 = 22,
    FLOAT4E2M1 = 23,
    FLOAT8E8M0 = 24,
    UINT2 = 25,
    INT2 = 26,
    FLOAT16QUANTIZED = 27,
};

enum class AttributeType {
    UNDEFINED = 0,
    FLOAT = 1,
    INT = 2,
    STRING = 3,
    TENSOR = 4,
    GRAPH = 5,
    FLOATS = 6,
    INTS = 7,
    STRINGS = 8,
    TENSORS = 9,
    GRAPHS = 10,
    SPARSE_TENSOR = 11,
    SPARSE_TENSORS = 12,
    TYPE_PROTO = 13,
    TYPE_PROTOS = 14,
};

struct StringTransparentHasher {
    using is_transparent = void;

    size_t operator()(const luisa::string &key) const {
        return luisa::hash_value(key);
    }

    size_t operator()(std::string_view key) const {
        return luisa::hash_value(key);
    }

    size_t operator()(const char *key) const {
        return luisa::hash_value(key);
    }
};

template<typename T>
using StringNodeMap = luisa::unordered_map<luisa::string, T, StringTransparentHasher, std::equal_to<>>;

// Forward declarations
class Graph;
class OperatorSet;

class LUISA_ONNX_API Attribute : public luisa::variant<float,
                                      int32_t,
                                      luisa::string,
                                      luisa::shared_ptr<Graph>,
                                      luisa::vector<float>,
                                      luisa::vector<int32_t>,
                                      luisa::vector<luisa::string>,
                                      luisa::vector<luisa::shared_ptr<Graph>>> {
    using variant::variant;
    using variant::operator=;

    luisa::string _name;

public:
    Attribute() = default;

    template<typename T>
    Attribute(luisa::string name, T &&value)
        : variant(std::forward<T>(value)), _name(std::move(name)) {}

    luisa::string const &get_name() const { return _name; }

    static constexpr AttributeType type_of(Attribute const &attr) {
        switch (attr.index()) {
            case 0: return AttributeType::FLOAT;
            case 1: return AttributeType::INT;
            case 2: return AttributeType::STRING;
            case 3: return AttributeType::GRAPH;
            case 4: return AttributeType::FLOATS;
            case 5: return AttributeType::INTS;
            case 6: return AttributeType::STRINGS;
            case 7: return AttributeType::GRAPHS;
            default: return AttributeType::UNDEFINED;
        }
    }
    static constexpr size_t index_of(AttributeType type) {
        switch (type) {
            case AttributeType::FLOAT: return 0;
            case AttributeType::INT: return 1;
            case AttributeType::STRING: return 2;
            case AttributeType::GRAPH: return 3;
            case AttributeType::FLOATS: return 4;
            case AttributeType::INTS: return 5;
            case AttributeType::STRINGS: return 6;
            case AttributeType::GRAPHS: return 7;
            default: return ~0ull;
        }
    }
    template<AttributeType T>
    auto &get() {
        constexpr auto idx = index_of(T);
        return luisa::get<idx>(*this);
    }
    template<AttributeType T>
    auto const &get() const {
        constexpr auto idx = index_of(T);
        return luisa::get<idx>(*this);
    }

    // Parse an Attribute from JSON (parent graph is used to set parent pointer on subgraphs)
    static Attribute from_json(json_cvalue const &json, Graph const *parent = nullptr);
};

class LUISA_ONNX_API Variable {
private:
    luisa::string _name;
    DataType _dtype;
    luisa::vector<size_t> _shape;
    luisa::string _raw_data;
    uint64_t _buffer_start, _buffer_end;
    bool _is_constant{false};

public:
    Variable() = default;
    Variable(const Variable &other) = default;
    Variable(Variable &&other) noexcept = default;
    Variable &operator=(const Variable &other) = default;
    Variable &operator=(Variable &&other) noexcept = default;
    ~Variable() = default;

    Variable(luisa::string name,
             DataType dtype,
             luisa::vector<size_t> shape,
             luisa::string raw_data = {},
             uint64_t buffer_start = 0,
             uint64_t buffer_end = 0)
        : _name(std::move(name)),
          _dtype(dtype),
          _shape(std::move(shape)),
          _raw_data(std::move(raw_data)),
          _buffer_start(buffer_start),
          _buffer_end(buffer_end) {
    }

    void set_is_constant(bool v) { _is_constant = v; }
    bool is_constant() const { return _is_constant; }

    luisa::string const &get_name() const { return _name; }
    DataType get_dtype() const { return _dtype; }
    luisa::vector<size_t> const &get_shape() const { return _shape; }
    luisa::string const &get_raw_data() const { return _raw_data; }
    bool is_trainable_weight() const { return _buffer_end > _buffer_start; }
    std::pair<uint64_t, uint64_t> get_buffer_range() const { return {_buffer_start, _buffer_end}; }

    // Parse a Variable from a graph input/output/value_info JSON node
    static Variable from_json(json_cvalue const &json);
    // Parse a Variable from an initializer JSON node
    static Variable from_initializer_json(json_cvalue const &json);
};

class LUISA_ONNX_API Node {
private:
    luisa::string _name;
    luisa::string _op_type;
    luisa::vector<std::reference_wrapper<Variable const>> _inputs;
    luisa::vector<std::reference_wrapper<Variable const>> _outputs;
    StringNodeMap<Attribute> _attributes;

public:
    Node() = default;
    Node(const Node &other) = default;
    Node(Node &&other) noexcept = default;
    Node &operator=(const Node &other) = default;
    Node &operator=(Node &&other) noexcept = default;
    ~Node() = default;

    Node(luisa::string name, luisa::string op_type) : _name(std::move(name)), _op_type(std::move(op_type)) {}

    // Parse a Node from JSON, resolving variable references from graph
    static Node from_json(json_cvalue const &json, Graph const &graph);

    luisa::string const &get_name() const { return _name; }
    luisa::string const &get_op_type() const { return _op_type; }
    luisa::vector<std::reference_wrapper<Variable const>> const &get_inputs() const { return _inputs; }
    luisa::vector<std::reference_wrapper<Variable const>> const &get_outputs() const { return _outputs; }
    StringNodeMap<Attribute> const &get_attributes() const { return _attributes; }

    bool has_attribute(std::string_view name) const {
        if (auto it = _attributes.find(name); it != _attributes.end()) {
            return true;
        }
        return false;
    }

    void add_input(Variable const &var) { _inputs.emplace_back(var); }
    void add_output(Variable const &var) { _outputs.emplace_back(var); }

    void set_attribute(Attribute value) {
        _attributes.emplace(value.get_name(), std::move(value));
    }
    Attribute const &get_attribute(std::string_view name) const {
        auto it = _attributes.find(name);
        LUISA_ASSERT(it != _attributes.end(), "Attribute not found: {}", name);
        return it->second;
    }
    Attribute const *try_get_attr(std::string_view name) const {
        if (auto it = _attributes.find(name); it != _attributes.end()) {
            return &it->second;
        }
        return nullptr;
    }
};

class LUISA_ONNX_API Graph {
private:
    luisa::string _name;
    Graph const *parent_ = nullptr;
    StringNodeMap<Variable> _variables;
    luisa::vector<std::reference_wrapper<Variable const>> _input;
    luisa::vector<std::reference_wrapper<Variable const>> _output;
    luisa::vector<Node> _nodes;

public:
    Graph() = default;
    Graph(const Graph &other) = default;
    Graph(Graph &&other) noexcept = default;
    Graph &operator=(const Graph &other) = default;
    Graph &operator=(Graph &&other) noexcept = default;
    ~Graph() = default;

    Graph(luisa::string name)
        : _name(std::move(name)) {}

    Variable &add_variable(Variable &&var) {
        return _variables.emplace(var.get_name(), std::move(var)).first->second;
    }

    /// @brief Set a parent graph for fallback variable lookups.
    void set_parent(Graph const *parent) noexcept { parent_ = parent; }

    /// @brief Get the parent graph (may be nullptr).
    [[nodiscard]] Graph const *parent() const noexcept { return parent_; }

    /// @brief Get variable by name, falling back to parent if not found locally.
    Variable const &get_var(std::string_view name) const {
        if (auto it = _variables.find(name); it != _variables.end()) {
            return it->second;
        }
        if (parent_) return parent_->get_var(name);
        LUISA_ASSERT(false, "Variable not found: {}", name);
    }
    void add_input(std::string_view name) {
        _input.emplace_back(get_var(name));
    }
    void add_output(std::string_view name) {
        _output.emplace_back(get_var(name));
    }
    void add_node(Node &&node) {
        _nodes.emplace_back(std::move(node));
    }

    luisa::string const &get_name() const { return _name; }

    // Parse a Graph from JSON (parent graph is used for variable fallback in subgraphs)
    static Graph from_json(json_cvalue const &json, OperatorSet const &opset, Graph const *parent = nullptr);

    StringNodeMap<Variable> const &get_variables() const { return _variables; }
    void mark_constants();
    luisa::vector<std::reference_wrapper<Variable const>> const &get_inputs() const { return _input; }
    luisa::vector<std::reference_wrapper<Variable const>> const &get_outputs() const { return _output; }
    luisa::vector<Node> const &get_nodes() const { return _nodes; }

    /// @brief Reorder nodes according to the given schedule.
    /// schedule[i] = original index of the node that should be at position i.
    void reorder_nodes(luisa::vector<size_t> const &schedule) {
        if (schedule.size() != _nodes.size()) return;
        luisa::vector<Node> reordered;
        reordered.reserve(_nodes.size());
        for (auto idx : schedule) {
            reordered.push_back(std::move(_nodes[idx]));
        }
        _nodes = std::move(reordered);
    }
};

class LUISA_ONNX_API OperatorSet {
private:
    using Factory = luisa::function<luisa::unique_ptr<Operator>(Node const &)>;

    luisa::string _domain;
    int64_t _version;
    StringNodeMap<Factory> _factories;

public:
    OperatorSet() = delete;
    OperatorSet(const OperatorSet &other) = default;
    OperatorSet(OperatorSet &&other) noexcept = default;
    OperatorSet &operator=(const OperatorSet &other) = default;
    OperatorSet &operator=(OperatorSet &&other) noexcept = default;
    ~OperatorSet() = default;

    OperatorSet(luisa::string domain, int64_t version) : _domain(std::move(domain)), _version(version) {}
    luisa::string const &get_domain() const { return _domain; }
    int64_t get_version() const { return _version; }
    void register_operator(luisa::string name, Factory factory) {
        _factories.emplace(std::move(name), std::move(factory));
    }
    luisa::unique_ptr<Operator> create_operator(std::string_view name, Node const &node) const {
        auto it = _factories.find(name);
        LUISA_ASSERT(it != _factories.end(), "Operator not found: {}", name);
        return it->second(node);
    }
    bool has_operator(std::string_view name) const {
        return _factories.find(name) != _factories.end();
    }

    static OperatorSet &get_default();

    struct RegHelper {
        luisa::string name;
        OperatorSet &opset;

        RegHelper(luisa::string name, OperatorSet &opset = OperatorSet::get_default()) : name(std::move(name)), opset(opset) {}

        bool operator<<(Factory factory) {
            opset.register_operator(std::move(name), std::move(factory));
            return true;
        }
    };
#ifndef REGISTER_TO_DEFAULT_OPSET
#define REGISTER_TO_DEFAULT_OPSET(name) [[maybe_unused]] static auto _RegOp##name = ::lcml::onnx::OperatorSet::RegHelper(#name) << []([[maybe_unused]] ::lcml::onnx::Node const &node)
#endif
};

class LUISA_ONNX_API Model {
private:
    int32_t _ir_version;
    luisa::string _producer_name;
    luisa::string _producer_version;
    Graph _graph;
    OperatorSet *_opset = &OperatorSet::get_default();
public:
    Model() = default;
    Model(const Model &other) = default;
    Model(Model &&other) noexcept = default;
    Model &operator=(const Model &other) = default;
    Model &operator=(Model &&other) noexcept = default;
    ~Model() = default;

    Model(int32_t ir_version, luisa::string producer_name, luisa::string producer_version) : _ir_version(ir_version),
                                                                                               _producer_name(std::move(producer_name)),
                                                                                               _producer_version(std::move(producer_version)) {
    }

    void set_graph(Graph &&graph) { 
        this->_graph = std::move(graph);
        this->_graph.mark_constants();
     }
    void set_opset(OperatorSet &opset) { this->_opset = &opset; }
    Graph const &get_graph() const { return _graph; }
    OperatorSet &get_opset() const { return *_opset; }
    int32_t get_ir_version() const { return _ir_version; }
    luisa::string const &get_producer_name() const { return _producer_name; }
    luisa::string const &get_producer_version() const { return _producer_version; }

    static Model load_from_json(std::string_view json);
};

}// namespace lcml::onnx
