#include "onnx/onnx.h"
#include "onnx/yyjson.hpp"

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/memory.h>

namespace lcml::onnx {
OperatorSet &OperatorSet::get_default() {
    static OperatorSet opset{"default", 21};
    return opset;
}

// Recursively collect all variable names that appear as node outputs
// in this graph and any subgraphs referenced by node attributes.
static void collect_output_names(Graph const &graph, luisa::unordered_set<luisa::string> &output_names) {
    for (auto const &node : graph.get_nodes()) {
        for (auto const &var_ref : node.get_outputs()) {
            output_names.insert(var_ref.get().get_name());
        }
        // Recurse into subgraphs stored in attributes
        for (auto const &[attr_name, attr] : node.get_attributes()) {
            auto attr_type = Attribute::type_of(attr);
            if (attr_type == AttributeType::GRAPH) {
                auto const &sub = attr.get<AttributeType::GRAPH>();
                collect_output_names(*sub, output_names);
            } else if (attr_type == AttributeType::GRAPHS) {
                for (auto const &sub : attr.get<AttributeType::GRAPHS>()) {
                    collect_output_names(*sub, output_names);
                }
            }
        }
    }
}

void Graph::mark_constants() {
    // Collect all variable names that are produced as node outputs
    // across this graph and all nested subgraphs.
    luisa::unordered_set<luisa::string> output_names;
    collect_output_names(*this, output_names);

    // Any variable that is never a node output is a constant
    for (auto &[name, var] : _variables) {
        var.set_is_constant(output_names.find(name) == output_names.end());
    }
}

template<class T = int32_t>
static luisa::optional<T> to_type(std::string_view input) {
    T out;
    const std::from_chars_result result = std::from_chars(input.data(), input.data() + input.size(), out);
    if (result.ec == std::errc{}) {
        return out;
    }
    return luisa::nullopt;
}

static constexpr int32_t decode_lut(uint8_t c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 71;
    if (c >= '0' && c <= '9') return c + 4;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}
static luisa::string base64_decode(std::string_view in) {
    luisa::string out;
    int32_t val = 0, valb = -8;
    for (char c : in) {
        if (c == '=') break;
        int32_t uc = decode_lut(c);
        if (uc == -1) {
            continue;
        }
        val = (val << 6) + uc;
        valb += 6;
        if (valb >= 0) {
            out.push_back(char((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

Variable Variable::from_json(json_cvalue const &json) {
    auto name = luisa::string{json["name"].as_string().value()};
    auto dtype = magic_enum::enum_cast<DataType>(
                     json["type"]["tensor_type"]["elem_type"].as_string().value())
                     .value();
    luisa::vector<size_t> shape;
    if (!json["type"]["tensor_type"]["shape"].contains("dim")) {
        shape.emplace_back(1);
    } else {
        for (auto const &dim : json["type"]["tensor_type"]["shape"]["dim"].as_array().value()) {
            shape.emplace_back(to_type(dim["dim_value"].as_string().value()).value());
        }
    }
    return Variable{std::move(name), dtype, std::move(shape)};
}

Variable Variable::from_initializer_json(json_cvalue const &json) {
    auto name = luisa::string{json["name"].as_string().value()};
    auto dtype = magic_enum::enum_cast<DataType>(json["data_type"].as_string().value()).value();
    luisa::vector<size_t> shape;
    if (!json.contains("dims")) {
        shape.emplace_back(1);
    } else {
        for (auto const &dim : json["dims"].as_array().value()) {
            shape.emplace_back(to_type(dim.as_string().value()).value());
        }
    }
    luisa::string raw_data;
    uint64_t buffer_start = 0;
    uint64_t buffer_end = 0;
    if (json.contains("data_offsets")) {
        auto data_offsets = json["data_offsets"].as_array().value();
        buffer_start = data_offsets[0].as_uint().value();
        buffer_end = data_offsets[1].as_uint().value();
    } else {
        raw_data = base64_decode(json["raw_data"].as_string().value());
    }
    return Variable{std::move(name), dtype, std::move(shape), std::move(raw_data), buffer_start, buffer_end};
}

Attribute Attribute::from_json(json_cvalue const &json, Graph const *parent) {
    auto attr_name = luisa::string{json["name"].as_string().value()};
    auto attr_type = magic_enum::enum_cast<AttributeType>(json["type"].as_string().value()).value();
    switch (attr_type) {
        case AttributeType::FLOAT:
            return {std::move(attr_name), static_cast<float>(json["f"].as_real().value())};
        case AttributeType::INT:
            return {std::move(attr_name), to_type<int32_t>(json["i"].as_string().value()).value()};
        case AttributeType::STRING:
            return {std::move(attr_name), base64_decode(json["s"].as_string().value())};
        case AttributeType::FLOATS: {
            luisa::vector<float> values;
            for (auto const &v : json["floats"].as_array().value()) {
                values.emplace_back(static_cast<float>(v.as_real().value()));
            }
            return {std::move(attr_name), std::move(values)};
        }
        case AttributeType::INTS: {
            luisa::vector<int32_t> values;
            for (auto const &v : json["ints"].as_array().value()) {
                values.emplace_back(to_type<int32_t>(v.as_string().value()).value());
            }
            return {std::move(attr_name), std::move(values)};
        }
        case AttributeType::STRINGS: {
            luisa::vector<luisa::string> values;
            for (auto const &v : json["strings"].as_array().value()) {
                values.emplace_back(base64_decode(v.as_string().value()));
            }
            return {std::move(attr_name), std::move(values)};
        }
        case AttributeType::GRAPH:
            // Note: requires OperatorSet context; parse with a default opset for now
            return {std::move(attr_name), luisa::make_shared<Graph>(Graph::from_json(json["g"], OperatorSet::get_default(), parent))};
        case AttributeType::GRAPHS: {
            luisa::vector<luisa::shared_ptr<Graph>> graphs;
            for (auto const &g : json["graphs"].as_array().value()) {
                graphs.emplace_back(luisa::make_shared<Graph>(Graph::from_json(g, OperatorSet::get_default(), parent)));
            }
            return {std::move(attr_name), std::move(graphs)};
        }
        default:
            LUISA_ASSERT(false, "Unsupported attribute type");
    }
}

Node Node::from_json(json_cvalue const &json, Graph const &graph) {
    Node node{luisa::string{json["name"].as_string().value()},
              luisa::string{json["op_type"].as_string().value()}};
    for (auto const &input : json["input"].as_array().value()) {
        node.add_input(graph.get_var(input.as_string().value()));
    }
    for (auto const &output : json["output"].as_array().value()) {
        node.add_output(graph.get_var(output.as_string().value()));
    }
    if (json.contains("attribute")) {
        for (auto const &attr : json["attribute"].as_array().value()) {
            node.set_attribute(Attribute::from_json(attr, &graph));
        }
    }
    return node;
}

Graph Graph::from_json(json_cvalue const &json, OperatorSet const &opset, Graph const *parent) {
    Graph graph{luisa::string{json["name"].as_string().value()}};
    graph.set_parent(parent);

    // Count total variables and reserve space to prevent reference invalidation.
    // _variables uses a dense hash map (vector-backed); without reservation,
    // adding variables can reallocate and invalidate reference_wrappers stored
    // in _inputs, _outputs, and Node::_inputs / Node::_outputs.
    size_t total_vars = 0;
    if (json.contains("input")) total_vars += json["input"].as_array().value().size();
    if (json.contains("output")) total_vars += json["output"].as_array().value().size();
    if (json.contains("value_info")) total_vars += json["value_info"].as_array().value().size();
    if (json.contains("initializer")) total_vars += json["initializer"].as_array().value().size();
    graph._variables.reserve(total_vars);

    if (json.contains("input")) {
        for (auto const &var : json["input"].as_array().value()) {
            graph.add_input(graph.add_variable(Variable::from_json(var)).get_name());
        }
    }
    if (json.contains("output")) {
        for (auto const &var : json["output"].as_array().value()) {
            graph.add_output(graph.add_variable(Variable::from_json(var)).get_name());
        }
    }
    if (json.contains("initializer")) {
        for (auto const &var : json["initializer"].as_array().value()) {
            graph.add_variable(Variable::from_initializer_json(var));
        }
    }
    if (json.contains("value_info")) {
        for (auto const &var : json["value_info"].as_array().value()) {
            graph.add_variable(Variable::from_json(var));
        }
    }

    for (auto const &n : json["node"].as_array().value()) {
        auto node = Node::from_json(n, graph);
        LUISA_ASSERT(opset.has_operator(node.get_op_type()),
                     "Operator not found: {}", node.get_op_type());
        graph.add_node(std::move(node));
    }
    return graph;
}

Model Model::load_from_json(std::string_view json) {
    auto doc = yyjson::read(json).as_object().value();
    auto model = Model(to_type(doc["ir_version"].as_string().value()).value(),
                       luisa::string{doc["producer_name"].as_string().value()},
                       luisa::string{doc["producer_version"].as_string().value()});
    model.set_graph(Graph::from_json(doc["graph"], model.get_opset()));
    return model;
}

}// namespace lcml::onnx
