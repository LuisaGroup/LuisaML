#pragma once

#include <string>
#include <vector>
#include <typeindex>
#include "onnx/onnx.h"

namespace lcml::onnx {

/// @brief Type alias for node index in the execution graph.
using NodeIndex = size_t;
/// @brief Type alias for color (register) identifier in graph coloring.
using ColorId = size_t;
/// @brief Type alias for tensor element counts and size classes.
using ElementCount = size_t;
/// @brief Type alias for the virtual execution schedule.
using Schedule = std::vector<NodeIndex>;
/// @brief Type alias for last-use map: variable name -> last node index.
using LastUseMap = onnx::StringNodeMap<NodeIndex>;

struct AllocationUnit {
    std::string name;
    std::type_index type_idx;
    ElementCount num_elements = 0;
    NodeIndex def_node = 0;     // virtual index where first defined
    NodeIndex last_use_node = 0;// virtual index of last use
    bool is_view = false;
    bool is_inplace = false;
    bool skip_allocation = false;

    AllocationUnit() : type_idx(typeid(void)) {}
    AllocationUnit(std::string name, std::type_index type_idx, ElementCount num_elements,
                   NodeIndex def_node, NodeIndex last_use_node)
        : name(std::move(name)), type_idx(type_idx), num_elements(num_elements),
          def_node(def_node), last_use_node(last_use_node) {}
};

struct ColorSlot {
    ColorId color_id = 0;
    std::type_index type_idx = std::type_index(typeid(void));
    ElementCount size_class = 0;
    std::vector<std::string> members;
    // If this slot borrows from an external (parent) slot, stores the external slot index.
    // SIZE_MAX means this slot owns its own storage and is not borrowed.
    size_t borrowed_slot_index = SIZE_MAX;
};

struct TensorMapping {
    ColorId color_id = 0;
    bool is_view = false;
    bool is_inplace = false;
    std::string view_source;
    std::string root_name;
};

struct AllocationPlan {
    std::vector<ColorSlot> color_slots;
    onnx::StringNodeMap<TensorMapping> tensor_map;
    Schedule schedule;// virtual execution order: schedule[vi] = orig_node_idx
    ElementCount total_intermediates = 0;
    ColorId total_physical_slots = 0;
    ElementCount view_merges = 0;
    ElementCount inplace_merges = 0;
};

/// @brief Describes a reusable storage slot from a parent graph or sibling branch
///        that a subgraph's RegisterAllocator can borrow during graph coloring.
struct ExternalSlot {
    std::type_index type_idx = std::type_index(typeid(void));
    ElementCount capacity = 0;///< Number of elements available in this storage
    size_t external_index = 0;///< Caller-assigned index to identify this slot
    bool exclusive = false;   ///< If true, only one branch can borrow this slot
                              ///< (used for sibling-branch sharing where the slot
                              ///< itself is also borrowed and not concurrently usable)
};

}// namespace lcml::onnx
