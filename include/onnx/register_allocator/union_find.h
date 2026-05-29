#pragma once

#include <string>
#include "allocation_types.h"

namespace lcml::onnx {

// Disjoint-set data structure for merging allocation units
class UnionFind {
public:
    void init(std::string const &x) {
        if (parent_.find(x) == parent_.end()) {
            parent_[x] = x;
            rank_[x] = 0;
        }
    }

    std::string find(std::string const &x) const {
        auto it = parent_.find(x);
        if (it == parent_.end()) {
            parent_.emplace(x, x);
            rank_.emplace(x, 0);
            return x;
        }
        if (it->second != x) {
            it->second = find(it->second);
        }
        return it->second;
    }

    std::string unite(std::string const &a, std::string const &b) {
        auto ra = find(a);
        auto rb = find(b);
        if (ra == rb) return ra;
        if (rank_[ra] < rank_[rb]) std::swap(ra, rb);
        parent_[rb] = ra;
        if (rank_[ra] == rank_[rb]) rank_[ra]++;
        return ra;
    }

    bool connected(std::string const &a, std::string const &b) const {
        return find(a) == find(b);
    }

private:
    mutable onnx::StringNodeMap<std::string> parent_;
    mutable onnx::StringNodeMap<NodeIndex> rank_;
};

}// namespace lcml::onnx
