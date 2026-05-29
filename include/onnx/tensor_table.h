#pragma once

#include <memory>
#include <functional>
#include <variant>
#include <stdexcept>
#include "tensor.h"
#include "onnx.h"

namespace lcml::onnx {

/**
 * @brief A single entry in a TensorTable.
 *
 * Holds either an owned tensor (unique_ptr) or a borrowed reference (reference_wrapper).
 * Provides uniform access to the underlying ITensor regardless of ownership.
 */
class TensorEntry {
public:
    using OwnedPtr = std::unique_ptr<ITensor>;
    using BorrowedRef = std::reference_wrapper<ITensor>;

private:
    std::variant<OwnedPtr, BorrowedRef> storage_;

public:
    // Construct an owned entry
    explicit TensorEntry(OwnedPtr ptr) : storage_(std::move(ptr)) {}

    // Construct a borrowed entry
    explicit TensorEntry(BorrowedRef ref) : storage_(ref) {}
    explicit TensorEntry(ITensor &ref) : storage_(std::ref(ref)) {}

    // Move-only (due to unique_ptr)
    TensorEntry(TensorEntry &&) noexcept = default;
    TensorEntry &operator=(TensorEntry &&) noexcept = default;
    TensorEntry(TensorEntry const &) = delete;
    TensorEntry &operator=(TensorEntry const &) = delete;

    ~TensorEntry() = default;

    /// @brief Whether this entry owns its tensor.
    [[nodiscard]] bool is_owned() const noexcept {
        return std::holds_alternative<OwnedPtr>(storage_);
    }

    /// @brief Whether this entry borrows (references) its tensor.
    [[nodiscard]] bool is_borrowed() const noexcept {
        return std::holds_alternative<BorrowedRef>(storage_);
    }

    /// @brief Get a reference to the underlying tensor.
    [[nodiscard]] ITensor &get() {
        if (auto *p = std::get_if<OwnedPtr>(&storage_)) {
            return **p;
        }
        return std::get<BorrowedRef>(storage_).get();
    }

    /// @brief Get a const reference to the underlying tensor.
    [[nodiscard]] ITensor const &get() const {
        if (auto *p = std::get_if<OwnedPtr>(&storage_)) {
            return **p;
        }
        return std::get<BorrowedRef>(storage_).get();
    }

    /// @brief Implicit conversion to ITensor reference for convenience.
    operator ITensor &() { return get(); }
    operator ITensor const &() const { return get(); }

    /**
     * @brief Extract the owned unique_ptr, leaving this entry in a moved-from state.
     *
     * Must only be called when is_owned() is true.
     * After this call, the entry should be erased from any containing table.
     *
     * @return The owned unique_ptr<ITensor>.
     */
    [[nodiscard]] OwnedPtr release_owned() {
        auto *ptr = std::get_if<OwnedPtr>(&storage_);
        return std::move(*ptr);
    }
};

/**
 * @brief A table mapping tensor names to TensorEntry instances.
 *
 * Supports both owned tensors (created internally, e.g. intermediates/weights)
 * and borrowed tensors (external input/output references).
 * This unifies the previous OwnedTensorTable and TensorMap into a single container.
 */
class TensorTable {
public:
    using MapType = onnx::StringNodeMap<TensorEntry>;
    using iterator = MapType::iterator;
    using const_iterator = MapType::const_iterator;

private:
    MapType map_;
    TensorTable *parent_ = nullptr;

public:
    TensorTable() = default;
    ~TensorTable() = default;

    // Move-only (TensorEntry is move-only)
    TensorTable(TensorTable &&) noexcept = default;
    TensorTable &operator=(TensorTable &&) noexcept = default;
    TensorTable(TensorTable const &) = delete;
    TensorTable &operator=(TensorTable const &) = delete;

    // ==================== Parent (fallback) ====================

    /// @brief Set a parent table for fallback lookups.
    void set_parent(TensorTable *parent) noexcept { parent_ = parent; }

    /// @brief Get the parent table (may be nullptr).
    [[nodiscard]] TensorTable *parent() const noexcept { return parent_; }

    // ==================== Insertion ====================

    /// @brief Bind a borrowed (external) tensor reference by name.
    void bind(std::string name, ITensor &tensor) {
        map_.insert_or_assign(std::move(name), TensorEntry{std::ref(tensor)});
    }

    /// @brief Insert an owned tensor by name.
    void own(std::string name, std::unique_ptr<ITensor> tensor) {
        map_.insert_or_assign(std::move(name), TensorEntry{std::move(tensor)});
    }

    // ==================== Lookup ====================

    /// @brief Get the tensor reference by name. Falls back to parent if not found locally.
    [[nodiscard]] ITensor &at(std::string_view name) {
        auto it = map_.find(name);
        if (it != map_.end()) return it->second.get();
        if (parent_) return parent_->at(name);
        LUISA_ASSERT(false, "TensorTable: key not found: {}", name);
    }

    /// @brief Get the const tensor reference by name. Falls back to parent if not found locally.
    [[nodiscard]] ITensor const &at(std::string_view name) const {
        auto it = map_.find(name);
        if (it != map_.end()) return it->second.get();
        if (parent_) return parent_->at(name);
        LUISA_ASSERT(false, "TensorTable: key not found: {}", name);
    }

    /// @brief Check if a tensor with the given name exists (including parent).
    [[nodiscard]] bool contains(std::string_view name) const {
        if (map_.find(name) != map_.end()) return true;
        return parent_ && parent_->contains(name);
    }

    /// @brief Check if a tensor with the given name exists locally only (no parent fallback).
    [[nodiscard]] bool contains_local(std::string_view name) const {
        return map_.find(name) != map_.end();
    }

    /// @brief Get the TensorEntry by name (for ownership inspection). Throws if not found.
    [[nodiscard]] TensorEntry &entry(std::string_view name) {
        auto it = map_.find(name);
        LUISA_ASSERT(it != map_.end(), "TensorTable: key not found: {}", name);
        return it->second;
    }

    /// @brief Get the const TensorEntry by name. Throws if not found.
    [[nodiscard]] TensorEntry const &entry(std::string_view name) const {
        auto it = map_.find(name);
        LUISA_ASSERT(it != map_.end(), "TensorTable: key not found: {}", name);
        return it->second;
    }

    // ==================== Capacity ====================

    [[nodiscard]] size_t size() const noexcept { return map_.size(); }
    [[nodiscard]] bool empty() const noexcept { return map_.empty(); }

    void clear() noexcept { map_.clear(); }

    /**
     * @brief Release ownership of a tensor entry, removing it from the table.
     *
     * The entry must exist locally and must be owned (not borrowed).
     * Returns the unique_ptr, transferring ownership to the caller.
     *
     * @param name  The name of the tensor to release.
     * @return The owned unique_ptr<ITensor>, or nullptr if not found/not owned.
     */
    [[nodiscard]] std::unique_ptr<ITensor> release_owned(std::string_view name) {
        auto it = map_.find(name);
        if (it == map_.end()) return nullptr;
        auto &entry = it->second;
        if (!entry.is_owned()) return nullptr;
        auto result = entry.release_owned();
        map_.erase(it);
        return result;
    }

    // ==================== Iterators ====================

    iterator begin() noexcept { return map_.begin(); }
    iterator end() noexcept { return map_.end(); }
    const_iterator begin() const noexcept { return map_.begin(); }
    const_iterator end() const noexcept { return map_.end(); }
    const_iterator cbegin() const noexcept { return map_.cbegin(); }
    const_iterator cend() const noexcept { return map_.cend(); }
};

}// namespace lcml::onnx
