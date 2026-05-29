#pragma once

#include "luisa_ml_config.h"

#include <array>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <span>
#include <type_traits>
// NOTE: typeinfo is required for the type-dispatch system. This violates the project's no-RTTI rule.
#include <typeinfo>
#include <luisa/core/logging.h>
#include <luisa/core/mathematics.h>
#include <luisa/dsl/sugar.h>

namespace lcml::onnx {

/**
 * @brief Base class for Tensor, providing shape and stride operations.
 * This allows polymorphic usage of tensors without knowing the underlying element type.
 */
class ITensor {
public:
    using size_type = uint32_t;
    using shape_type = std::vector<size_type>;

protected:
    shape_type shape_;  // Dimensions of the tensor
    shape_type strides_;// Strides for each dimension

    // Calculate strides from shape (row-major order)
    void compute_strides() {
        strides_.resize(shape_.size());
        if (shape_.empty()) return;

        strides_.back() = 1;
        for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
    }

    // Calculate total number of elements
    size_type compute_size() const {
        if (shape_.empty()) return 0;
        return std::accumulate(shape_.begin(), shape_.end(),
                               size_type(1), std::multiplies<size_type>());
    }

    // Convert multi-dimensional index to linear index
    size_type linear_index_impl(const std::vector<size_type> &indices) const {
        size_type index = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            index += indices[i] * strides_[i];
        }
        return index;
    }

    // Convert multi-dimensional index to linear index
    template<typename... Indices>
    size_type linear_index(Indices... indices) const {
        static_assert(sizeof...(indices) > 0, "At least one index required");
        std::array<size_type, sizeof...(indices)> idx = {static_cast<size_type>(indices)...};
        return linear_index_impl(idx);
    }

    template<size_t N>
    size_type linear_index_impl(const std::array<size_type, N> &indices) const {
        size_type index = 0;
        for (size_t i = 0; i < N; ++i) {
            index += indices[i] * strides_[i];
        }
        return index;
    }

public:
    // Virtual destructor for proper cleanup
    virtual ~ITensor() = default;

    // ==================== Shape Operations ====================

    // Get the shape of the tensor
    shape_type const &shape() const noexcept { return shape_; }

    // Get the strides of the tensor
    shape_type const &strides() const noexcept { return strides_; }

    // Get the number of dimensions
    size_type ndim() const noexcept { return shape_.size(); }

    // Get the total number of elements
    virtual size_type size() const noexcept = 0;

    // Get the size of a specific dimension
    size_type shape(size_type dim) const { return shape_[dim]; }

    // Check if tensor is empty
    bool empty() const noexcept { return shape_.empty(); }

    // Reshape the tensor (must have same total size)
    void reshape(const shape_type &new_shape) {
        shape_ = new_shape;
        compute_strides();
    }

    // Reshape with span
    void reshape(std::span<size_type> new_shape) {
        reshape(shape_type(new_shape.begin(), new_shape.end()));
    }

    // Convert linear index to multi-dimensional index
    std::vector<size_type> unravel_index(size_type linear_idx) const {
        std::vector<size_type> indices(shape_.size());
        for (size_t i = 0; i < shape_.size(); ++i) {
            indices[i] = linear_idx / strides_[i];
            linear_idx %= strides_[i];
        }
        return indices;
    }

    // Get the type_info of the element type T (pure virtual)
    virtual std::type_info const &element_type() const noexcept = 0;

    virtual void set_name(std::string_view name) const noexcept {}

    virtual bool is_constant() const noexcept { return false; }

    /// Whether the underlying storage is a view of another tensor's data.
    virtual bool is_view() const noexcept { return false; }

    /// Whether the underlying storage is a local (shader-local) array.
    virtual bool is_local() const noexcept { return false; }

protected:
    // Protected constructor to prevent direct instantiation
    ITensor() = default;

    // Constructor with shape
    explicit ITensor(shape_type shape) : shape_(std::move(shape)) {
        compute_strides();
    }

    // Constructor with span shape
    explicit ITensor(std::span<size_type> shape)
        : shape_(shape.begin(), shape.end()) {
        compute_strides();
    }

    // Allow copy/move for derived classes
    ITensor(const ITensor &) = default;
    ITensor(ITensor &&) noexcept = default;
    ITensor &operator=(const ITensor &) = default;
    ITensor &operator=(ITensor &&) noexcept = default;
};

/**
 * @brief A lightweight Tensor class that supports arbitrary container types.
 * @tparam T The element type stored in the tensor.
 * @tparam Container The underlying container type (must support size() and operator[]).
 */
template<typename T, typename Container = std::vector<T>>
class Tensor : public ITensor {
public:
    // Type aliases
    using value_type = T;
    using container_type = Container;
    using reference = decltype(std::declval<Container>()[0]);
    using const_reference = decltype(std::declval<const Container>()[0]);

private:
    Container data_;// Underlying data storage

    // Use base class version for vector indices
    using ITensor::linear_index;
    using ITensor::linear_index_impl;

    // Compute linear index using DSL arithmetic for DSL index types.
    // Optimized: stride==0 dimensions are skipped, stride==1 avoids multiply.
    // Host-side zero indices (size_type == 0) are folded to skip the dimension entirely,
    // avoiding redundant 0*stride DSL nodes (e.g. Gemm M==1 with A(0u, k)).
    // All branches return Var<uint> for consistent type deduction.
    template<typename First, typename... Rest>
    auto dsl_linear_index_impl(size_t dim, First first, Rest... rest) const {
        using luisa::compute::def;
        using luisa::compute::cast;
        auto as_var = [](auto x) { return def(cast<uint32_t>(x)); };

        // Check if this dimension can be skipped:
        // stride==0 always skips; host-constant index==0 also skips (0*stride = 0).
        auto skip_dim = [&]() {
            if (strides_[dim] == 0) return true;
            if constexpr (std::is_convertible_v<First, size_type>) {
                return static_cast<size_type>(first) == 0;
            } else {
                return false;
            }
        };

        if constexpr (sizeof...(rest) > 0) {
            if (skip_dim()) {
                return dsl_linear_index_impl(dim + 1, rest...);
            }
            auto s = strides_[dim];
            if (s == 1) {
                return as_var(first) + dsl_linear_index_impl(dim + 1, rest...);
            } else {
                return as_var(first) * s + dsl_linear_index_impl(dim + 1, rest...);
            }
        } else {
            if (skip_dim()) {
                return def(0u);
            }
            auto s = strides_[dim];
            if (s == 1) {
                return as_var(first);
            } else {
                return as_var(first) * s;
            }
        }
    }

    template<typename... Indices>
    auto dsl_linear_index(Indices... indices) const {
        static_assert(sizeof...(indices) > 0, "At least one index required");
        return dsl_linear_index_impl(0, indices...);
    }

public:
    // ==================== Constructors ====================

    // Default constructor
    Tensor() = default;

    // Constructor with shape and existing container (move)
    Tensor(shape_type shape, Container data)
        : ITensor(std::move(shape)), data_(std::move(data)) {}

    // Constructor with span for shape and existing container (move)
    Tensor(std::span<size_type> shape, Container data)
        : ITensor(shape), data_(std::move(data)) {}

    // Copy constructor
    Tensor(const Tensor &other) = default;

    // Move constructor
    Tensor(Tensor &&other) noexcept = default;

    // Copy assignment
    Tensor &operator=(const Tensor &other) = default;

    // Move assignment
    Tensor &operator=(Tensor &&other) noexcept = default;

    // Destructor
    ~Tensor() override = default;

    // ==================== Override ====================

    // Get the total number of elements
    size_type size() const noexcept override { return data_.size(); }

    // Get the type_info of the element type T
    std::type_info const &element_type() const noexcept override {
        return typeid(T);
    }

    // Forward is_view / is_local to the container if it supports them
    bool is_view() const noexcept override {
        if constexpr (requires { data_.is_view(); }) {
            return data_.is_view();
        } else {
            return false;
        }
    }

    bool is_local() const noexcept override {
        if constexpr (requires { data_.is_local(); }) {
            return data_.is_local();
        } else {
            return false;
        }
    }

    // ==================== Element Access ====================

    // Access element with variadic indices (plain integral types)
    template<typename... Indices>
        requires(std::is_convertible_v<Indices, size_type> && ...)
    reference operator()(Indices... indices) {
        return data_[linear_index(indices...)];
    }

    template<typename... Indices>
        requires(std::is_convertible_v<Indices, size_type> && ...)
    const_reference operator()(Indices... indices) const {
        return data_[linear_index(indices...)];
    }

    // Access element with variadic DSL index types (e.g. UInt)
    template<typename... Indices>
        requires(!(std::is_convertible_v<Indices, size_type> && ...))
    auto operator()(Indices... indices) -> decltype(data_[std::declval<std::common_type_t<Indices...>>()]) {
        return data_[dsl_linear_index(indices...)];
    }

    template<typename... Indices>
        requires(!(std::is_convertible_v<Indices, size_type> && ...))
    auto operator()(Indices... indices) const -> decltype(data_[std::declval<std::common_type_t<Indices...>>()]) {
        return data_[dsl_linear_index(indices...)];
    }

    // Access element with vector of indices
    reference at(const std::vector<size_type> &indices) {
        return data_[linear_index_impl(indices)];
    }

    const_reference at(const std::vector<size_type> &indices) const {
        return data_[linear_index_impl(indices)];
    }
    // Linear access with DSL index types (e.g. UInt)
    template<typename Index>
    auto operator[](Index &&index) -> decltype(data_[std::forward<Index>(index)]) {
        return data_[std::forward<Index>(index)];
    }

    template<typename Index>
    auto operator[](Index &&index) const -> decltype(data_[std::forward<Index>(index)]) {
        return data_[std::forward<Index>(index)];
    }

    // Get reference to underlying container
    Container &container() noexcept { return data_; }
    Container const &container() const noexcept { return data_; }

    void set_name(std::string_view name) const noexcept override {
        if constexpr (requires { data_.set_name(name); }) {
            data_.set_name(name);
        }
    }

    // ==================== Utility Functions ====================

    // Swap with another tensor
    void swap(Tensor &other) noexcept {
        std::swap(data_, other.data_);
        std::swap(shape_, other.shape_);
        std::swap(strides_, other.strides_);
    }
};

// Non-member swap
template<typename T, typename Container>
void swap(Tensor<T, Container> &a, Tensor<T, Container> &b) noexcept {
    a.swap(b);
}

template<typename T, typename Container>
class ConstTensor : public Tensor<T, Container> {
public:
    using Base = Tensor<T, Container>;
    using storage_type = std::conditional_t<std::is_same_v<T, bool>, uint8_t, T>;
    using typename Base::shape_type;
    using typename Base::container_type;

private:
    std::vector<storage_type> storage_data_;

public:
    /**
     * @brief Construct a ConstTensor with shape, GPU container, and CPU-side data.
     * @param shape   Tensor dimensions
     * @param data    The DynamicArray container (GPU-side)
     * @param cpu_data A vector holding the same constant values on CPU
     */
    ConstTensor(shape_type shape, container_type data, std::vector<storage_type> cpu_data)
        : Base(std::move(shape), std::move(data)),
          storage_data_(std::move(cpu_data)) {}

    ~ConstTensor() override = default;

    // Move
    ConstTensor(ConstTensor &&) noexcept = default;
    ConstTensor &operator=(ConstTensor &&) noexcept = default;

    // No copy (due to DynamicArray)
    ConstTensor(ConstTensor const &) = delete;
    ConstTensor &operator=(ConstTensor const &) = delete;

    /// @brief Returns true, indicating this tensor holds constant data.
    [[nodiscard]] bool is_constant() const noexcept override { return true; }

    /// @brief Access the CPU-side constant data.
    [[nodiscard]] std::vector<storage_type> const &const_data() const noexcept {
        return storage_data_;
    }
};

}// namespace lcml::onnx
