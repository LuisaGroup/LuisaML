#pragma once
#include "luisa_ml_config.h"
#include <utility>
#include <type_traits>
#include <luisa/core/stl/variant.h>
#include "local_data.h"
#include "buffer_data.h"
#include "view_data.h"
#include "scalar_data.h"
#include "linear_data.h"
#include "fp4_data.h"
#include "fp8_data.h"

namespace luisa::compute {

using namespace dynamic_array;

template<typename T>
class DynamicArray {

public:
    using storage_t = luisa::variant<
        LocalData<T>,
        BufferData<T>,
        ViewData<T>,
        ScalarData<T>,
        LinearData<T>,
        FP4Data<T>,
        FP8Data<T>>;

    // Tag type for constructing Scalar-mode DynamicArray
    struct scalar_tag_t {};
    static constexpr scalar_tag_t scalar_tag{};

    // Tag type for constructing Linear-mode DynamicArray
    struct linear_tag_t {};
    static constexpr linear_tag_t linear_tag{};

private:
    storage_t _data;

public:
    // Construct from an existing Local<T>
    explicit DynamicArray(Local<T> &&local)
        : _data{LocalData<T>{std::move(local)}} {}

    // Construct a Scalar-mode DynamicArray from a CPU constant value
    explicit DynamicArray(scalar_tag_t, T cpu_value, size_t n = 1u) noexcept
        : _data{ScalarData<T>{cpu_value, n}} {}

    // Construct a Linear-mode DynamicArray (arithmetic sequence)
    explicit DynamicArray(linear_tag_t, T start, T delta, size_t n) noexcept
        : _data{LinearData<T>{start, delta, n}} {}

    // Construct a new local/shared array of n elements
    explicit DynamicArray(size_t n, bool as_shared = false) noexcept
        : _data{LocalData<T>{n, as_shared}} {}

    // Construct from a ByteBuffer pointer with size and byte offset
    explicit DynamicArray(size_t n, Var<ByteBuffer> *buffer,
                          size_t byte_offset = 0, size_t byte_size = ~0ull) noexcept {
        if constexpr (std::is_same_v<T, FP4E2M1>) {
            _data = FP4Data<T>{n, buffer, byte_offset, byte_size};
        } else if constexpr (std::is_same_v<T, FP8E4M3FN> || std::is_same_v<T, FP8E5M2>) {
            _data = FP8Data<T>{n, buffer, byte_offset, byte_size};
        } else {
            _data = BufferData<T>{n, buffer, byte_offset, byte_size};
        }
    }

    /**
     * @brief Construct a non-owning view into another DynamicArray.
     * If the source is itself a View, we flatten the chain.
     * If the source is a ByteBuffer, we become a ByteBuffer with combined offset.
     */
    explicit DynamicArray(size_t n, DynamicArray const &source, size_t offset = 0) noexcept {
        // Flatten View-of-View chains: walk to the root source
        DynamicArray const *root = &source;
        size_t accumulated_offset = offset;
        while (auto *vp = luisa::get_if<ViewData<T>>(&root->_data)) {
            accumulated_offset += vp->offset;
            // Walk up: ViewData stores a pointer to LocalData, not DynamicArray.
            // We have reached the root Local already; break.
            break;
        }

        // Handle each variant case
        if (auto *sp = luisa::get_if<ScalarData<T>>(&root->_data)) {
            _data = ScalarData<T>{sp->cpu_value, n};
            return;
        }
        if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>) {
            if (auto *lp = luisa::get_if<LinearData<T>>(&root->_data)) {
                _data = LinearData<T>{
                    static_cast<T>(lp->start + static_cast<T>(accumulated_offset) * lp->delta),
                    lp->delta, n};
                return;
            }
        }
        if (auto *bp = luisa::get_if<BufferData<T>>(&root->_data)) {
            LUISA_ASSERT(accumulated_offset * sizeof(T) + n * sizeof(T) + bp->byte_offset <= bp->byte_size,
                         "DynamicArray view ByteBuffer out of range");
            _data = BufferData<T>{n, bp->byte_buffer,
                                  bp->byte_offset + accumulated_offset * sizeof(T),
                                  bp->byte_size};
            return;
        }
        if (auto *fp4 = luisa::get_if<FP4Data<T>>(&root->_data)) {
            LUISA_ASSERT(accumulated_offset % 8 == 0,
                         "DynamicArray FP4 view offset must be a multiple of 8 elements");
            auto element_bytes = accumulated_offset / 8 * 4;
            LUISA_ASSERT(element_bytes + ((n + 7) / 8) * 4 + fp4->byte_offset <= fp4->byte_size,
                         "DynamicArray view FP4 ByteBuffer out of range");
            _data = FP4Data<T>{n, fp4->byte_buffer,
                               fp4->byte_offset + element_bytes,
                               fp4->byte_size};
            return;
        }
        if (auto *fp8 = luisa::get_if<FP8Data<T>>(&root->_data)) {
            LUISA_ASSERT(accumulated_offset % 4 == 0,
                         "DynamicArray FP8 view offset must be a multiple of 4 elements");
            auto element_bytes = accumulated_offset / 4 * 4;
            LUISA_ASSERT(element_bytes + ((n + 3) / 4) * 4 + fp8->byte_offset <= fp8->byte_size,
                         "DynamicArray view FP8 ByteBuffer out of range");
            _data = FP8Data<T>{n, fp8->byte_buffer,
                               fp8->byte_offset + element_bytes,
                               fp8->byte_size};
            return;
        }
        if (auto *loc = luisa::get_if<LocalData<T>>(&root->_data)) {
            LUISA_ASSERT(loc->size >= accumulated_offset + n,
                         "DynamicArray view Local out of range");
            _data = ViewData<T>{n, loc, accumulated_offset};
            return;
        }
        // ViewData root — get the underlying LocalData
        auto *vp = luisa::get_if<ViewData<T>>(&root->_data);
        LUISA_ASSERT(vp != nullptr, "DynamicArray view failed");
        auto total_off = accumulated_offset + vp->offset;
        LUISA_ASSERT(vp->source->size >= total_off + n,
                     "DynamicArray view Local out of range");
        _data = ViewData<T>{n, vp->source, total_off};
    }

    DynamicArray(DynamicArray &&) noexcept = default;
    DynamicArray(const DynamicArray &another) noexcept
        : _data{luisa::visit([](auto const &d) -> storage_t {
              return storage_t{d};
          },
                           another._data)} {}

    void set_name(string_view name) const noexcept {
        if (auto *loc = luisa::get_if<LocalData<T>>(&_data)) {
            loc->set_name(name);
        }
        // Scalar/Linear/View/Buffer modes — no-op
    }

    DynamicArray &operator=(const DynamicArray &rhs) noexcept {
        if (std::addressof(rhs) != this) [[likely]] {
            this->~DynamicArray();
            new (this) DynamicArray(rhs);
        }
        return *this;
    }
    DynamicArray &operator=(DynamicArray &&rhs) noexcept {
        *this = static_cast<const DynamicArray &>(rhs);
        return *this;
    }

    [[nodiscard]] auto size() const noexcept {
        return luisa::visit([](auto const &d) { return d.size; }, _data);
    }
    [[nodiscard]] bool is_byte_buffer() const noexcept {
        return luisa::holds_alternative<BufferData<T>>(_data) ||
               luisa::holds_alternative<FP4Data<T>>(_data) ||
               luisa::holds_alternative<FP8Data<T>>(_data);
    }
    [[nodiscard]] bool is_view() const noexcept {
        return luisa::holds_alternative<ViewData<T>>(_data);
    }
    [[nodiscard]] bool is_local() const noexcept {
        return luisa::holds_alternative<LocalData<T>>(_data);
    }
    [[nodiscard]] bool is_scalar_mode() const noexcept {
        return luisa::holds_alternative<ScalarData<T>>(_data);
    }
    [[nodiscard]] bool is_linear_mode() const noexcept {
        return luisa::holds_alternative<LinearData<T>>(_data);
    }

    /// Check whether this DynamicArray and `other` have overlapping storage.
    [[nodiscard]] bool shares_storage_with(DynamicArray const &other) const noexcept {
        struct Resolved {
            RefExpr const *local_expr = nullptr;
            Var<ByteBuffer> *bb_ptr = nullptr;
            size_t offset = 0;
            size_t count = 0;
        };
        auto resolve = [](DynamicArray const *p) -> Resolved {
            return luisa::visit([&](auto const &d) -> Resolved {
                using D = std::decay_t<decltype(d)>;
                if constexpr (std::is_same_v<D, ViewData<T>>) {
                    return {d.source->expression, nullptr, d.offset, d.source->size};
                } else if constexpr (std::is_same_v<D, LocalData<T>>) {
                    return {d.expression, nullptr, 0, d.size};
                } else if constexpr (std::is_same_v<D, BufferData<T>> ||
                                      std::is_same_v<D, FP4Data<T>> ||
                                      std::is_same_v<D, FP8Data<T>>) {
                    return {nullptr, d.byte_buffer, d.byte_offset, d.size * sizeof(T)};
                } else {
                    return {};
                }
            },
                              p->_data);
        };
        auto a = resolve(this);
        auto b = resolve(&other);
        if (a.local_expr && a.local_expr == b.local_expr) {
            auto a_size = size();
            auto b_size = other.size();
            return a.offset < b.offset + b_size && b.offset < a.offset + a_size;
        }
        if (a.bb_ptr && a.bb_ptr == b.bb_ptr) {
            return a.offset < b.offset + b.count && b.offset < a.offset + a.count;
        }
        return false;
    }

    [[nodiscard]] auto expression() const noexcept {
        auto *loc = luisa::get_if<LocalData<T>>(&_data);
        LUISA_ASSERT(loc != nullptr, "Only Local mode has an expression");
        return loc->expression;
    }

    [[nodiscard]] auto type() const noexcept {
        return luisa::visit([](auto const &d) {
            using D = std::decay_t<decltype(d)>;
            if constexpr (std::is_same_v<D, ScalarData<T>> || std::is_same_v<D, LinearData<T>>) {
                return Type::of<T>();
            } else if constexpr (std::is_same_v<D, LocalData<T>>) {
                return d.expression->type();
            } else if constexpr (std::is_same_v<D, ViewData<T>>) {
                return d.source->expression->type();
            } else {
                // BufferData
                return Type::of<T>();
            }
        },
                          _data);
    }

    [[nodiscard]] T scalar_cpu_value() const noexcept {
        auto *sp = luisa::get_if<ScalarData<T>>(&_data);
        LUISA_ASSERT(sp != nullptr, "Not in Scalar mode");
        return sp->cpu_value;
    }

    void set_size(size_t n) noexcept {
        luisa::visit([n](auto &d) { d.size = n; }, _data);
    }

    void set_byte_offset(size_t offset) noexcept {
        luisa::visit([offset](auto &d) {
            using D = std::decay_t<decltype(d)>;
            if constexpr (std::is_same_v<D, BufferData<T>> ||
                          std::is_same_v<D, FP4Data<T>> ||
                          std::is_same_v<D, FP8Data<T>>) {
                d.byte_offset = offset;
            } else {
                LUISA_ERROR_WITH_LOCATION("Not in ByteBuffer mode");
            }
        }, _data);
    }

    void set_byte_buffer(Var<ByteBuffer> *buffer) noexcept {
        luisa::visit([buffer](auto &d) {
            using D = std::decay_t<decltype(d)>;
            if constexpr (std::is_same_v<D, BufferData<T>> ||
                          std::is_same_v<D, FP4Data<T>> ||
                          std::is_same_v<D, FP8Data<T>>) {
                d.byte_buffer = buffer;
            } else {
                LUISA_ERROR_WITH_LOCATION("Not in ByteBuffer mode");
            }
        }, _data);
    }

    [[nodiscard]] Var<ByteBuffer> *get_byte_buffer() const noexcept {
        return luisa::visit([](auto const &d) -> Var<ByteBuffer> * {
            using D = std::decay_t<decltype(d)>;
            if constexpr (std::is_same_v<D, BufferData<T>> ||
                          std::is_same_v<D, FP4Data<T>> ||
                          std::is_same_v<D, FP8Data<T>>) {
                return d.byte_buffer;
            } else {
                LUISA_ERROR_WITH_LOCATION("Not in ByteBuffer mode");
            }
        }, _data);
    }

    [[nodiscard]] size_t get_byte_offset() const noexcept {
        return luisa::visit([](auto const &d) -> size_t {
            using D = std::decay_t<decltype(d)>;
            if constexpr (std::is_same_v<D, BufferData<T>> ||
                          std::is_same_v<D, FP4Data<T>> ||
                          std::is_same_v<D, FP8Data<T>>) {
                return d.byte_offset;
            } else {
                LUISA_ERROR_WITH_LOCATION("Not in ByteBuffer mode");
            }
        }, _data);
    }

    template<typename U>
        requires is_integral_expr_v<U>
    [[nodiscard]] Var<T> &operator[](U &&index) const noexcept {
        return luisa::visit([&](auto const &d) -> Var<T> & {
            return d.access(std::forward<U>(index));
        },
                          _data);
    }

    // Access the underlying variant storage
    [[nodiscard]] storage_t const &data() const noexcept { return _data; }
    [[nodiscard]] storage_t &data() noexcept { return _data; }
};

}// namespace luisa::compute
