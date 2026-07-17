#pragma once
#include "luisa_ml_config.h"
#include "local_data.h"
#include <utility>

namespace luisa::compute::dynamic_array {

template<typename T>
struct ViewData {
    LocalData<T> const *source;
    size_t offset;
    size_t size;

    explicit ViewData(size_t n, LocalData<T> const *src, size_t off) noexcept
        : source{src}, offset{off}, size{n} {
        LUISA_ASSERT(src->size >= off + n,
                     "DynamicArray view Local out of range");
    }

    ViewData(ViewData const &) noexcept = default;
    ViewData(ViewData &&) noexcept = default;
    ViewData &operator=(ViewData const &) = default;
    ViewData &operator=(ViewData &&) = default;

    template<typename U>
        requires is_integral_expr_v<U>
    [[nodiscard]] Var<T> &access(U &&index) const noexcept {
        if (offset == 0) {
            return source->access(std::forward<U>(index));
        }
        auto i = def(std::forward<U>(index));
        auto adjusted = i.template as<uint>() + static_cast<uint>(offset);
        return source->access(adjusted);
    }
};

}// namespace luisa::compute::dynamic_array
