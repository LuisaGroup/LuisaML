#pragma once
#include "luisa_ml_config.h"
#include <cstddef>
#include <type_traits>

namespace luisa::compute::dynamic_array {

template<typename T>
struct LinearData {
    T start;
    T delta;
    size_t size;

    explicit LinearData(T s, T d, size_t n) noexcept
        : start{s}, delta{d}, size{n} {}

    LinearData(LinearData const &) noexcept = default;
    LinearData(LinearData &&) noexcept = default;
    LinearData &operator=(LinearData const &) = default;
    LinearData &operator=(LinearData &&) = default;

    template<typename U>
        requires is_integral_expr_v<U>
    [[nodiscard]] Var<T> &access(U &&index) const noexcept {
        auto f = detail::FunctionBuilder::current();
        auto i = def(std::forward<U>(index));
        if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>) {
            using cast_t = std::conditional_t<std::is_integral_v<T>, int, T>;
            auto v = def(static_cast<cast_t>(start)) +
                     def(static_cast<cast_t>(delta)) * i.template cast<cast_t>();
            return *f->template create_temporary<Var<T>>(v.template cast<T>().expression());
        } else {
            LUISA_ERROR_WITH_LOCATION("Linear mode not supported for bool type");
        }
    }
};

}// namespace luisa::compute::dynamic_array
