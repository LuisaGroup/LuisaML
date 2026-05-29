#pragma once
#include "luisa_ml_config.h"
#include <cstddef>

namespace luisa::compute::dynamic_array {

template<typename T>
struct ScalarData {
    T cpu_value;
    size_t size;

    explicit ScalarData(T value, size_t n = 1u) noexcept
        : cpu_value{value}, size{n} {}

    ScalarData(ScalarData const &) noexcept = default;
    ScalarData(ScalarData &&) noexcept = default;
    ScalarData &operator=(ScalarData const &) = default;
    ScalarData &operator=(ScalarData &&) = default;

    template<typename U>
        requires is_integral_expr_v<U>
    [[nodiscard]] Var<T> &access(U &&) const noexcept {
        auto f = detail::FunctionBuilder::current();
        if constexpr (requires(T x) { x.bits; }) {
            auto v = Var<T>{};
            v.bits = def(cpu_value.bits);
            return *f->create_temporary<Var<T>>(v.expression());
        } else {
            auto v = def(T{cpu_value});
            return *f->create_temporary<Var<T>>(v.expression());
        }
    }
};

}// namespace luisa::compute::dynamic_array
