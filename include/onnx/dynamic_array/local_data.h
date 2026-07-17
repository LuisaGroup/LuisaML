#pragma once
#include "luisa_ml_config.h"
#include <utility>
#include <luisa/dsl/shared.h>

namespace luisa::compute::dynamic_array {

template<typename T>
struct LocalData {
    RefExpr const *expression;
    bool as_shared;
    size_t size;

    LocalData() noexcept : expression{nullptr}, as_shared{false}, size{0} {}

    // Construct from an existing Local<T>
    explicit LocalData(Local<T> &&local)
        : expression{local.expression()}, as_shared{false}, size{local.size()} {}

    // Construct a new local/shared array of n elements
    explicit LocalData(size_t n, bool shared = false) noexcept
        : as_shared{shared}, size{n} {
        expression = shared ? detail::FunctionBuilder::current()->shared(
                                  Type::array(Type::of<T>(), n)) :
                              detail::FunctionBuilder::current()->local(
                                  detail::local_array_choose_type<T>(n));
    }

    // Deep copy: creates a new local array and assigns from source
    LocalData(LocalData const &other) noexcept : as_shared{false}, size{other.size} {
        if (other.as_shared) [[unlikely]] {
            LUISA_ERROR("Shared not allowed copy.");
        }
        auto fb = detail::FunctionBuilder::current();
        expression = fb->local(detail::local_array_choose_type<T>(size));
        fb->assign(expression, other.expression);
    }

    LocalData(LocalData &&) noexcept = default;
    LocalData &operator=(LocalData const &) = default;
    LocalData &operator=(LocalData &&) = default;

    void set_name(string_view name) const noexcept {
        auto *fb = expression->builder();
        auto &&var = expression->variable();
        fb->set_variable_name(var.uid(), name);
    }

    [[nodiscard]] auto type() const noexcept {
        return expression->type();
    }

    template<typename U>
        requires is_integral_expr_v<U>
    [[nodiscard]] Var<T> &access(U &&index) const noexcept {
        auto f = detail::FunctionBuilder::current();
        auto i = def(std::forward<U>(index));
        if (size == 1u) {
            return *f->create_temporary<Var<T>>(expression);
        }
        auto expr = f->access(Type::of<T>(), expression, i.expression());
        return *f->create_temporary<Var<T>>(expr);
    }
};

}// namespace luisa::compute::dynamic_array
