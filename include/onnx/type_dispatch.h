#pragma once

#include "luisa_ml_config.h"

#include <stdexcept>
#include <string>
#include <tuple>
#include <cstring>

#include <luisa/core/stl/hash.h>
#include "onnx/type_name.h"

namespace lcml::onnx::refl {

using TypeIndex = uint64_t;

template<typename T>
constexpr TypeIndex type_index_of() noexcept {
    return luisa::hash_value(type_raw_name_v<T>);
}

template<auto V>
struct Value {
    static constexpr auto value = V;
};

namespace detail {

// Helper to get the first type from a parameter pack
template<typename T, typename...>
struct FirstType {
    using type = T;
};

// Dispatcher using linear search over types
template<typename... Ts>
class TypeDispatcher {
    using FirstT = typename FirstType<Ts...>::type;

public:
    template<typename F>
    static auto visit(TypeIndex idx, F &&f) -> decltype(auto) {
        using ReturnType = decltype(std::declval<F &>().template operator()<FirstT>());
        return visit_impl<0, F, ReturnType>(idx, f);
    }

private:
    template<size_t I, typename F, typename ReturnType>
    static auto visit_impl(TypeIndex idx, F &f) -> ReturnType {
        if constexpr (I < sizeof...(Ts)) {
            using T = std::tuple_element_t<I, std::tuple<Ts...>>;
            if (type_index_of<T>() == idx) {
                if constexpr (std::is_void_v<ReturnType>) {
                    f.template operator()<T>();
                } else {
                    return f.template operator()<T>();
                }
            } else {
                return visit_impl<I + 1, F, ReturnType>(idx, f);
            }
        } else {
            LUISA_ASSERT(false, "visit_type_index: unsupported type index {}", idx);
        }
    }
};

// Unpack std::tuple into TypeDispatcher
template<typename TupleT>
struct TupleVisitor;

template<typename... Ts>
struct TupleVisitor<std::tuple<Ts...>> {
    template<typename F>
    static auto visit(TypeIndex idx, F &&f) -> decltype(auto) {
        return TypeDispatcher<Ts...>::visit(idx, std::forward<F>(f));
    }
};

}// namespace detail

/**
 * @brief Visit a type by its type index, invoking a generic lambda with the matching type.
 *
 * Uses linear search. Safe across DLL boundaries.
 *
 * @tparam Types A std::tuple<T1, T2, ...> of supported types
 * @tparam F A generic lambda that can be instantiated with any type in Types
 * @param idx The type index to match against
 * @param f The generic lambda to invoke
 * @return The result of invoking f<MatchedType>()
 *
 * @example
 * ```cpp
 * using SupportedTypes = std::tuple<int, float, double>;
 *
 * TypeIndex idx = tensor.element_type_index();
 * visit_type_index<SupportedTypes>(idx, [&]<typename T>() {
 *     // T is the matched type
 *     auto& data = static_cast<Tensor<T>&>(tensor);
 *     // ... do something with data
 * });
 * ```
 */
template<typename Types, typename F>
auto visit_type_index(TypeIndex idx, F &&f) -> decltype(auto) {
    return detail::TupleVisitor<Types>::visit(idx, std::forward<F>(f));
}

/**
 * @brief Convenience overload that accepts types directly instead of std::tuple.
 *
 * @example
 * ```cpp
 * visit_type_index<int, float, double>(idx, [&]<typename T>() {
 *     // ...
 * });
 * ```
 */
template<typename T1, typename T2, typename... Rest, typename F>
auto visit_type_index(TypeIndex idx, F &&f) -> decltype(auto) {
    return detail::TypeDispatcher<T1, T2, Rest...>::visit(idx, std::forward<F>(f));
}
}// namespace refl

namespace luisa {
using TypeIndex = lcml::onnx::refl::TypeIndex;
}
