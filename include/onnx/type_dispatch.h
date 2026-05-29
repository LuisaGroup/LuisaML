#pragma once

// NOTE: typeinfo is required for the type-dispatch system. This violates the project's no-RTTI rule.
#include <typeinfo>
#include <stdexcept>
#include <string>
#include <tuple>
#include <cstring>

namespace lcml::onnx::refl {

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
    static auto visit(const std::type_info &tid, F &&f) -> decltype(auto) {
        using ReturnType = decltype(std::declval<F &>().template operator()<FirstT>());
        return visit_impl<0, F, ReturnType>(tid, f);
    }

private:
    template<size_t I, typename F, typename ReturnType>
    static auto visit_impl(const std::type_info &tid, F &f) -> ReturnType {
        if constexpr (I < sizeof...(Ts)) {
            using T = std::tuple_element_t<I, std::tuple<Ts...>>;
            if (std::strcmp(typeid(T).name(), tid.name()) == 0) {
                if constexpr (std::is_void_v<ReturnType>) {
                    f.template operator()<T>();
                } else {
                    return f.template operator()<T>();
                }
            } else {
                return visit_impl<I + 1, F, ReturnType>(tid, f);
            }
        } else {
            LUISA_ASSERT(false, "visit_typeid: unsupported type {}", tid.name());
        }
    }
};

// Unpack std::tuple into TypeDispatcher
template<typename TupleT>
struct TupleVisitor;

template<typename... Ts>
struct TupleVisitor<std::tuple<Ts...>> {
    template<typename F>
    static auto visit(const std::type_info &tid, F &&f) -> decltype(auto) {
        return TypeDispatcher<Ts...>::visit(tid, std::forward<F>(f));
    }
};

}// namespace detail

/**
 * @brief Visit a type by its typeid, invoking a generic lambda with the matching type.
 * 
 * Uses linear search. Safe across DLL boundaries.
 * 
 * @tparam Types A std::tuple<T1, T2, ...> of supported types
 * @tparam F A generic lambda that can be instantiated with any type in Types
 * @param tid The type_info to match against
 * @param f The generic lambda to invoke
 * @return The result of invoking f<MatchedType>()
 * 
 * @example
 * ```cpp
 * using SupportedTypes = std::tuple<int, float, double>;
 * 
 * const std::type_info& tid = tensor.element_type();
 * visit_typeid<SupportedTypes>(tid, [&]<typename T>() {
 *     // T is the matched type
 *     auto& data = static_cast<Tensor<T>&>(tensor);
 *     // ... do something with data
 * });
 * ```
 */
template<typename Types, typename F>
auto visit_typeid(const std::type_info &tid, F &&f) -> decltype(auto) {
    return detail::TupleVisitor<Types>::visit(tid, std::forward<F>(f));
}

/**
 * @brief Convenience overload that accepts types directly instead of std::tuple.
 * 
 * @example
 * ```cpp
 * visit_typeid<int, float, double>(tid, [&]<typename T>() {
 *     // ...
 * });
 * ```
 */
template<typename T1, typename T2, typename... Rest, typename F>
auto visit_typeid(const std::type_info &tid, F &&f) -> decltype(auto) {
    return detail::TypeDispatcher<T1, T2, Rest...>::visit(tid, std::forward<F>(f));
}
}// namespace refl
