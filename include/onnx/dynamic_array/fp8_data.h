#pragma once
#include "luisa_ml_config.h"
#include <utility>
#include <type_traits>
#include "onnx/fp_quantized.h"

namespace luisa::compute::dynamic_array {

template<typename T>
struct FP8Data {
    Var<ByteBuffer> *byte_buffer;
    size_t byte_offset;
    size_t byte_size;
    size_t size;

    explicit FP8Data(size_t n, Var<ByteBuffer> *buffer,
                     size_t offset = 0, size_t bs = ~0ull) noexcept
        : byte_buffer{buffer}, byte_offset{offset}, byte_size{bs}, size{n} {}

    FP8Data(FP8Data const &) noexcept = default;
    FP8Data(FP8Data &&) noexcept = default;
    FP8Data &operator=(FP8Data const &) = default;
    FP8Data &operator=(FP8Data &&) = default;

    template<typename U>
        requires is_integral_expr_v<U>
    [[nodiscard]] Var<T> &access(U &&index) const noexcept {
        auto f = detail::FunctionBuilder::current();
        auto i = def(std::forward<U>(index));
        if constexpr (std::is_same_v<T, FP8E4M3FN> || std::is_same_v<T, FP8E5M2>) {
            auto word_idx = i / 4u;
            auto byte_idx = i % 4u;
            auto addr = def(static_cast<uint>(byte_offset)) + word_idx * 4u;
            auto word = byte_buffer->read<uint>(addr);
            auto shift = byte_idx * 8u;
            auto byte_val = (word >> shift) & 0xffu;
            auto tmp = Var<T>{};
            tmp.bits = byte_val.cast<uint16_t>();
            return *f->create_temporary<Var<T>>(tmp.expression());
        } else {
            return *f->create_temporary<Var<T>>(f->local(Type::of<T>()));
        }
    }
};

}// namespace luisa::compute::dynamic_array
