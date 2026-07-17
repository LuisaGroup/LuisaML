#pragma once
#include "luisa_ml_config.h"
#include <utility>
#include <type_traits>
#include "onnx/fp_quantized.h"

namespace luisa::compute::dynamic_array {

template<typename T>
struct FP4Data {
    Var<ByteBuffer> *byte_buffer;
    size_t byte_offset;
    size_t byte_size;
    size_t size;

    explicit FP4Data(size_t n, Var<ByteBuffer> *buffer,
                     size_t offset = 0, size_t bs = ~0ull) noexcept
        : byte_buffer{buffer}, byte_offset{offset}, byte_size{bs}, size{n} {}

    FP4Data(FP4Data const &) noexcept = default;
    FP4Data(FP4Data &&) noexcept = default;
    FP4Data &operator=(FP4Data const &) = default;
    FP4Data &operator=(FP4Data &&) = default;

    template<typename U>
        requires is_integral_expr_v<U>
    [[nodiscard]] Var<T> &access(U &&index) const noexcept {
        auto f = detail::FunctionBuilder::current();
        auto i = def(std::forward<U>(index));
        if constexpr (std::is_same_v<T, FP4E2M1>) {
            auto word_idx = i / 8u;
            auto nibble_idx = i % 8u;
            auto addr = def(static_cast<uint>(byte_offset)) + word_idx * 4u;
            auto word = byte_buffer->read<uint>(addr);
            auto shift = nibble_idx * 4u;
            auto nibble = (word >> shift) & 0x0fu;
            auto tmp = Var<T>{};
            tmp.bits = nibble.cast<uint16_t>();
            return *f->create_temporary<Var<T>>(tmp.expression());
        } else {
            return *f->create_temporary<Var<T>>(f->local(Type::of<T>()));
        }
    }
};

}// namespace luisa::compute::dynamic_array
