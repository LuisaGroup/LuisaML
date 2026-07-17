#pragma once
#include "luisa_ml_config.h"
#include <utility>

namespace luisa::compute::dynamic_array {

template<typename T>
struct BufferData {
    Var<ByteBuffer> *byte_buffer;
    size_t byte_offset;
    size_t byte_size;
    size_t size;

    explicit BufferData(size_t n, Var<ByteBuffer> *buffer,
                        size_t offset = 0, size_t bs = ~0ull) noexcept
        : byte_buffer{buffer}, byte_offset{offset}, byte_size{bs}, size{n} {}

    BufferData(BufferData const &) noexcept = default;
    BufferData(BufferData &&) noexcept = default;
    BufferData &operator=(BufferData const &) = default;
    BufferData &operator=(BufferData &&) = default;

    template<typename U>
        requires is_integral_expr_v<U>
    [[nodiscard]] Var<T> &access(U &&index) const noexcept {
        auto f = detail::FunctionBuilder::current();
        auto i = def(std::forward<U>(index));
        auto addr = def(static_cast<uint>(byte_offset)) +
                    i.template as<uint>() * static_cast<uint>(sizeof(T));
        return *f->create_temporary<Var<T>>(
            byte_buffer->read<T>(addr).expression());
    }
};

}// namespace luisa::compute::dynamic_array
