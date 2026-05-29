#pragma once

#include <cstdint>
#include <cmath>
#include <limits>

#include <luisa/core/basic_traits.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>

using luisa::half;

// ==========================================================================
// FP4 E2M1 format: 1 sign bit, 2 exponent bits (bias=1), 1 mantissa bit
// Layout: S EE M
// No Inf/NaN encoding; all 16 bit patterns map to finite values.
// ==========================================================================
struct FP4E2M1 {

    static constexpr int kExpBits = 2;
    static constexpr int kMantBits = 1;
    static constexpr int kBias = 1;
    static constexpr int kMaxExp = (1 << kExpBits) - 1;        // 3
    static constexpr uint8_t kMantMask = (1 << kMantBits) - 1;// 0x01
    static constexpr uint8_t kSignMask = 0x08;                 // bit 3 for 4-bit value
    static constexpr float kMaxFinite = 6.0f;

    uint16_t bits;

    // Convert FP32 to FP4 E2M1 with round-to-nearest-even
    static uint8_t from_float(float v) {
        if (v == 0.0f) {
            return 0;
        }
        uint32_t sign = 0;
        if (v < 0.0f) {
            sign = 1;
            v = -v;
        }

        if (v > kMaxFinite) {
            // Clamp to max finite: S 11 1
            return static_cast<uint8_t>((sign ? kSignMask : 0) | (kMaxExp << kMantBits) | kMantMask);
        }

        // Decompose float
        int exp;
        float mant = std::frexp(v, &exp);// v = mant * 2^exp, mant in [0.5, 1.0)
        // Adjust to get mant in [1.0, 2.0)
        mant *= 2.0f;
        exp -= 1;

        int e = exp + kBias;
        if (e <= 0) {
            // Denormal or underflow to zero
            if (e < -kMantBits) {
                return static_cast<uint8_t>(sign ? kSignMask : 0);
            }
            // Denormal: shift mantissa right
            int shift = 1 - e;
            mant = mant / static_cast<float>(1 << shift);
            e = 0;
        }

        // Round mantissa to 1 bit
        float mant_scaled = mant * (1 << kMantBits);
        float mant_q = std::floor(mant_scaled);
        float frac = mant_scaled - mant_q;
        uint8_t m = static_cast<uint8_t>(mant_q) & kMantMask;

        // Round-to-nearest-even
        if (frac > 0.5f || (frac == 0.5f && (m & 1))) {
            m += 1;
            if (m > kMantMask) {
                m = 0;
                e += 1;
            }
        }

        // Clamp overflow to max finite
        if (e > kMaxExp) {
            e = kMaxExp;
            m = kMantMask;
        }

        if (e == 0 && m == 0) {
            return static_cast<uint8_t>(sign ? kSignMask : 0);
        }

        return static_cast<uint8_t>((sign ? kSignMask : 0) | (e << kMantBits) | m);
    }

    // Convert FP4 E2M1 to FP32
    static float to_float(uint8_t bits) {
        uint8_t sign = (bits & kSignMask) >> 3;
        uint8_t e = (bits >> kMantBits) & ((1 << kExpBits) - 1);
        uint8_t m = bits & kMantMask;

        if (e == 0 && m == 0) {
            return sign ? -0.0f : 0.0f;
        }

        float value;
        if (e == 0) {
            // Denormal: (m/2) * 2^(1-bias)
            value = static_cast<float>(m) / static_cast<float>(1 << kMantBits) * std::pow(2.0f, 1 - kBias);
        } else {
            // Normal: (1 + m/2) * 2^(e-bias)
            value = (1.0f + static_cast<float>(m) / static_cast<float>(1 << kMantBits)) * std::pow(2.0f, e - kBias);
        }
        return sign ? -value : value;
    }
};

LUISA_STRUCT(FP4E2M1, bits) {};

// CPU helpers for packing two 4-bit nibbles into one byte
inline uint8_t pack_fp4(uint8_t upper, uint8_t lower) noexcept {
    return (upper << 4) | (lower & 0x0f);
}

inline uint8_t unpack_fp4_upper(uint8_t packed) noexcept {
    return packed >> 4;
}

inline uint8_t unpack_fp4_lower(uint8_t packed) noexcept {
    return packed & 0x0f;
}

// ==========================================================================
// FP8 E4M3 format: 1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits
// Layout: S EEEE MMM
// ==========================================================================
struct FP8E4M3FN {

    static constexpr int kExpBits = 4;
    static constexpr int kMantBits = 3;
    static constexpr int kBias = 7;
    static constexpr int kMaxExp = (1 << kExpBits) - 1;       // 15
    static constexpr int kInfNanExp = kMaxExp;                // 15
    static constexpr uint8_t kMantMask = (1 << kMantBits) - 1;// 0x07
    static constexpr uint8_t kSignMask = 0x80;

    uint16_t bits;

    // Convert FP32 to FP8 E4M3 with round-to-nearest-even
    static uint8_t from_float(float v) {
        uint8_t bits{};
        if (std::isnan(v)) {
            bits = 0x7f;// canonical NaN
            return bits;
        }
        if (v == 0.0f) {
            bits = 0;
            return bits;
        }
        uint32_t sign = 0;
        if (v < 0.0f) {
            sign = 1;
            v = -v;
        }

        // E4M3 max finite value: E=15, M=6 -> (1 + 6/8) * 2^(15-7) = 1.75 * 256 = 448
        constexpr float max_finite = 448.0f;
        if (v > max_finite) {
            // Clamp to max finite
            bits = sign ? 0xfe : 0x7e;// E=15, M=6
            return bits;
        }

        // Decompose float
        int exp;
        float mant = std::frexp(v, &exp);// v = mant * 2^exp, mant in [0.5, 1.0)
        // Adjust to get mant in [1.0, 2.0)
        mant *= 2.0f;
        exp -= 1;

        int e = exp + kBias;
        if (e <= 0) {
            // Denormal or underflow to zero
            if (e < -kMantBits) {
                bits = sign ? kSignMask : 0;
                return bits;
            }
            // Denormal: shift mantissa right
            int shift = 1 - e;
            mant = mant / static_cast<float>(1 << shift);
            e = 0;
        }

        // Round mantissa to 3 bits
        float mant_q = std::floor(mant * (1 << kMantBits));
        float frac = mant * (1 << kMantBits) - mant_q;
        uint8_t m = static_cast<uint8_t>(mant_q) & kMantMask;

        // Round-to-nearest-even
        if (frac > 0.5f || (frac == 0.5f && (m & 1))) {
            m += 1;
            if (m > kMantMask) {
                m = 0;
                e += 1;
            }
        }

        // E4M3: E=15, M=7 is NaN, so max finite is E=15, M=6
        if (e >= kInfNanExp) {
            e = kInfNanExp;
            if (m >= kMantMask) {
                m = kMantMask - 1;// clamp to 6
            }
        }

        // Reconstruct
        if (e == 0 && m == 0) {
            bits = sign ? kSignMask : 0;
            return bits;
        }

        bits = (sign ? kSignMask : 0) | static_cast<uint8_t>(e << kMantBits) | m;
        return bits;
    }

    // Convert FP8 E4M3 to FP32
    static float to_float(uint8_t bits) {
        uint8_t sign = (bits & kSignMask) >> 7;
        uint8_t e = (bits >> kMantBits) & ((1 << kExpBits) - 1);
        uint8_t m = bits & kMantMask;

        if (e == 0 && m == 0) {
            return sign ? -0.0f : 0.0f;
        }

        // E4M3 NaN: E=15, M=7
        if (e == kInfNanExp && m == kMantMask) {
            return std::numeric_limits<float>::quiet_NaN();
        }

        float value;
        if (e == 0) {
            // Denormal
            value = static_cast<float>(m) / static_cast<float>(1 << kMantBits) * std::pow(2.0f, 1 - kBias);
        } else {
            // Normal
            value = (1.0f + static_cast<float>(m) / static_cast<float>(1 << kMantBits)) * std::pow(2.0f, e - kBias);
        }
        return sign ? -value : value;
    }
};

LUISA_STRUCT(FP8E4M3FN, bits) {};

// ==========================================================================
// FP8 E5M2 format: 1 sign bit, 5 exponent bits (bias=15), 2 mantissa bits
// Layout: S EEEEE MM
// E5M2 supports Inf/NaN: E=31, M=0 is Inf; E=31, M!=0 is NaN
// ==========================================================================
struct FP8E5M2 {

    static constexpr int kExpBits = 5;
    static constexpr int kMantBits = 2;
    static constexpr int kBias = 15;
    static constexpr int kMaxExp = (1 << kExpBits) - 1;       // 31
    static constexpr int kInfNanExp = kMaxExp;                // 31
    static constexpr uint8_t kMantMask = (1 << kMantBits) - 1;// 0x03
    static constexpr uint8_t kSignMask = 0x80;

    uint16_t bits;

    // Convert FP32 to FP8 E5M2 with round-to-nearest-even
    static uint8_t from_float(float v) {
        uint8_t bits{};
        if (std::isnan(v)) {
            bits = 0x7f;// canonical NaN
            return bits;
        }
        if (std::isinf(v)) {
            bits = (v < 0.0f) ? 0xfc : 0x7c;// E=31, M=0
            return bits;
        }
        if (v == 0.0f) {
            bits = 0;
            return bits;
        }
        uint32_t sign = 0;
        if (v < 0.0f) {
            sign = 1;
            v = -v;
        }

        // E5M2 max finite value: E=30, M=3 -> (1 + 3/4) * 2^(30-15) = 1.75 * 32768 = 57344
        constexpr float max_finite = 57344.0f;
        if (v > max_finite) {
            bits = sign ? 0xfc : 0x7c;// E=31, M=0 -> Inf
            return bits;
        }

        // Decompose float
        int exp;
        float mant = std::frexp(v, &exp);// v = mant * 2^exp, mant in [0.5, 1.0)
        // Adjust to get mant in [1.0, 2.0)
        mant *= 2.0f;
        exp -= 1;

        int e = exp + kBias;
        if (e <= 0) {
            // Denormal or underflow to zero
            if (e < -kMantBits) {
                bits = sign ? kSignMask : 0;
                return bits;
            }
            // Denormal: shift mantissa right
            int shift = 1 - e;
            mant = mant / static_cast<float>(1 << shift);
            e = 0;
        }

        // Round mantissa to 2 bits
        float mant_q = std::floor(mant * (1 << kMantBits));
        float frac = mant * (1 << kMantBits) - mant_q;
        uint8_t m = static_cast<uint8_t>(mant_q) & kMantMask;

        // Round-to-nearest-even
        if (frac > 0.5f || (frac == 0.5f && (m & 1))) {
            m += 1;
            if (m > kMantMask) {
                m = 0;
                e += 1;
            }
        }

        // E5M2: E=31, M=0 is Inf, any M!=0 is NaN
        if (e >= kInfNanExp) {
            e = kInfNanExp;
            m = 0;// Inf
        }

        // Reconstruct
        if (e == 0 && m == 0) {
            bits = sign ? kSignMask : 0;
            return bits;
        }

        bits = (sign ? kSignMask : 0) | static_cast<uint8_t>(e << kMantBits) | m;
        return bits;
    }

    // Convert FP8 E5M2 to FP32
    static float to_float(uint8_t bits) {
        uint8_t sign = (bits & kSignMask) >> 7;
        uint8_t e = (bits >> kMantBits) & ((1 << kExpBits) - 1);
        uint8_t m = bits & kMantMask;

        if (e == 0 && m == 0) {
            return sign ? -0.0f : 0.0f;
        }

        // E5M2 Inf: E=31, M=0
        if (e == kInfNanExp && m == 0) {
            return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        }
        // E5M2 NaN: E=31, M!=0
        if (e == kInfNanExp && m != 0) {
            return std::numeric_limits<float>::quiet_NaN();
        }

        float value;
        if (e == 0) {
            // Denormal
            value = static_cast<float>(m) / static_cast<float>(1 << kMantBits) * std::pow(2.0f, 1 - kBias);
        } else {
            // Normal
            value = (1.0f + static_cast<float>(m) / static_cast<float>(1 << kMantBits)) * std::pow(2.0f, e - kBias);
        }
        return sign ? -value : value;
    }
};

LUISA_STRUCT(FP8E5M2, bits) {};

// ==========================================================================
// FP16 Quantized: half-precision storage with float scale/offset
// Compute in float, store in half.
// ==========================================================================
struct FP16Quantized {

    half bits;

    // Quantize FP32 to half: compute (v - offset) / scale in float, store in half
    static half from_float(float v, float scale, float offset) {
        float q = (v - offset) / scale;
        return half(q);
    }

    // Dequantize half to FP32: load half, compute q * scale + offset in float
    static float to_float(half bits, float scale, float offset) {
        float q = static_cast<float>(bits);
        return q * scale + offset;
    }
};

LUISA_STRUCT(FP16Quantized, bits) {};

namespace luisa::compute {

// Helper to create UShort (Var<ushort>) from DSL or scalar values
template<typename T>
[[nodiscard]] inline auto _us(T &&x) noexcept {
    if constexpr (is_dsl_v<std::remove_cvref_t<T>>) {
        return cast<ushort>(std::forward<T>(x));
    } else {
        return UShort(std::forward<T>(x));
    }
}

// ==========================================================================
// DSL Callables for FP4
// ==========================================================================
inline Callable<ushort(half)> fp4e2m1_from_float() {
    static Callable _c{[](Half v) noexcept {
        $if (v == half(0.0f)) {
            $return(_us(0u));
        };

        UShort sign = _us(0u);
        $if (v < half(0.0f)) {
            sign = _us(1u);
            v = -v;
        };

        $if (v > half(FP4E2M1::kMaxFinite)) {
            $return(ite(sign == _us(1u), _us(0x0fu), _us(0x07u)));
        };

        // Decompose float: v = mant * 2^exp, mant in [1.0, 2.0)
        Int exp = floor(log2(v)).cast<int>();
        Half mant = v / pow(half(2.0f), exp.cast<half>());
        $if (mant >= half(2.0f)) {
            mant = mant * half(0.5f);
            exp = exp + 1;
        };
        $if (mant < half(1.0f)) {
            mant = mant * half(2.0f);
            exp = exp - 1;
        };

        Int e = exp + FP4E2M1::kBias;
        $if (e <= 0) {
            $if (e < -FP4E2M1::kMantBits) {
                $return(ite(sign == _us(1u), _us(0x08u), _us(0u)));
            };
            Int shift = 1 - e;
            mant = mant / pow(half(2.0f), shift.cast<half>());
            e = 0;
        };

        // Round mantissa to 1 bit
        Half mant_scaled = mant * half(2.0f);
        Half mant_q = floor(mant_scaled);
        Half frac = mant_scaled - mant_q;
        UShort m = _us(mant_q.cast<ushort>() & static_cast<ushort>(FP4E2M1::kMantMask));

        // Round-to-nearest-even
        $if (frac > half(0.5f) | (frac == half(0.5f) & _us(m & _us(1u)) != _us(0u))) {
            m = _us(m + _us(1u));
            $if (m > static_cast<ushort>(FP4E2M1::kMantMask)) {
                m = _us(0u);
                e = e + 1;
            };
        };

        // Clamp overflow to max finite
        $if (e > FP4E2M1::kMaxExp) {
            e = FP4E2M1::kMaxExp;
            m = _us(static_cast<ushort>(FP4E2M1::kMantMask));
        };

        $if (e == 0 & m == _us(0u)) {
            $return(ite(sign == _us(1u), _us(0x08u), _us(0u)));
        };

        auto t1 = _us(sign << 3u);
        auto t2 = _us(e.cast<ushort>() << FP4E2M1::kMantBits);
        UShort bits = _us(_us(t1 | t2) | m);
        return bits;
    }};
    return _c;
}

inline Callable<half(ushort)> fp4e2m1_to_float() {
    static Callable _c{[](UShort bits) noexcept {
        UShort sign = _us((bits >> 3u) & 1u);
        UShort e = _us(_us(bits >> FP4E2M1::kMantBits) & static_cast<ushort>((1 << FP4E2M1::kExpBits) - 1));
        UShort m = _us(bits & static_cast<ushort>(FP4E2M1::kMantMask));

        $if (e == _us(0u) & m == _us(0u)) {
            $return(ite(sign != _us(0u), -half(0.0f), half(0.0f)));
        };

        Half value;
        $if (e == _us(0u)) {
            // Denormal: (m/2) * 2^(1-bias)
            value = (m.cast<half>() / half(2.0f)) * pow(half(2.0f), half(1.0f) - half(FP4E2M1::kBias));
        }
        $else {
            // Normal: (1 + m/2) * 2^(e-bias)
            value = (half(1.0f) + m.cast<half>() / half(2.0f)) * pow(half(2.0f), e.cast<half>() - half(FP4E2M1::kBias));
        };
        return ite(sign != _us(0u), -value, value);
    }};
    return _c;
}

// Pack two 4-bit nibbles (in lower 4 bits of each ushort) into one byte
inline Callable<ushort(ushort, ushort)> pack_fp4() {
    static Callable _c{[](UShort upper, UShort lower) noexcept {
        return _us(_us(upper << 4u) | _us(lower & _us(0x0fu)));
    }};
    return _c;
}

// Unpack nibble from packed byte at index: idx=0 -> upper, idx=1 -> lower
inline Callable<ushort(ushort, ushort)> unpack_fp4() {
    static Callable _c{[](UShort packed, UShort idx) noexcept {
        return ite(_us(idx & _us(1u)) == _us(0u), _us(packed >> 4u), _us(packed & _us(0x0fu)));
    }};
    return _c;
}

// ==========================================================================
// DSL Callables for FP8 E4M3
// ==========================================================================
inline Callable<ushort(half)> fp8e4m3_from_float() {
    static Callable _c{[](Half v) noexcept {
        $if (luisa::compute::dsl::isnan(v)) {
            $return(_us(0x7fu));
        };
        $if (v == half(0.0f)) {
            $return(_us(0u));
        };

        UShort sign = _us(0u);
        $if (v < half(0.0f)) {
            sign = _us(1u);
            v = -v;
        };

        $if (v > half(448.0f)) {
            $return(ite(sign == _us(1u), _us(0xfeu), _us(0x7eu)));
        };

        // Decompose float: v = mant * 2^exp, mant in [1.0, 2.0)
        Int exp = floor(log2(v)).cast<int>();
        Half mant = v / pow(half(2.0f), exp.cast<half>());
        $if (mant >= half(2.0f)) {
            mant = mant * half(0.5f);
            exp = exp + 1;
        };
        $if (mant < half(1.0f)) {
            mant = mant * half(2.0f);
            exp = exp - 1;
        };

        Int e = exp + FP8E4M3FN::kBias;
        $if (e <= 0) {
            $if (e < -FP8E4M3FN::kMantBits) {
                $return(ite(sign == _us(1u), _us(0x80u), _us(0u)));
            };
            Int shift = 1 - e;
            mant = mant / pow(half(2.0f), shift.cast<half>());
            e = 0;
        };

        // Round mantissa to 3 bits
        Half mant_scaled = mant * half(8.0f);
        Half mant_q = floor(mant_scaled);
        Half frac = mant_scaled - mant_q;
        UShort m = _us(mant_q.cast<ushort>() & static_cast<ushort>(FP8E4M3FN::kMantMask));

        // Round-to-nearest-even
        $if (frac > half(0.5f) | (frac == half(0.5f) & _us(m & _us(1u)) != _us(0u))) {
            m = _us(m + _us(1u));
            $if (m > static_cast<ushort>(FP8E4M3FN::kMantMask)) {
                m = _us(0u);
                e = e + 1;
            };
        };

        // E4M3: E=15, M=7 is NaN, so max finite is E=15, M=6
        $if (e >= FP8E4M3FN::kInfNanExp) {
            e = FP8E4M3FN::kInfNanExp;
            $if (m >= static_cast<ushort>(FP8E4M3FN::kMantMask)) {
                m = _us(static_cast<ushort>(FP8E4M3FN::kMantMask - 1));
            };
        };

        $if (e == 0 & m == _us(0u)) {
            $return(ite(sign == _us(1u), _us(0x80u), _us(0u)));
        };

        auto t1 = _us(sign << 7u);
        auto t2 = _us(e.cast<ushort>() << FP8E4M3FN::kMantBits);
        UShort bits = _us(_us(t1 | t2) | m);
        return bits;
    }};
    return _c;
}

inline Callable<half(ushort)> fp8e4m3_to_float() {
    static Callable _c{[](UShort bits) noexcept {
        UShort sign = _us((bits >> 7u) & 1u);
        UShort e = _us(_us(bits >> FP8E4M3FN::kMantBits) & static_cast<ushort>((1 << FP8E4M3FN::kExpBits) - 1));
        UShort m = _us(bits & static_cast<ushort>(FP8E4M3FN::kMantMask));

        $if (e == _us(0u) & m == _us(0u)) {
            $return(ite(sign != _us(0u), -half(0.0f), half(0.0f)));
        };

        // E4M3 NaN: E=15, M=7
        $if (e == static_cast<ushort>(FP8E4M3FN::kInfNanExp) & m == static_cast<ushort>(FP8E4M3FN::kMantMask)) {
            UInt nan_bits = 0x7fc00000u;
            $return(cast<half>(as<float>(nan_bits)));
        };

        Half value;
        $if (e == _us(0u)) {
            // Denormal
            value = (m.cast<half>() / half(8.0f)) * pow(half(2.0f), half(1.0f) - half(FP8E4M3FN::kBias));
        }
        $else {
            // Normal
            value = (half(1.0f) + m.cast<half>() / half(8.0f)) * pow(half(2.0f), e.cast<half>() - half(FP8E4M3FN::kBias));
        };
        return ite(sign != _us(0u), -value, value);
    }};
    return _c;
}

// ==========================================================================
// DSL Callables for FP8 E5M2
// ==========================================================================
inline Callable<ushort(half)> fp8e5m2_from_float() {
    static Callable _c{[](Half v) noexcept {
        $if (luisa::compute::dsl::isnan(v)) {
            $return(_us(0x7fu));
        };
        $if (luisa::compute::dsl::isinf(v)) {
            $return(ite(v < half(0.0f), _us(0xfcu), _us(0x7cu)));
        };
        $if (v == half(0.0f)) {
            $return(_us(0u));
        };

        UShort sign = _us(0u);
        $if (v < half(0.0f)) {
            sign = _us(1u);
            v = -v;
        };

        $if (v > half(57344.0f)) {
            $return(ite(sign == _us(1u), _us(0xfcu), _us(0x7cu)));
        };

        // Decompose float: v = mant * 2^exp, mant in [1.0, 2.0)
        Int exp = floor(log2(v)).cast<int>();
        Half mant = v / pow(half(2.0f), exp.cast<half>());
        $if (mant >= half(2.0f)) {
            mant = mant * half(0.5f);
            exp = exp + 1;
        };
        $if (mant < half(1.0f)) {
            mant = mant * half(2.0f);
            exp = exp - 1;
        };

        Int e = exp + FP8E5M2::kBias;
        $if (e <= 0) {
            $if (e < -FP8E5M2::kMantBits) {
                $return(ite(sign == _us(1u), _us(0x80u), _us(0u)));
            };
            Int shift = 1 - e;
            mant = mant / pow(half(2.0f), shift.cast<half>());
            e = 0;
        };

        // Round mantissa to 2 bits
        Half mant_scaled = mant * half(4.0f);
        Half mant_q = floor(mant_scaled);
        Half frac = mant_scaled - mant_q;
        UShort m = _us(mant_q.cast<ushort>() & static_cast<ushort>(FP8E5M2::kMantMask));

        // Round-to-nearest-even
        $if (frac > half(0.5f) | (frac == half(0.5f) & _us(m & _us(1u)) != _us(0u))) {
            m = _us(m + _us(1u));
            $if (m > static_cast<ushort>(FP8E5M2::kMantMask)) {
                m = _us(0u);
                e = e + 1;
            };
        };

        // E5M2: E=31, M=0 is Inf, any M!=0 is NaN
        $if (e >= FP8E5M2::kInfNanExp) {
            e = FP8E5M2::kInfNanExp;
            m = _us(0u);
        };

        $if (e == 0 & m == _us(0u)) {
            $return(ite(sign == _us(1u), _us(0x80u), _us(0u)));
        };

        auto t1 = _us(sign << 7u);
        auto t2 = _us(e.cast<ushort>() << FP8E5M2::kMantBits);
        UShort bits = _us(_us(t1 | t2) | m);
        return bits;
    }};
    return _c;
}

inline Callable<half(ushort)> fp8e5m2_to_float() {
    static Callable _c{[](UShort bits) noexcept {
        UShort sign = _us((bits >> 7u) & 1u);
        UShort e = _us(_us(bits >> FP8E5M2::kMantBits) & static_cast<ushort>((1 << FP8E5M2::kExpBits) - 1));
        UShort m = _us(bits & static_cast<ushort>(FP8E5M2::kMantMask));

        $if (e == _us(0u) & m == _us(0u)) {
            $return(ite(sign != _us(0u), -half(0.0f), half(0.0f)));
        };

        // E5M2 Inf: E=31, M=0
        $if (e == static_cast<ushort>(FP8E5M2::kInfNanExp) & m == _us(0u)) {
            UInt inf_bits = ite(sign != _us(0u), 0xff800000u, 0x7f800000u);
            $return(cast<half>(as<float>(inf_bits)));
        };

        // E5M2 NaN: E=31, M!=0
        $if (e == static_cast<ushort>(FP8E5M2::kInfNanExp) & m != _us(0u)) {
            UInt nan_bits = 0x7fc00000u;
            $return(cast<half>(as<float>(nan_bits)));
        };

        Half value;
        $if (e == _us(0u)) {
            // Denormal
            value = (m.cast<half>() / half(4.0f)) * pow(half(2.0f), half(1.0f) - half(FP8E5M2::kBias));
        }
        $else {
            // Normal
            value = (half(1.0f) + m.cast<half>() / half(4.0f)) * pow(half(2.0f), e.cast<half>() - half(FP8E5M2::kBias));
        };
        return ite(sign != _us(0u), -value, value);
    }};
    return _c;
}

// ==========================================================================
// DSL Callables for FP16 Quantized
// ==========================================================================
inline Callable<half(float, float, float)> fp16_quantized_from_float() {
    static Callable _c{[](Float v, Float scale, Float offset) noexcept {
        Float q = (v - offset) / scale;
        return cast<half>(q);
    }};
    return _c;
}

inline Callable<float(half, float, float)> fp16_quantized_to_float() {
    static Callable _c{[](Half bits, Float scale, Float offset) noexcept {
        Float q = cast<float>(bits);
        return q * scale + offset;
    }};
    return _c;
}

}// namespace luisa::compute
