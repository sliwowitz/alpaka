/* Copyright 2022 Jiri Vyskocil, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>

namespace alpaka::rand
{
    /// Get high 32 bits of a 64-bit number
    ALPAKA_FN_HOST_ACC constexpr static auto high32Bits(std::uint64_t const x) -> std::uint32_t
    {
        return static_cast<std::uint32_t>(x >> 32);
    }

    /// Get low 32 bits of a 64-bit number
    ALPAKA_FN_HOST_ACC constexpr static auto low32Bits(std::uint64_t const x) -> std::uint32_t
    {
        return static_cast<std::uint32_t>(x & 0xffffffff);
    }

    /** Multiply two 64-bit numbers and split the result into high and low 32 bits, also known as "mulhilo32"
     *
     * @param a first 64-bit multiplier
     * @param b second 64-bit multiplier
     * @return a pair of 32 numbers { resultHigh, resultLow } with the high and low 32 bits of the product a*b
     */
    // TODO: See single-instruction implementations in original Philox source code
    ALPAKA_FN_HOST_ACC constexpr static auto multiplyAndSplit64to32(std::uint64_t const a, std::uint64_t const b)
        -> std::pair<std::uint32_t, std::uint32_t>
    {
        std::uint64_t res64 = a * b;
        return std::pair<std::uint32_t, std::uint32_t>{high32Bits(res64), low32Bits(res64)};
    }
} // namespace alpaka::rand
