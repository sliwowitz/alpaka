/* Copyright 2022 Jeffrey Kelling
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/rand/Philox/PhiloxStateless.hpp"

#include <utility>

namespace alpaka::rand::engine
{
    /** Philox-stateless engine generating a vector of numbers
     *
     * This engine's operator() will return a vector of numbers corresponding to the full size of its counter.
     * This is a convenience vs. memory size tradeoff since the user has to deal with the output array
     * themselves, but the internal state comprises only of a single counter and a key.
     *
     * @tparam TParams Basic parameters for the Philox algorithm
     */
    template<typename TParams>
    class PhiloxStatelessVector : public PhiloxStateless<TParams>
    {
    };
} // namespace alpaka::rand::engine
