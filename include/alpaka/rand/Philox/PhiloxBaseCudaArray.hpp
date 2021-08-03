/* Copyright 2021 Jiri Vyskocil
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/rand/Philox/PhiloxBaseArrayLike.hpp>
#include <alpaka/rand/Philox/helpers/cuintArray.hpp>
#include <alpaka/rand/Philox/mulhilo.hpp>

#include <utility>

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) || defined(ALPAKA_ACC_GPU_CUDA_ENABLED)

namespace alpaka
{
    namespace rand
    {
        namespace engine
        {
            namespace traits
            {
                template<typename TScalar>
                struct PhiloxResultContainerTraits
                {
                    using type = void;
                };
                template<>
                struct PhiloxResultContainerTraits<float>
                {
                    using type = float4_array;
                };
                template<>
                struct PhiloxResultContainerTraits<double>
                {
                    using type = double4_array;
                };
                template<typename TScalar>
                using PhiloxResultContainer = typename PhiloxResultContainerTraits<TScalar>::type;
            } // namespace traits

            /** Philox backend using array-like interface to CUDA uintN types for the storage of Key and Counter
             *
             * @tparam TParams Philox algorithm parameters \sa PhiloxParams
             * @tparam TImpl engine type implementation (CRTP)
             */
            template<typename TParams, typename TImpl>
            class PhiloxBaseCudaArray
                : public PhiloxBaseArrayLike<
                      TParams,
                      uint4_array, // Counter
                      uint2_array, // Key
                      TImpl>
            {
                static_assert(TParams::counterSize == 4, "GPU Philox implemented only for counters of width == 4");

            public:
                using Counter = uint4_array; ///< Counter type = array-like interface to CUDA uint4
                using Key = uint2_array; ///< Key type = array-like interface to CUDA uint2
                template<typename TDistributionResultScalar>
                using ResultContainer
                    = typename traits::PhiloxResultContainer<TDistributionResultScalar>; ///< Vector template for distribution
                                                                                ///< results
            };
        } // namespace engine
    } // namespace rand
} // namespace alpaka

#endif
