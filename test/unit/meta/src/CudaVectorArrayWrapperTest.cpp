/* Copyright 2021 Jiri Vyskocil
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/meta/CudaVectorArrayWrapper.hpp>

#    include <catch2/catch.hpp>

TEST_CASE("cudaVectorArrayWrapperSetAndRead", "[meta]")
{
    CudaVectorArrayWrapper<float, 1> f1{-1.0f};
    REQUIRE(f4[0] == -1.0f);

    CudaVectorArrayWrapper<unsigned, 2> arr2 = {0u, 1u};
    REQUIRE(arr2[0] == 0u);
    REQUIRE(arr2[1] == 1u);

    CudaVectorArrayWrapper<unsigned, 4> arr4{0u, 0u, 0u, 0u};
    arr4[1] = 1u;
    arr4[2] = arr4[1] + 1u;
    arr4[3] = arr4[2] + arr2[1];
    REQUIRE(arr4[0] == 0u);
    REQUIRE(arr4[1] == 1u);
    REQUIRE(arr4[2] == 2u);
    REQUIRE(arr4[3] == 3u);

    CudaVectorArrayWrapper<double, 3> d4{0.0, 0.0, 0.0};
    d4 = {0.0, -1.0, -2.0};
    REQUIRE(d4[0] == 0.0);
    REQUIRE(d4[1] == -1.0);
    REQUIRE(d4[2] == -2.0);
}

#endif
