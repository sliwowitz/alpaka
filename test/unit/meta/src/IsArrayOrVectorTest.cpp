/* Copyright 2021 Jiri Vyskocil
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/meta/IsArrayOrVector.hpp>

#include <catch2/catch.hpp>

#include <array>
#include <vector>


float arrayFloat[4] = {1.0f, 2.0f, 3.0f, 4.0f};

float notAnArrayFloat = 15.0f;
float* notAnArrayFloatPointer = &notAnArrayFloat;
std::string notAnArrayString{"alpaka"};

TEST_CASE("isArrayOrVector", "[meta]")
{
    constexpr bool isArrayOrVectorStdArray = alpaka::meta::IsArrayOrVector<std::array<int, 10>>::value;
    static_assert(isArrayOrVectorStdArray, "alpaka::meta::IsArrayOrVector failed!");
    constexpr bool isArrayOrVectorStdVector = alpaka::meta::IsArrayOrVector<std::vector<float>>::value;
    static_assert(isArrayOrVectorStdVector, "alpaka::meta::IsArrayOrVector failed!");
    constexpr bool isArrayOrVectorCArray = alpaka::meta::IsArrayOrVector<decltype(arrayFloat)>::value;
    static_assert(isArrayOrVectorCArray, "alpaka::meta::IsArrayOrVector failed!");
}

TEST_CASE("isActuallyNotArrayOrVector", "[meta]")
{
    constexpr bool isArrayOrVectorFloat = alpaka::meta::IsArrayOrVector<decltype(notAnArrayFloat)>::value;
    static_assert(!isArrayOrVectorFloat, "alpaka::meta::IsArrayOrVector failed!");
    constexpr bool isArrayOrVectorFloatPointer
        = alpaka::meta::IsArrayOrVector<decltype(notAnArrayFloatPointer)>::value;
    static_assert(!isArrayOrVectorFloatPointer, "alpaka::meta::IsArrayOrVector failed!");
    constexpr bool isArrayOrVectorString = alpaka::meta::IsArrayOrVector<decltype(notAnArrayString)>::value;
    static_assert(!isArrayOrVectorString, "alpaka::meta::IsArrayOrVector failed!");
}

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
TEST_CASE("isArrayOrVectorCudaWrappers", "[meta]")
{
    constexpr bool isArrayOrVectorDouble4arr = alpaka::meta::IsArrayOrVector<CudaVectorArrayWrapper<double, 1>>::value;
    static_assert(isArrayOrVectorDouble1arr, "alpaka::meta::IsArrayOrVector failed!");
    constexpr bool isArrayOrVectorUint2arr = alpaka::meta::IsArrayOrVector<CudaVectorArrayWrapper<unsigned, 2>>::value;
    static_assert(isArrayOrVectorUint2arr, "alpaka::meta::IsArrayOrVector failed!");
    constexpr bool isArrayOrVectorInt3arr = alpaka::meta::IsArrayOrVector<CudaVectorArrayWrapper<int, 3>>::value;
    static_assert(isArrayOrVectorInt3arr, "alpaka::meta::IsArrayOrVector failed!");
    constexpr bool isArrayOrVectorFloat4arr = alpaka::meta::IsArrayOrVector<CudaVectorArrayWrapper<float, 4>>::value;
    static_assert(isArrayOrVectorFloat4arr, "alpaka::meta::IsArrayOrVector failed!");

    constexpr bool isArrayOrVectorUint4 = alpaka::meta::IsArrayOrVector<uint4>::value;
    static_assert(!isArrayOrVectorUint4, "alpaka::meta::IsArrayOrVector failed!");
}
#endif
