/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/core/Config.hpp>

#include <catch2/catch_test_macros.hpp>

#include <iostream>

TEST_CASE("printDefines", "[core]")
{
#if ALPAKA_LANG_CUDA
    std::cout << "ALPAKA_LANG_CUDA:" << ALPAKA_LANG_CUDA << std::endl;
#endif
#if ALPAKA_LANG_HIP
    std::cout << "ALPAKA_LANG_HIP:" << ALPAKA_LANG_HIP << std::endl;
#endif
#if ALPAKA_ARCH_PTX
    std::cout << "ALPAKA_ARCH_PTX:" << ALPAKA_ARCH_PTX << std::endl;
#endif
#if ALPAKA_COMP_NVCC
    std::cout << "ALPAKA_COMP_NVCC:" << ALPAKA_COMP_NVCC << std::endl;
#endif
#if ALPAKA_COMP_HIP
    std::cout << "ALPAKA_COMP_HIP:" << ALPAKA_COMP_HIP << std::endl;
#endif
#if ALPAKA_COMP_CLANG
    std::cout << "ALPAKA_COMP_CLANG:" << ALPAKA_COMP_CLANG << std::endl;
#endif
#if ALPAKA_COMP_GNUC
    std::cout << "ALPAKA_COMP_GNUC:" << ALPAKA_COMP_GNUC << std::endl;
#endif
#if ALPAKA_COMP_CLANG_CUDA
    std::cout << "ALPAKA_COMP_CLANG_CUDA:" << ALPAKA_COMP_CLANG_CUDA << std::endl;
#endif
}

TEST_CASE("configVersionMacros", "")
{
    static_assert(ALPAKA_VERSION_NUMBER(2025, 2, 1) == 202'500'200'001);

    static_assert(ALPAKA_VERSION_NUMBER_NOT_AVAILABLE == 0000000000000);
    static_assert(ALPAKA_VERSION_NUMBER_NOT_AVAILABLE == ALPAKA_VERSION_NUMBER(0, 0, 0));

    static_assert(ALPAKA_YYYYMMDD_TO_VERSION(20'250'201) == 202'500'200'001);
    static_assert(ALPAKA_YYYYMMDD_TO_VERSION(20'250'201) == ALPAKA_VERSION_NUMBER(2025, 2, 1));

    static_assert(ALPAKA_YYYYMM_TO_VERSION(202502) == 202'500'200'000);
    static_assert(ALPAKA_YYYYMM_TO_VERSION(202502) == ALPAKA_VERSION_NUMBER(2025, 2, 0));

    static_assert(ALPAKA_VVRRP_10_TO_VERSION(12081) == 1'200'800'001);
    static_assert(ALPAKA_VVRRP_10_TO_VERSION(12081) == ALPAKA_VERSION_NUMBER(12, 8, 1));

    static_assert(ALPAKA_VRP_TO_VERSION(751) == 700'500'001);
    static_assert(ALPAKA_VRP_TO_VERSION(751) == ALPAKA_VERSION_NUMBER(7, 5, 1));
}
