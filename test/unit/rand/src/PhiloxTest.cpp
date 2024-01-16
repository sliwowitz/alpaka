/* Copyright 2024 Jiri Vyskocil
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/rand/Traits.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

class PhiloxTest
{
protected:
    alpaka::rand::Philox4x32x10 statelessEngine;
    alpaka::rand::Philox4x32x10Vector statefulVectorEngine;
    alpaka::rand::Philox4x32x10 statefulSingleEngine;
};

TEST_CASE_METHOD(PhiloxTest, "HostStatelessEngineTest")
{
    auto result = statelessEngine();
    REQUIRE(result >= statelessEngine.min());
    REQUIRE(result <= statelessEngine.max());
}

TEST_CASE_METHOD(PhiloxTest, "HostStatefulVectorEngineTest")
{
    auto resultVec = statefulVectorEngine();
    for(auto& result : resultVec)
    {
        REQUIRE(result >= statefulVectorEngine.min());
        REQUIRE(result <= statefulVectorEngine.max());
    }
}

TEST_CASE_METHOD(PhiloxTest, "HostStatefulSingleEngineTest")
{
    auto result = statefulSingleEngine();
    REQUIRE(result >= statefulSingleEngine.min());
    REQUIRE(result <= statefulSingleEngine.max());
}

template<typename T>
class PhiloxTestKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename T_Generator>
    ALPAKA_FN_ACC void genNumbers(TAcc const& acc, bool* success, T_Generator& gen) const
    {
        {
            static_cast<void>(acc);
            alpaka::rand::UniformReal<T> dist;
            auto const r = dist(gen);
            ALPAKA_CHECK(*success, static_cast<T>(0.0) <= r);
            ALPAKA_CHECK(*success, static_cast<T>(1.0) > r);
        }
    }

public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        // Philox generator for accelerator
        auto generator = alpaka::rand::Philox4x32x10(42, 12345, 6789);
        genNumbers<TAcc, decltype(generator)>(acc, success, generator);
    }
};

using TestScalars = std::tuple<float, double>;
using TestTypes = alpaka::meta::CartesianProduct<std::tuple, alpaka::test::TestAccs, TestScalars>;

TEMPLATE_LIST_TEST_CASE("PhiloxRandomGeneratorIsWorking", "[rand]", TestTypes)
{
    using Acc = std::tuple_element_t<0, TestType>;
    using DataType = std::tuple_element_t<1, TestType>;
    using Idx = alpaka::Idx<Acc>;
    using Dim = alpaka::Dim<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    PhiloxTestKernel<DataType> kernel;

    REQUIRE(fixture(kernel));
}
