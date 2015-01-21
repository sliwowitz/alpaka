/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

// Base classes.
#include <alpaka/openmp/AccOpenMpFwd.hpp>
#include <alpaka/openmp/WorkExtent.hpp>             // InterfacedWorkExtentOpenMp
#include <alpaka/openmp/Index.hpp>                  // InterfacedIndexOpenMp
#include <alpaka/openmp/Atomic.hpp>                 // InterfacedAtomicOpenMp

// User functionality.
#include <alpaka/host/Memory.hpp>                   // MemCopy
#include <alpaka/openmp/Event.hpp>                  // Event
#include <alpaka/openmp/Stream.hpp>                 // Stream
#include <alpaka/openmp/Device.hpp>                 // Devices

// Specialized templates.
#include <alpaka/interfaces/KernelExecCreator.hpp>  // KernelExecCreator

// Implementation details.
#include <alpaka/openmp/Common.hpp>
#include <alpaka/interfaces/BlockSharedExternMemSizeBytes.hpp>
#include <alpaka/interfaces/IAcc.hpp>

#include <cstddef>                                  // std::size_t
#include <cstdint>                                  // std::uint32_t
#include <vector>                                   // std::vector
#include <cassert>                                  // assert
#include <stdexcept>                                // std::except
#include <string>                                   // std::to_string
#include <algorithm>                                // std::min, std::max
#ifdef ALPAKA_DEBUG
    #include <iostream>                             // std::cout
#endif

#include <boost/mpl/apply.hpp>                      // boost::mpl::apply

namespace alpaka
{
    namespace openmp
    {
        namespace detail
        {
            template<
                typename TAcceleratedKernel>
            class KernelExecutorOpenMp;

            //#############################################################################
            //! The OpenMP accelerator.
            //!
            //! This accelerator allows parallel kernel execution on the host.
            // \TODO: Offloading?
            //! It uses OpenMP to implement the parallelism.
            //#############################################################################
            class AccOpenMp :
                protected InterfacedWorkExtentOpenMp,
                protected InterfacedIndexOpenMp,
                protected InterfacedAtomicOpenMp
            {
            public:
                using MemorySpace = alpaka::memory::MemSpaceHost;

                template<
                    typename TAcceleratedKernel>
                friend class alpaka::openmp::detail::KernelExecutorOpenMp;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA AccOpenMp() :
                    InterfacedWorkExtentOpenMp(),
                    InterfacedIndexOpenMp(*static_cast<InterfacedWorkExtentOpenMp const *>(this), m_v3uiGridBlockIdx),
                    InterfacedAtomicOpenMp()
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                // Do not copy most members because they are initialized by the executor for each accelerated execution.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA AccOpenMp(AccOpenMp const &) :
                    InterfacedWorkExtentOpenMp(),
                    InterfacedIndexOpenMp(*static_cast<InterfacedWorkExtentOpenMp const *>(this), m_v3uiGridBlockIdx),
                    InterfacedAtomicOpenMp(),
                    m_v3uiGridBlockIdx(),
                    m_vvuiSharedMem(),
                    m_vuiExternalSharedMem()
                {}
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA AccOpenMp(AccOpenMp &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA AccOpenMp & operator=(AccOpenMp const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA virtual ~AccOpenMp() noexcept = default;

            protected:
                //-----------------------------------------------------------------------------
                //! \return The requested index.
                //-----------------------------------------------------------------------------
                template<
                    typename TOrigin, 
                    typename TUnit, 
                    typename TDimensionality = dim::Dim3>
                ALPAKA_FCT_ACC_NO_CUDA typename dim::DimToVecT<TDimensionality> getIdx() const
                {
                    return this->InterfacedIndexOpenMp::getIdx<TOrigin, TUnit, TDimensionality>(
                        *static_cast<InterfacedWorkExtentOpenMp const *>(this));
                }

                //-----------------------------------------------------------------------------
                //! Syncs all kernels in the current block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA void syncBlockKernels() const
                {
                    #pragma omp barrier
                }

                //-----------------------------------------------------------------------------
                //! \return Allocates block shared memory.
                //-----------------------------------------------------------------------------
                template<
                    typename T, 
                    std::size_t TuiNumElements>
                ALPAKA_FCT_ACC_NO_CUDA T * allocBlockSharedMem() const
                {
                    static_assert(TuiNumElements > 0, "The number of elements to allocate in block shared memory must not be zero!");

                    // Assure that all threads have executed the return of the last allocBlockSharedMem function (if there was one before).
                    syncBlockKernels();

                    // Arbitrary decision: The thread with id 0 has to allocate the memory.
                    if(::omp_get_thread_num() == 0)
                    {
                        // \TODO: C++14 std::make_unique would be better.
                        m_vvuiSharedMem.emplace_back(
                            std::unique_ptr<uint8_t[]>(
                                reinterpret_cast<uint8_t*>(new T[TuiNumElements])));
                    }
                    syncBlockKernels();

                    return reinterpret_cast<T*>(m_vvuiSharedMem.back().get());
                }

                //-----------------------------------------------------------------------------
                //! \return The pointer to the externally allocated block shared memory.
                //-----------------------------------------------------------------------------
                template<
                    typename T>
                ALPAKA_FCT_ACC_NO_CUDA T * getBlockSharedExternMem() const
                {
                    return reinterpret_cast<T*>(m_vuiExternalSharedMem.get());
                }

#ifdef ALPAKA_NVCC_FRIEND_ACCESS_BUG
            protected:
#else
            private:
#endif
                // getIdx
                Vec<3u> mutable m_v3uiGridBlockIdx;                         //!< The index of the currently executed block.

                // allocBlockSharedMem
                std::vector<
                    std::unique_ptr<uint8_t[]>> mutable m_vvuiSharedMem;    //!< Block shared memory.

                // getBlockSharedExternMem
                std::unique_ptr<uint8_t[]> mutable m_vuiExternalSharedMem;  //!< External block shared memory.
            }; 

            //#############################################################################
            //! The OpenMP accelerator executor.
            //#############################################################################
            template<
                typename TAcceleratedKernel>
            class KernelExecutorOpenMp :
                private TAcceleratedKernel,
                private IAcc<AccOpenMp>
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkExtent, 
                    typename... TKernelConstrArgs>
                ALPAKA_FCT_HOST KernelExecutorOpenMp(
                    IWorkExtent<TWorkExtent> const & workExtent, 
                    StreamOpenMp const &, 
                    TKernelConstrArgs && ... args) :
                    TAcceleratedKernel(std::forward<TKernelConstrArgs>(args)...)
                {
#ifdef ALPAKA_DEBUG
                    std::cout << "[+] AccOpenMp::KernelExecutorOpenMp()" << std::endl;
#endif
                    (*static_cast<InterfacedWorkExtentOpenMp *>(this)) = workExtent;

                    /*auto const uiNumKernelsPerBlock(workExtent.template getExtent<Block, Kernels, dim::Dim1>()[0]);
                    auto const uiMaxKernelsPerBlock(AccOpenMp::getExtentBlockKernelsLinearMax());
                    if(uiNumKernelsPerBlock > uiMaxKernelsPerBlock)
                    {
                        throw std::runtime_error(("The given block kernels count '" + std::to_string(uiNumKernelsPerBlock) + "' is larger then the supported maximum of '" + std::to_string(uiMaxKernelsPerBlock) + "' by the OpenMp accelerator!").c_str());
                    }*/

                    m_v3uiGridBlocksExtent = workExtent.template getExtent<Grid, Blocks, dim::Dim3>();
                    m_v3uiBlockKernelsExtent = workExtent.template getExtent<Block, Kernels, dim::Dim3>();
#ifdef ALPAKA_DEBUG
                    std::cout << "[-] AccOpenMp::KernelExecutorOpenMp()" << std::endl;
#endif
                }
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutorOpenMp(KernelExecutorOpenMp const &) = default;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutorOpenMp(KernelExecutorOpenMp &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutorOpenMp & operator=(KernelExecutorOpenMp const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST virtual ~KernelExecutorOpenMp() noexcept = default;

                //-----------------------------------------------------------------------------
                //! Executes the accelerated kernel.
                //-----------------------------------------------------------------------------
                template<
                    typename... TArgs>
                ALPAKA_FCT_HOST void operator()(
                    TArgs && ... args) const
                {
#ifdef ALPAKA_DEBUG
                    std::cout << "[+] AccOpenMp::KernelExecutorOpenMp::operator()" << std::endl;
#endif
                    auto const uiBlockSharedExternMemSizeBytes(BlockSharedExternMemSizeBytes<TAcceleratedKernel>::getBlockSharedExternMemSizeBytes(m_v3uiBlockKernelsExtent, std::forward<TArgs>(args)...));
                    this->AccOpenMp::m_vuiExternalSharedMem.reset(
                        new uint8_t[uiBlockSharedExternMemSizeBytes]);

                    // The number of threads in this block.
                    auto const uiNumKernelsInBlock(this->AccOpenMp::getExtent<Block, Kernels, dim::Dim1>()[0]);

                    // Execute the blocks serially.
                    for(std::uint32_t bz(0); bz<m_v3uiGridBlocksExtent[2]; ++bz)
                    {
                        this->AccOpenMp::m_v3uiGridBlockIdx[2] = bz;
                        for(std::uint32_t by(0); by<m_v3uiGridBlocksExtent[1]; ++by)
                        {
                            this->AccOpenMp::m_v3uiGridBlockIdx[1] = by;
                            for(std::uint32_t bx(0); bx<m_v3uiGridBlocksExtent[0]; ++bx)
                            {
                                this->AccOpenMp::m_v3uiGridBlockIdx[0] = bx;

                                // Execute the threads in parallel threads.

                                // Force the environment to use the given number of threads.
                                ::omp_set_dynamic(0);

                                // Parallel execution of the kernels in a block is required because when syncBlockKernels is called all of them have to be done with their work up to this line.
                                // So we have to spawn one real thread per kernel in a block.
                                // 'omp for' is not useful because it is meant for cases where multiple iterations are executed by one thread but in our case a 1:1 mapping is required.
                                // Therefore we use 'omp parallel' with the specified number of threads in a block.
                                //
                                // \TODO: Does this hinder executing multiple kernels in parallel because their block sizes/omp thread numbers are interfering? Is this num_threads global? Is this a real use case? 
                                #pragma omp parallel num_threads(static_cast<int>(uiNumKernelsInBlock))
                                {
#ifdef ALPAKA_DEBUG
                                    if((::omp_get_thread_num() == 0) && (bz == 0) && (by == 0) && (bx == 0))
                                    {
                                        assert(::omp_get_num_threads()>=0);
                                        auto const uiNumThreads(static_cast<std::uint32_t>(::omp_get_num_threads()));
                                        std::cout << "omp_get_num_threads: " << uiNumThreads << std::endl;
                                        if(uiNumThreads != uiNumKernelsInBlock)
                                        {
                                            throw std::runtime_error("The OpenMP runtime did not use the number of threads that had been required!");
                                        }
                                    }
#endif
                                    this->TAcceleratedKernel::operator()(
                                        (*static_cast<IAcc<AccOpenMp> const *>(this)),
                                        std::forward<TArgs>(args)...);

                                    // Wait for all threads to finish before deleting the shared memory.
                                    this->AccOpenMp::syncBlockKernels();
                                }

                                // After a block has been processed, the shared memory can be deleted.
                                this->AccOpenMp::m_vvuiSharedMem.clear();
                            }
                        }
                    }
                    // After all blocks have been processed, the external shared memory can be deleted.
                    this->AccOpenMp::m_vuiExternalSharedMem.reset();
#ifdef ALPAKA_DEBUG
                    std::cout << "[-] AccOpenMp::KernelExecutorOpenMp::operator()" << std::endl;
#endif
                }

            private:
                Vec<3u> m_v3uiGridBlocksExtent;
                Vec<3u> m_v3uiBlockKernelsExtent;
            };
        }
    }

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The OpenMP accelerator kernel executor accelerator type trait specialization.
            //#############################################################################
            template<
                typename AcceleratedKernel>
            struct GetAcc<
                openmp::detail::KernelExecutorOpenMp<AcceleratedKernel>>
            {
                using type = AccOpenMp;
            };
        }
    }

    namespace detail
    {
        //#############################################################################
        //! The OpenMP accelerator kernel executor builder.
        //#############################################################################
        template<
            typename TKernel, 
            typename... TKernelConstrArgs>
        class KernelExecCreator<
            AccOpenMp, 
            TKernel, 
            TKernelConstrArgs...>
        {
        public:
            using AcceleratedKernel = typename boost::mpl::apply<TKernel, AccOpenMp>::type;
            using AcceleratedKernelExecutorExtent = KernelExecutorExtent<openmp::detail::KernelExecutorOpenMp<AcceleratedKernel>, TKernelConstrArgs...>;

            //-----------------------------------------------------------------------------
            //! Creates an kernel executor for the serial accelerator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST AcceleratedKernelExecutorExtent operator()(
                TKernelConstrArgs && ... args) const
            {
                return AcceleratedKernelExecutorExtent(std::forward<TKernelConstrArgs>(args)...);
            }
        };
    }
}
