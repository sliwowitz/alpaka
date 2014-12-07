/**
* Copyright 2014 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of either the GNU General Public License or
* the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License and the GNU Lesser General Public License
* for more details.
*
* You should have received a copy of the GNU General Public License
* and the GNU Lesser General Public License along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <string>           // std::string

#include <boost/predef.h>   // BOOST_XXX

#if BOOST_OS_WINDOWS || BOOST_OS_CYGWIN
    #define NOMINMAX
    #define WIN32_LEAN_AND_MEAN
    // We could use some more macros to reduce the number of sub-headers included, but this would restrict user code.
    #include <windows.h>
#elif BOOST_OS_UNIX
    #include <cstdint>          // std::uint64_t, std::uin32_t
    #include <unistd.h>
    #include <sys/types.h>
    #include <sys/param.h>
    #if BOOST_OS_BSD
        #include <sys/sysctl.h>
    #endif
#endif

namespace alpaka
{
    namespace host
    {
#ifdef BOOST_ARCH_X86_AVAILABLE
        namespace detail
        {
    #if BOOST_COMP_GNUC || BOOST_COMP_CLANG
        #include <cpuid.h>
            //-----------------------------------------------------------------------------
            //!
            //-----------------------------------------------------------------------------
            void cpuid(std::uint32_t const level, std::uint32_t const subfunction, std::uint32_t ex[4])
            {
                __cpuid_count(level, subfunction, ex[0], ex[1], ex[2], ex[3]);
            }

    #elif BOOST_COMP_MSVC
        #include <intrin.h>
            //-----------------------------------------------------------------------------
            //! 
            //-----------------------------------------------------------------------------
            void cpuid(std::uint32_t const level, std::uint32_t const subfunction, std::uint32_t ex[4])
            {
                __cpuidex(reinterpret_cast<int*>(ex), level, subfunction);
            }
    #endif
        }
#endif
        //-----------------------------------------------------------------------------
        //! \return The name of the CPU the host code is running on.
        //-----------------------------------------------------------------------------
        inline std::string getCpuName()
        {
#if BOOST_ARCH_X86
            // Get extended ids.
            std::uint32_t ex[4] = {0};
            detail::cpuid(0x80000000, 0, ex);
            std::uint32_t const nExIds(ex[0]);

            // Get the information associated with each extended ID.
            char pCPUBrandString[0x40] = {0};
            for(std::uint32_t i(0x80000000); i<=nExIds; ++i)
            {
                detail::cpuid(i, 0, ex);

                // Interpret CPU brand string and cache information.
                if(i == 0x80000002)
                {
                    std::memcpy(pCPUBrandString, ex, sizeof(ex));
                }
                else if(i == 0x80000003)
                {
                    std::memcpy(pCPUBrandString + 16, ex, sizeof(ex));
                }
                else if(i == 0x80000004)
                {
                    std::memcpy(pCPUBrandString + 32, ex, sizeof(ex));
                }
            }
            return std::string(pCPUBrandString);
#else
            return "<unknown>";
#endif
        }
        //-----------------------------------------------------------------------------
        //! \return The frequency of the CPU the host code is running on.
        //-----------------------------------------------------------------------------
        // TODO: implement!
        /*inline std::size_t getCpuFrequency()
        {
            return 0;
        }*/
        //-----------------------------------------------------------------------------
        //! \return The number of bytes of global memory.
        //! Adapted from David Robert Nadeau: http://nadeausoftware.com/articles/2012/09/c_c_tip_how_get_physical_memory_size_system
        //-----------------------------------------------------------------------------
        inline std::size_t getGlobalMemorySizeBytes()
        {
#if BOOST_OS_WINDOWS
            MEMORYSTATUSEX status;
            status.dwLength = sizeof(status);
            GlobalMemoryStatusEx(&status);
            return static_cast<std::size_t>(status.ullTotalPhys);

#elif BOOST_OS_CYGWIN
            // New 64-bit MEMORYSTATUSEX isn't available.
            MEMORYSTATUS status;
            status.dwLength = sizeof(status);
            GlobalMemoryStatus(&status);
            return static_cast<std::size_t>(status.dwTotalPhys);

#elif BOOST_OS_UNIX
            // Unix : Prefer sysctl() over sysconf() except sysctl() with HW_REALMEM and HW_PHYSMEM which are not always reliable
    #if defined(CTL_HW) && (defined(HW_MEMSIZE) || defined(HW_PHYSMEM64))
            int const mib[2] = {CTL_HW,
        #if defined(HW_MEMSIZE)        // OSX
                HW_MEMSIZE
        #elif defined(HW_PHYSMEM64)    // NetBSD, OpenBSD.
                HW_PHYSMEM64
        #endif
            };
            std::uint64_t uiSize(0);
            std::size_t const uiSizeLen{sizeof(uiSize)};
            if(sysctl(mib, 2, &uiSize, &uiSizeLen, nullptr, 0) < 0)
            {
                throw std::logic_error("getGlobalMemorySizeBytes failed calling sysctl!");
            }
            return static_cast<std::size_t>(uiSize);

    #elif defined(_SC_AIX_REALMEM)                                             // AIX.
            return static_cast<std::size_t>(sysconf(_SC_AIX_REALMEM)) * static_cast<std::size_t>(1024);

    #elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)                    // Linux, FreeBSD, OpenBSD, Solaris.
            return static_cast<std::size_t>(sysconf(_SC_PHYS_PAGES)) * static_cast<std::size_t>(sysconf(_SC_PAGESIZE));

    #elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGE_SIZE)                   // Legacy.
            return static_cast<std::size_t>(sysconf(_SC_PHYS_PAGES)) * static_cast<std::size_t>(sysconf(_SC_PAGE_SIZE));

    #elif defined(CTL_HW) && (defined(HW_PHYSMEM) || defined(HW_REALMEM))    // DragonFly BSD, FreeBSD, NetBSD, OpenBSD, and OSX.
            int mib[2] = {CTL_HW,
        #if defined(HW_REALMEM)        // FreeBSD.
                HW_REALMEM;
        #elif defined(HW_PYSMEM)        // Others.
                HW_PHYSMEM;
        #endif
            };
            std::uint32_t uiSize(0);
            std::size_t const uiSizeLen{sizeof(uiSize)};
            if(sysctl(mib, 2, &uiSize, &uiSizeLen, nullptr, 0) < 0)
            {
                throw std::logic_error("getGlobalMemorySizeBytes failed calling sysctl!");
            }
            return static_cast<std::size_t>(uiSize);
    #endif

#else
            throw std::logic_error("getGlobalMemorySizeBytes not implemented for this system!");
#endif
        }
    }
}