// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// Copyright (c) 2026 Hygon Info Technologies Ltd.
// Modified by Hygon Information Technology Co., Ltd.
//
// Adapted for Hygon HCU: uses __builtin_hcu_raw_buffer_load*/store* and
// __builtin_hcu_make_buffer_rsrc instead of AMD builtins.
// gfx125+ specific features (async copy, constant AS workarounds) are gated.
// BF16 buffer atomics (gfx942+) gated behind architecture guards.

#pragma once
#include "data_type.hpp"
#include "ck/utility/amd_buffer_coherence.hpp"

namespace ck {

template <typename T>
union BufferResource
{
    __device__ constexpr BufferResource() : content{} {}

    int32x4_t content;
    StaticallyIndexedArray<T*, 2> address;
    StaticallyIndexedArray<int32_t, 4> range;
    StaticallyIndexedArray<int32_t, 4> config;
};

template <typename T>
__device__ int32x4_t make_wave_buffer_resource(T* p_wave, index_t element_space_size)
{
    BufferResource<T> wave_buffer_resource;

    // wavewise base address (64 bit)
    wave_buffer_resource.address(Number<0>{}) = const_cast<remove_cv_t<T>*>(p_wave);
    // wavewise range (32 bit)
    wave_buffer_resource.range(Number<2>{}) = element_space_size * sizeof(T);
    // wavewise setting (32 bit)
    wave_buffer_resource.config(Number<3>{}) = CK_BUFFER_RESOURCE_3RD_DWORD;

    return wave_buffer_resource.content;
}

template <typename T>
__device__ int32x4_t make_wave_buffer_resource_with_default_range(T* p_wave)
{
    BufferResource<T> wave_buffer_resource;

    wave_buffer_resource.address(Number<0>{}) = const_cast<remove_cv_t<T>*>(p_wave);
    wave_buffer_resource.range(Number<2>{}) = 0xffffffff;
    wave_buffer_resource.config(Number<3>{}) = CK_BUFFER_RESOURCE_3RD_DWORD;

    return wave_buffer_resource.content;
}

// HCU version: __builtin_hcu_make_buffer_rsrc uses short stride
template <typename T>
__device__ __amdgpu_buffer_rsrc_t make_wave_buffer_resource_new(T* p_wave,
                                                                index_t element_space_size)
{
    auto p         = const_cast<remove_cv_t<T>*>(p_wave);
    int32_t num    = static_cast<int32_t>(element_space_size * sizeof(T));
    auto flags     = CK_BUFFER_RESOURCE_3RD_DWORD;

#if defined(__gfx928__) || defined(__gfx92a__) || defined(__gfx936__) || defined(__gfx938__) || \
    defined(__gfx946__)
    return __builtin_hcu_make_buffer_rsrc(p, static_cast<short>(0), num, flags);
#else
    return __builtin_amdgcn_make_buffer_rsrc(p, 0, num, flags);
#endif
}

template <typename T>
__device__ __amdgpu_buffer_rsrc_t make_wave_buffer_resource_with_default_range_new(T* p_wave)
{
    auto p         = const_cast<remove_cv_t<T>*>(p_wave);
    int32_t num    = 0xffffffff;
    auto flags     = CK_BUFFER_RESOURCE_3RD_DWORD;

#if defined(__gfx928__) || defined(__gfx92a__) || defined(__gfx936__) || defined(__gfx938__) || \
    defined(__gfx946__)
    return __builtin_hcu_make_buffer_rsrc(p, static_cast<short>(0), num, flags);
#else
    return __builtin_amdgcn_make_buffer_rsrc(p, 0, num, flags);
#endif
}

// HCU raw buffer load wrappers: coherence parameter absorbed (HCU builtins don't take it)
template <AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ int8_t hcu_raw_buffer_load_b8(__amdgpu_buffer_rsrc_t rsrc,
                                          index_t voffset,
                                          index_t soffset)
{
    return static_cast<int8_t>(__builtin_hcu_raw_buffer_load_b8(rsrc, voffset, soffset));
}

template <AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ int16_t hcu_raw_buffer_load_b16(__amdgpu_buffer_rsrc_t rsrc,
                                            index_t voffset,
                                            index_t soffset)
{
    return __builtin_hcu_raw_buffer_load_b16(rsrc, voffset, soffset);
}

template <AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ int32_t hcu_raw_buffer_load_b32(__amdgpu_buffer_rsrc_t rsrc,
                                            index_t voffset,
                                            index_t soffset)
{
    return __builtin_hcu_raw_buffer_load_b32(rsrc, voffset, soffset);
}

template <AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ int32x2_t hcu_raw_buffer_load_b64(__amdgpu_buffer_rsrc_t rsrc,
                                               index_t voffset,
                                               index_t soffset)
{
    return __builtin_hcu_raw_buffer_load_b64(rsrc, voffset, soffset);
}

template <AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ int32x4_t hcu_raw_buffer_load_b128(__amdgpu_buffer_rsrc_t rsrc,
                                               index_t voffset,
                                               index_t soffset)
{
    return __builtin_hcu_raw_buffer_load_b128(rsrc, voffset, soffset);
}

template <index_t N, AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ typename vector_type<int8_t, N>::type
amd_buffer_load_impl_raw(__amdgpu_buffer_rsrc_t src_wave_buffer_resource,
                         index_t src_thread_addr_offset,
                         index_t src_wave_addr_offset)
{
    static_assert(N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32 || N == 64,
                  "wrong! not implemented");

    if constexpr(N == 1)
    {
        return hcu_raw_buffer_load_b8<coherence>(
            src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset);
    }
    else if constexpr(N == 2)
    {
        int16_t tmp = hcu_raw_buffer_load_b16<coherence>(
            src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset);
        return bit_cast<int8x2_t>(tmp);
    }
    else if constexpr(N == 4)
    {
        int32_t tmp = hcu_raw_buffer_load_b32<coherence>(
            src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset);
        return bit_cast<int8x4_t>(tmp);
    }
    else if constexpr(N == 8)
    {
        int32x2_t tmp = hcu_raw_buffer_load_b64<coherence>(
            src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset);
        return bit_cast<int8x8_t>(tmp);
    }
    else if constexpr(N == 16)
    {
        int32x4_t tmp = hcu_raw_buffer_load_b128<coherence>(
            src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset);
        return bit_cast<int8x16_t>(tmp);
    }
    else if constexpr(N == 32)
    {
        int32x4_t tmp0 = hcu_raw_buffer_load_b128<coherence>(
            src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset);
        int32x4_t tmp1 = hcu_raw_buffer_load_b128<coherence>(
            src_wave_buffer_resource, src_thread_addr_offset,
            src_wave_addr_offset + 4 * sizeof(int32_t));
        vector_type<int32_t, 8> tmp;
        tmp.AsType<int32x4_t>()(Number<0>{}) = tmp0;
        tmp.AsType<int32x4_t>()(Number<1>{}) = tmp1;
        return bit_cast<int8x32_t>(tmp);
    }
    else if constexpr(N == 64)
    {
        int32x4_t tmp0 = hcu_raw_buffer_load_b128<coherence>(
            src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset);
        int32x4_t tmp1 = hcu_raw_buffer_load_b128<coherence>(
            src_wave_buffer_resource, src_thread_addr_offset,
            src_wave_addr_offset + 4 * sizeof(int32_t));
        int32x4_t tmp2 = hcu_raw_buffer_load_b128<coherence>(
            src_wave_buffer_resource, src_thread_addr_offset,
            src_wave_addr_offset + 8 * sizeof(int32_t));
        int32x4_t tmp3 = hcu_raw_buffer_load_b128<coherence>(
            src_wave_buffer_resource, src_thread_addr_offset,
            src_wave_addr_offset + 12 * sizeof(int32_t));
        vector_type<int32_t, 16> tmp;
        tmp.AsType<int32x4_t>()(Number<0>{}) = tmp0;
        tmp.AsType<int32x4_t>()(Number<1>{}) = tmp1;
        tmp.AsType<int32x4_t>()(Number<2>{}) = tmp2;
        tmp.AsType<int32x4_t>()(Number<3>{}) = tmp3;
        return bit_cast<int8x64_t>(tmp);
    }
}

template <typename T,
          index_t N,
          AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ typename vector_type<T, N>::type
amd_buffer_load_impl(__amdgpu_buffer_rsrc_t src_wave_buffer_resource,
                     index_t src_thread_addr_offset,
                     index_t src_wave_addr_offset)
{
    static_assert(
        (is_same<T, double>::value && (N == 1 || N == 2 || N == 4 || N == 8)) ||
            (is_same<T, float>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, half_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, bhalf_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, int32_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, f8_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32)) ||
            (is_same<T, bf8_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32)) ||
            (is_same<T, int8_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32)) ||
            (is_same<T, uint8_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32)) ||
            (is_same<T, pk_i4_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32)),
        "wrong! not implemented");

    using r_t     = typename vector_type<T, N>::type;
    auto raw_data = amd_buffer_load_impl_raw<sizeof(T) * N, coherence>(
        src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset);
    return bit_cast<r_t>(raw_data);
}

// HCU raw buffer store wrappers
template <AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ void hcu_raw_buffer_store_b8(int8_t vdata,
                                         __amdgpu_buffer_rsrc_t rsrc,
                                         index_t voffset,
                                         index_t soffset)
{
    __builtin_hcu_raw_buffer_store_b8(vdata, rsrc, voffset, soffset, 0);
}

template <AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ void hcu_raw_buffer_store_b16(int16_t vdata,
                                          __amdgpu_buffer_rsrc_t rsrc,
                                          index_t voffset,
                                          index_t soffset)
{
    __builtin_hcu_raw_buffer_store_b16(vdata, rsrc, voffset, soffset, 0);
}

template <AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ void hcu_raw_buffer_store_b32(int32_t vdata,
                                          __amdgpu_buffer_rsrc_t rsrc,
                                          index_t voffset,
                                          index_t soffset)
{
    __builtin_hcu_raw_buffer_store_b32(vdata, rsrc, voffset, soffset, 0);
}

template <AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ void hcu_raw_buffer_store_b64(int32x2_t vdata,
                                          __amdgpu_buffer_rsrc_t rsrc,
                                          index_t voffset,
                                          index_t soffset)
{
    __builtin_hcu_raw_buffer_store_b64(vdata, rsrc, voffset, soffset, 0);
}

template <AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ void hcu_raw_buffer_store_b128(int32x4_t vdata,
                                           __amdgpu_buffer_rsrc_t rsrc,
                                           index_t voffset,
                                           index_t soffset)
{
    __builtin_hcu_raw_buffer_store_b128(vdata, rsrc, voffset, soffset, 0);
}

template <index_t N, AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ void
amd_buffer_store_impl_raw(const typename vector_type<int8_t, N>::type src_thread_data,
                          __amdgpu_buffer_rsrc_t dst_wave_buffer_resource,
                          index_t dst_thread_addr_offset,
                          index_t dst_wave_addr_offset)
{
    static_assert(N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32 || N == 64,
                  "wrong! not implemented");

    if constexpr(N == 1)
    {
        hcu_raw_buffer_store_b8<coherence>(
            src_thread_data, dst_wave_buffer_resource, dst_thread_addr_offset, dst_wave_addr_offset);
    }
    else if constexpr(N == 2)
    {
        hcu_raw_buffer_store_b16<coherence>(bit_cast<int16_t>(src_thread_data),
                                             dst_wave_buffer_resource,
                                             dst_thread_addr_offset,
                                             dst_wave_addr_offset);
    }
    else if constexpr(N == 4)
    {
        hcu_raw_buffer_store_b32<coherence>(bit_cast<int32_t>(src_thread_data),
                                             dst_wave_buffer_resource,
                                             dst_thread_addr_offset,
                                             dst_wave_addr_offset);
    }
    else if constexpr(N == 8)
    {
        hcu_raw_buffer_store_b64<coherence>(bit_cast<int32x2_t>(src_thread_data),
                                             dst_wave_buffer_resource,
                                             dst_thread_addr_offset,
                                             dst_wave_addr_offset);
    }
    else if constexpr(N == 16)
    {
        hcu_raw_buffer_store_b128<coherence>(bit_cast<int32x4_t>(src_thread_data),
                                              dst_wave_buffer_resource,
                                              dst_thread_addr_offset,
                                              dst_wave_addr_offset);
    }
    else if constexpr(N == 32)
    {
        vector_type<int32_t, 8> tmp{bit_cast<int32x8_t>(src_thread_data)};
        hcu_raw_buffer_store_b128<coherence>(tmp.template AsType<int32x4_t>()[Number<0>{}],
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset);
        hcu_raw_buffer_store_b128<coherence>(tmp.template AsType<int32x4_t>()[Number<1>{}],
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset + sizeof(int32_t) * 4);
    }
    else if constexpr(N == 64)
    {
        vector_type<int32_t, 16> tmp{bit_cast<int32x16_t>(src_thread_data)};
        hcu_raw_buffer_store_b128<coherence>(tmp.template AsType<int32x4_t>()[Number<0>{}],
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset);
        hcu_raw_buffer_store_b128<coherence>(tmp.template AsType<int32x4_t>()[Number<1>{}],
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset + sizeof(int32_t) * 4);
        hcu_raw_buffer_store_b128<coherence>(tmp.template AsType<int32x4_t>()[Number<2>{}],
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset + sizeof(int32_t) * 8);
        hcu_raw_buffer_store_b128<coherence>(tmp.template AsType<int32x4_t>()[Number<3>{}],
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset + sizeof(int32_t) * 12);
    }
}

template <typename T,
          index_t N,
          AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ void amd_buffer_store_impl(const typename vector_type<T, N>::type src_thread_data,
                                      __amdgpu_buffer_rsrc_t dst_wave_buffer_resource,
                                      index_t dst_thread_addr_offset,
                                      index_t dst_wave_addr_offset)
{
    static_assert(
        (is_same<T, double>::value && (N == 1 || N == 2 || N == 4 || N == 8)) ||
            (is_same<T, float>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, half_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, bhalf_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, int32_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, f8_fnuz_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, bf8_fnuz_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, fp8_storage_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, int8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)),
        "wrong! not implemented");

    using r_t = typename vector_type<int8_t, sizeof(T) * N>::type;

    amd_buffer_store_impl_raw<sizeof(T) * N, coherence>(bit_cast<r_t>(src_thread_data),
                                                        dst_wave_buffer_resource,
                                                        dst_thread_addr_offset,
                                                        dst_wave_addr_offset);
}

// Public API: buffer load
template <typename T,
          index_t N,
          AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ typename vector_type_maker<T, N>::type::type
amd_buffer_load_invalid_element_return_zero(const T* p_src_wave,
                                            index_t src_thread_element_offset,
                                            bool src_thread_element_valid,
                                            index_t src_element_space_size)
{
    const __amdgpu_buffer_rsrc_t src_wave_buffer_resource =
        make_wave_buffer_resource_new(p_src_wave, src_element_space_size);

    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);

    using vector_t = typename vector_type_maker<T, N>::type::type;
    using scalar_t = typename scalar_type<vector_t>::type;

    constexpr index_t vector_size = scalar_type<vector_t>::vector_size;

    vector_t tmp{amd_buffer_load_impl<scalar_t, vector_size, coherence>(
        src_wave_buffer_resource, src_thread_addr_offset, 0)};
    return src_thread_element_valid ? tmp : vector_t(0);
}

template <typename T,
          index_t N,
          AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ typename vector_type_maker<T, N>::type::type
amd_buffer_load_invalid_element_return_customized_value(const T* p_src_wave,
                                                        index_t src_thread_element_offset,
                                                        bool src_thread_element_valid,
                                                        index_t src_element_space_size,
                                                        T customized_value)
{
    const __amdgpu_buffer_rsrc_t src_wave_buffer_resource =
        make_wave_buffer_resource_new(p_src_wave, src_element_space_size);

    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);

    using vector_t = typename vector_type_maker<T, N>::type::type;
    using scalar_t = typename scalar_type<vector_t>::type;

    constexpr index_t vector_size = scalar_type<vector_t>::vector_size;

    vector_t tmp{amd_buffer_load_impl<scalar_t, vector_size, coherence>(
        src_wave_buffer_resource, src_thread_addr_offset, 0)};

    return src_thread_element_valid ? tmp : vector_t(customized_value);
}

// Public API: buffer store
template <typename T,
          index_t N,
          AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ void amd_buffer_store(const typename vector_type_maker<T, N>::type::type src_thread_data,
                                 T* p_dst_wave,
                                 const index_t dst_thread_element_offset,
                                 const bool dst_thread_element_valid,
                                 const index_t dst_element_space_size)
{
    const __amdgpu_buffer_rsrc_t dst_wave_buffer_resource =
        make_wave_buffer_resource_new(p_dst_wave, dst_element_space_size);

    index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);

    using vector_t                = typename vector_type_maker<T, N>::type::type;
    using scalar_t                = typename scalar_type<vector_t>::type;
    constexpr index_t vector_size = scalar_type<vector_t>::vector_size;

    if(dst_thread_element_valid)
    {
        amd_buffer_store_impl<scalar_t, vector_size, coherence>(
            src_thread_data, dst_wave_buffer_resource, dst_thread_addr_offset, 0);
    }
}

} // namespace ck
