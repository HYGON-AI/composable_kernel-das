// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Info Technologies Ltd.

#pragma once

#include <string>
#include <map>
#include <mutex>
#include <hip/hip_runtime.h>
#include "ck/ck.hpp"
#include "ck_tile/host/device_prop.hpp"

namespace ck {

// HCUTargetEnum identifies ISA families, not a total feature order.
// gfx92a belongs to the gfx928 family with extra fp8/TLS support, while
// gfx938 belongs to the gfx936 family. Use the family helpers below for
// runtime architecture gating instead of relying on >= comparisons.
enum struct HCUTargetEnum
{
    HCU_TARGET_UNKNOWN = 0,
    HCU_TARGET_GFX928,
    HCU_TARGET_GFX92A,
    HCU_TARGET_GFX936,
    HCU_TARGET_GFX938,
    HCU_TARGET_GFX946,
};

inline std::string get_device_name()
{
    static const auto& ctx = ck_tile::hcu_device_ctx::get_instance();

    const std::string raw_name(ctx.device_prop().gcnArchName);

    // https://github.com/ROCmSoftwarePlatform/MIOpen/blob/8498875aef84878e04c1eabefdf6571514891086/src/target_properties.cpp#L40
    static std::map<std::string, std::string> device_name_map = {
        {"Ellesmere", "gfx803"},
        {"Baffin", "gfx803"},
        {"RacerX", "gfx803"},
        {"Polaris10", "gfx803"},
        {"Polaris11", "gfx803"},
        {"Tonga", "gfx803"},
        {"Fiji", "gfx803"},
        {"gfx800", "gfx803"},
        {"gfx802", "gfx803"},
        {"gfx804", "gfx803"},
        {"Vega10", "gfx900"},
        {"gfx901", "gfx900"},
        {"10.3.0 Sienna_Cichlid 18", "gfx1030"},
    };

    const auto name = raw_name.substr(0, raw_name.find(':')); // str.substr(0, npos) returns str.

    auto match = device_name_map.find(name);
    if(match != device_name_map.end())
        return match->second;
    return name;
}

inline HCUTargetEnum get_hcu_target_enum()
{
    static std::map<std::string, HCUTargetEnum> hcu_target_map = {
        {"gfx928", HCUTargetEnum::HCU_TARGET_GFX928},
        {"gfx936", HCUTargetEnum::HCU_TARGET_GFX936},
        {"gfx938", HCUTargetEnum::HCU_TARGET_GFX938},
        {"gfx92a", HCUTargetEnum::HCU_TARGET_GFX92A},
        {"gfx946", HCUTargetEnum::HCU_TARGET_GFX946},
    };

    const auto device_name = get_device_name();

    auto match = hcu_target_map.find(device_name);

    if(match != hcu_target_map.end())
        return match->second;
    return HCUTargetEnum::HCU_TARGET_UNKNOWN;
}

inline bool is_hcu_supported()
{
    return get_hcu_target_enum() != HCUTargetEnum::HCU_TARGET_UNKNOWN;
}

inline bool is_gfx928_family(HCUTargetEnum target)
{
    return target == HCUTargetEnum::HCU_TARGET_GFX928 ||
           target == HCUTargetEnum::HCU_TARGET_GFX92A ||
           target == HCUTargetEnum::HCU_TARGET_GFX946;
}

inline bool is_gfx92a_family(HCUTargetEnum target)
{
    return target == HCUTargetEnum::HCU_TARGET_GFX92A ||
           target == HCUTargetEnum::HCU_TARGET_GFX946;
}

inline bool is_gfx936_family(HCUTargetEnum target)
{
    return target == HCUTargetEnum::HCU_TARGET_GFX936 ||
           target == HCUTargetEnum::HCU_TARGET_GFX938 ||
           target == HCUTargetEnum::HCU_TARGET_GFX946;
}

inline bool is_any_mmac_arch(HCUTargetEnum target)
{
    return target != HCUTargetEnum::HCU_TARGET_UNKNOWN;
}

inline bool is_hcu_xdl_supported()
{
    return is_any_mmac_arch(get_hcu_target_enum());
}

inline bool is_gfx90a() { return ck::get_device_name() == "gfx90a"; }

inline bool is_gfx11_supported() { return false; }

inline bool is_gfx12_supported() { return false; }

inline bool is_gfx101_supported() { return false; }

inline bool is_gfx103_supported() { return false; }

inline bool is_gfx120_supported() { return false; }

inline bool is_gfx125_supported() { return false; }

inline bool is_xdl_supported()
{
    return is_hcu_xdl_supported() || ck::get_device_name() == "gfx908" ||
           ck::get_device_name() == "gfx90a" || ck::get_device_name() == "gfx942";
}

template <typename ADataType,
          typename BDataType,
          index_t MPerXDL64,
          index_t NPerXDL64,
          index_t MPerXDL32 = MPerXDL64,
          index_t NPerXDL32 = NPerXDL64>
inline bool is_xdl_wmma_supported()
{
    if(is_hcu_xdl_supported())
    {
        // HCU MMAC is 16x16 only; 32x32 / 4x4 AMD MFMA sizes cannot be emulated.
        return MPerXDL64 == 16 && NPerXDL64 == 16 && MPerXDL32 == 16 && NPerXDL32 == 16;
    }

    if(ck::get_device_name() == "gfx908" || ck::get_device_name() == "gfx90a" ||
       ck::get_device_name() == "gfx942")
    {
        return true;
    }

    return false;
}

template <typename ADataType, index_t KPerBlock, index_t KPack = 256>
inline bool is_xdl_wmma_k_supported()
{
    if(is_hcu_xdl_supported() || ck::get_device_name() == "gfx908" ||
       ck::get_device_name() == "gfx90a" || ck::get_device_name() == "gfx942")
    {
        return true;
    }

    if(is_gfx11_supported())
    {
        return (KPerBlock % 16 == 0) && (KPack % 16 == 0);
    }

    return false;
}

template <typename ADataType, index_t KPerBlock, index_t KPack = 256>
inline bool is_xdl_wmma_k_supported(Number<KPerBlock>, Number<KPack> = Number<KPack>{})
{
    return is_xdl_wmma_k_supported<ADataType, KPerBlock, KPack>();
}

inline bool is_lds_direct_load_supported() { return false; }

inline bool is_bf16_atomic_supported() { return false; }

inline bool is_wmma_supported() { return false; }

inline bool is_tf32_supported() { return false; }

inline int __host__ get_lds_size()
{
    int device = 0;
    int result = 0;
    auto status = hipGetDevice(&device);
    if(status == hipSuccess)
    {
        status = hipDeviceGetAttribute(&result, hipDeviceAttributeMaxSharedMemoryPerBlock, device);
        if(status == hipSuccess)
        {
            return result;
        }
    }

    return 64 * 1024;
}

template <typename ADataType,
          typename BDataType,
          index_t MPerXDL64,
          index_t NPerXDL64,
          index_t MPerXDL32 = MPerXDL64,
          index_t NPerXDL32 = NPerXDL64>
bool is_xdl_mmac_supported()
{
    if(is_hcu_xdl_supported())
    {
        if constexpr((MPerXDL32 != 16) || (NPerXDL32 != 16))
        {
            return false;
        }

        (void)sizeof(ADataType);
        (void)sizeof(BDataType);
        (void)MPerXDL64;
        (void)NPerXDL64;
        return true;
    }

    return false;
}

} // namespace ck
