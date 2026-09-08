// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>
#include <sstream>

#include "ck/stream_config.hpp"
#include "ck/utility/get_id.hpp"
#include "ck/utility/sequence.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <index_t BlockSize_,
          index_t MPerBlock_,
          index_t NPerBlock_,
          index_t MPerXDL_,
          index_t NPerXDL_,
          index_t MXdlPerWave_,
          bool IsWave64>
static constexpr auto GetXdlPerWave2()
{
    constexpr index_t Waves  = IsWave64 ? BlockSize_ / 64 : BlockSize_ / 32;
    constexpr index_t MWaves = MPerBlock_ / (MXdlPerWave_ * MPerXDL_);
    static_assert(MWaves > 0);

    constexpr index_t NWaves = Waves / MWaves;
    if constexpr(NWaves == 0)
    {
        return 0;
    }
    else
    {
        if constexpr(NPerBlock_ % (NPerXDL_ * NWaves) == 0)
        {
            return NPerBlock_ / (NWaves * NPerXDL_);
        }
        else
        {
            return 0;
        }
    }
}

template <index_t BlockSize_,
          index_t MPerBlock_,
          index_t NPerBlock_,
          index_t MPerXDL_,
          index_t NPerXDL_,
          index_t MXdlPerWave_,
          index_t CShuffleMXdlPerWavePerShuffle_,
          index_t CShuffleNXdlPerWavePerShuffle_,
          bool IsWave64>
static constexpr auto GetWarpTileConfig()
{
    constexpr auto MXdlPerWave64                   = MXdlPerWave_;
    constexpr auto MXdlPerWave32                   = MXdlPerWave_ * MPerXDL_ / 16;
    constexpr auto CShuffleMXdlPerWavePerShuffle32 = CShuffleMXdlPerWavePerShuffle_ * MPerXDL_ / 16;

    constexpr auto NXdlPerWave =
        IsWave64
            ? GetXdlPerWave2<BlockSize_,
                             MPerBlock_,
                             NPerBlock_,
                             MPerXDL_,
                             NPerXDL_,
                             MXdlPerWave_,
                             true>()
            : GetXdlPerWave2<BlockSize_, MPerBlock_, NPerBlock_, 16, 16, MXdlPerWave32, false>();

    if constexpr(IsWave64 == false && NXdlPerWave != 0)
    {
        constexpr auto CShuffleNXdlPerWavePerShuffle32 =
            (NXdlPerWave >= CShuffleNXdlPerWavePerShuffle_ * NPerXDL_ / 16) &&
                    (NXdlPerWave % (CShuffleNXdlPerWavePerShuffle_ * NPerXDL_ / 16) == 0)
                ? CShuffleNXdlPerWavePerShuffle_ * NPerXDL_ / 16
                : NXdlPerWave;
        static_assert(CShuffleNXdlPerWavePerShuffle32 > 0);
        return Sequence<16,
                        16,
                        MXdlPerWave32,
                        NXdlPerWave,
                        CShuffleMXdlPerWavePerShuffle32,
                        CShuffleNXdlPerWavePerShuffle32>{};
    }
    else
    {
        return Sequence<MPerXDL_,
                        NPerXDL_,
                        MXdlPerWave64,
                        NXdlPerWave,
                        CShuffleMXdlPerWavePerShuffle_,
                        CShuffleNXdlPerWavePerShuffle_>{};
    }
}

template <index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t MPerXdl,
          index_t NPerXdl,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename CDataType,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation_ = InMemoryDataOperationEnum::Set>
__device__ static bool constexpr IsValidGemmCompilationParameter()
{
#if defined(__gfx11__) || defined(__gfx12__)
    if constexpr(MPerXdl != 16 || NPerXdl != 16)
    {
        return false;
    }
#endif

#if defined(__gfx11__)
    constexpr bool SupportMemOp = CGlobalMemoryDataOperation_ == InMemoryDataOperationEnum::Set;
#else
    constexpr bool SupportMemOp =
        sizeof(CDataType) >= 2 || (CGlobalMemoryDataOperation_ == InMemoryDataOperationEnum::Set);
#endif
    if constexpr(SupportMemOp == false)
    {
        return false;
    }

    if constexpr(MXdlPerWave > 0 && NXdlPerWave > 0)
    {
        constexpr index_t MWaves = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWaves = NPerBlock / (NXdlPerWave * NPerXdl);
        if constexpr(MWaves > 0 && NWaves > 0)
        {
            constexpr index_t WaveSize = BlockSize / (MWaves * NWaves);
            return WaveSize == get_warp_size();
        }
    }
    return false;
}

#define IS_VALID_COMPILATION_PARAMETER_IMPL(CDataType_)                       \
    template <InMemoryDataOperationEnum CGlobalMemoryDataOperation_ =         \
                  InMemoryDataOperationEnum::Set>                             \
    __device__ static bool constexpr IsValidCompilationParameter()            \
    {                                                                         \
        return ck::tensor_operation::device::IsValidGemmCompilationParameter< \
            BlockSize,                                                        \
            MPerBlock,                                                        \
            NPerBlock,                                                        \
            MPerXDL,                                                          \
            NPerXDL,                                                          \
            MXdlPerWave,                                                      \
            NXdlPerWave,                                                      \
            CDataType_,                                                       \
            CGlobalMemoryDataOperation_>();                                   \
    }

#define INVOKER_RUN_IMPL                                                               \
    float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{}) \
    {                                                                                  \
        if(get_warp_size() == 64)                                                      \
        {                                                                              \
            if constexpr(NXdlPerWave64 > 0)                                            \
            {                                                                          \
                return RunImp<GridwiseGemm64>(arg, stream_config);                     \
            }                                                                          \
        }                                                                              \
        else                                                                           \
        {                                                                              \
            if constexpr(NXdlPerWave32 > 0)                                            \
            {                                                                          \
                return RunImp<GridwiseGemm32>(arg, stream_config);                     \
            }                                                                          \
        }                                                                              \
        return 0;                                                                      \
    }

struct BaseArgument
{
    BaseArgument()                    = default;
    BaseArgument(const BaseArgument&) = default;
    BaseArgument& operator=(const BaseArgument&) = default;

    virtual __host__ __device__ ~BaseArgument() {}

    void* p_workspace_ = nullptr;
};

struct BaseInvoker
{
    BaseInvoker()                   = default;
    BaseInvoker(const BaseInvoker&) = default;
    BaseInvoker& operator=(const BaseInvoker&) = default;

    virtual float Run(const BaseArgument*, const StreamConfig& = StreamConfig{})
    {
        return float{0};
    }

    virtual ~BaseInvoker() {}
};

struct BaseOperator
{
    BaseOperator()                    = default;
    BaseOperator(const BaseOperator&) = default;
    BaseOperator& operator=(const BaseOperator&) = default;

    virtual bool IsSupportedArgument(const BaseArgument*) { return false; }
    virtual std::string GetTypeString() const { return ""; }

    virtual std::string GetTypeIdName() const { return typeid(*this).name(); }

    virtual std::string GetTypeIdHashCode() const
    {
        std::ostringstream oss;

        oss << std::hex << typeid(*this).hash_code();

        return oss.str();
    };

    virtual size_t GetWorkSpaceSize(const BaseArgument*) const { return 0; }

    virtual void SetWorkSpacePointer(BaseArgument* p_arg, void* p_workspace) const
    {
        assert(p_arg);
        p_arg->p_workspace_ = p_workspace;
    }

    virtual ~BaseOperator() {}
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
