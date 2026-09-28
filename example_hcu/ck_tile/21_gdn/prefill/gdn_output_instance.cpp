// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "ck_tile/ops/gdn/kernel/gdn_output_streaming_kernel.hpp"
#include "ck_tile/ops/gdn/kernel/gdn_output_cooperative_kernel.hpp"
#include "gdn_prefill_launch.hpp"
#include "ck_tile/host/kernel_launch.hpp"

#include "ck_tile/ops/gdn/kernel/gdn_fwd_output_kernel.hpp"

namespace gdn_example {
namespace {

// The final chunk of T65 contains exactly one query and one key.
// One wave computes QH with MMAC and QK with a 128-element dot product;
// it does not build a padded 64x64 score tile or use cross-wave LDS.
template<typename DataType, bool PackedH>
struct GdnOutputSingletonMmacKernel
{
    using Core = gdn::GdnOutputCooperativeKernel<DataType,PackedH,16,16,true>;
    using Kargs = typename Core::Kargs;
    using Mmac = typename Core::Mmac;
    using A = typename Core::A;
    using C = typename Core::C;
    CK_TILE_DEVICE void operator()(Kargs a) const
    {
        using namespace ck_tile;
        if(get_warp_id()!=0)return;
        const index_t lane=get_lane_id(),group=lane/16,vh=blockIdx.y;
        const index_t vb=blockIdx.z*32;
        const auto qb=(static_cast<long_index_t>(64)*2+vh/4)*128;
        const auto hb=(static_cast<long_index_t>(8)+vh)*128*128;
        thread_buffer<DataType,2> qdot,kdot;
        qdot.template set_as<uint32_t>(number<0>{},*reinterpret_cast<const uint32_t*>(a.q+qb+lane*2));
        kdot.template set_as<uint32_t>(number<0>{},*reinterpret_cast<const uint32_t*>(a.k+qb+lane*2));
        float score=type_convert<float>(qdot[number<0>{}])*type_convert<float>(kdot[number<0>{}])+
                    type_convert<float>(qdot[number<1>{}])*type_convert<float>(kdot[number<1>{}]);
        static_for<0,6,1>{}([&](auto j){score+=Core::shuffle(score,lane^(1<<j));});
        const float gate=a.g[64*8+vh];
        const float eg=__builtin_amdgcn_exp2f(gate),eng=__builtin_amdgcn_exp2f(-gate);
        const float weight=type_convert<float>(gdn_type_convert<DataType>(score*eg*eng));
        C hist[2];hist[0]=C{0.f};hist[1]=C{0.f};
        static_for<0,4,1>{}([&](auto p){
            thread_buffer<DataType,8> q;
            q.template set_as<uint32x4_t>(number<0>{},*reinterpret_cast<const uint32x4_t*>(a.q+qb+p*32+group*8));
            static_for<0,2,1>{}([&](auto n){
                const index_t v=vb+n*16+lane%16,k=p*32+group*8;
                thread_buffer<DataType,8> h;
                if constexpr(PackedH)
                {
                    const index_t off=(((v/16)*16+k/8)*16+v%16)*8;
                    h.template set_as<uint32x4_t>(number<0>{},*reinterpret_cast<const uint32x4_t*>(a.h+hb+off));
                }
                else
                    static_for<0,8,1>{}([&](auto j){h(j)=a.h[hb+(k+j)*128+v];});
                Mmac{}(hist[n],q.template get_as<A>()[number<0>{}],h.template get_as<A>()[number<0>{}]);
                Mmac{}(hist[n],q.template get_as<A>()[number<1>{}],h.template get_as<A>()[number<1>{}]);
            });
        });
        static_for<0,2,1>{}([&](auto n){
            thread_buffer<float,4> hx;hx.template set_as<C>(number<0>{},hist[n]);Core::transpose(hx);
            if(lane%16==0)
            {
                const auto off=(static_cast<long_index_t>(64)*8+vh)*128+vb+n*16+group*4;
                thread_buffer<DataType,4> out;
                static_for<0,4,1>{}([&](auto j){
                    out(j)=gdn_type_convert<DataType>((hx[j]*eg+weight*type_convert<float>(a.v_new[off+j]))*a.scale);
                });
                buffer_store<8>{}(out,make_wave_buffer_resource(a.o),off*sizeof(DataType),0,0);
            }
        });
    }
};

// This dispatcher is used only for B1/T65. Its first two row CTAs are
// complete tiles; only the third CTA needs element-level tail predicates.
template<typename DataType, bool PackedH>
struct GdnOutputShortTailKernel
{
    using Kargs = gdn::GdnOutputFwdKargs<DataType>;
    static constexpr ck_tile::index_t kBlockSize = 256;
    CK_TILE_DEVICE void operator()(Kargs a) const
    {
        // Host dispatch has already checked this complete shape contract.
        a.total_tokens = 65;
        a.num_qk_heads = 2;
        a.num_value_heads = 8;
        if(blockIdx.x < 2)
            gdn::GdnOutputCooperativeKernel<DataType,PackedH,16,16,false>{}(a);
        else
        {
            if constexpr(std::is_same_v<DataType,ck_tile::fp16_t>)
                GdnOutputSingletonMmacKernel<DataType,PackedH>{}(a);
            else
                gdn::GdnOutputCooperativeKernel<DataType,PackedH,16,16,true>{}(a);
        }
    }
};

// Only instantiated behind exact head-count guards. Constant head
// strides let the compiler simplify the streamed Q/K/H address schedule.
template<typename Kernel, int QHeads, int VHeads>
struct GdnOutputKnownHeadsKernel
{
    using Kargs = typename Kernel::Kargs;
    static constexpr ck_tile::index_t kBlockSize = Kernel::kBlockSize;
    CK_TILE_DEVICE void operator()(Kargs a) const
    {
        a.num_qk_heads = QHeads;
        a.num_value_heads = VHeads;
        Kernel{}(a);
    }
};

constexpr ck_tile::index_t kValueSplitCtaThreshold = 96;

ck_tile::index_t select_group4_value_split(ck_tile::index_t num_chunks,
                                           ck_tile::index_t num_qk_heads)
{
    return num_chunks * num_qk_heads <= kValueSplitCtaThreshold ? 2 : 1;
}

} // namespace

template <typename DataType, int GroupSize>
void launch_output_impl(const PrefillArguments& args, hipStream_t stream)
{
    const int num_chunks = args.is_varlen
        ? args.num_chunks
        : ck_tile::integer_divide_ceil(args.t, 64);
    const gdn::GdnOutputFwdKargs<DataType> kargs{
        static_cast<const DataType*>(args.q),
        static_cast<const DataType*>(args.k),
        reinterpret_cast<const DataType*>(args.v_new),
        reinterpret_cast<const DataType*>(args.h),
        args.g_cum,
        args.is_varlen ? args.cu_seqlens : nullptr,
        args.is_varlen ? args.chunk_indices : nullptr,
        reinterpret_cast<DataType*>(args.output),
        args.scale,
        args.t,
        num_chunks,
        args.num_sequences,
        args.h_qk,
        args.h_v,
        args.is_varlen};
    const bool parallel_vh = args.t <= 256;
    const bool preshuffled_h = use_preshuffled_state(args);
    const auto launch = [&](auto kernel, dim3 grid) {
        using Kernel = decltype(kernel);
        const dim3 block(Kernel::kBlockSize, 1, 1);
        auto callable = ck_tile::make_kernel<Kernel::kBlockSize, 1>(
            kernel, grid, block, 0, kargs);
        callable(ck_tile::stream_config{stream, false});
    };
    if(!args.is_varlen && args.num_sequences==1 && args.h_qk==2 && args.h_v==8 &&
       args.t==65)
    {
        if(preshuffled_h)
            launch(GdnOutputShortTailKernel<DataType,true>{},dim3(3,args.h_v,4));
        else
            launch(GdnOutputShortTailKernel<DataType,false>{},dim3(3,args.h_v,4));
        return;
    }
    if(!args.is_varlen && args.num_sequences==1 && args.h_qk==2 && args.h_v==8 &&
       !preshuffled_h && args.t>0 && args.t<=64)
    {
        launch(gdn::GdnOutputCooperativeKernel<DataType,false,16,16,true>{},
               dim3(num_chunks*2,args.h_v,4));
        return;
    }
    if(!args.is_varlen && args.num_sequences==1 && args.h_qk==16 && args.h_v==64 &&
       preshuffled_h && args.t>=64 && args.t%64==0)
    {
        constexpr bool bf=std::is_same_v<DataType,ck_tile::bf16_t>;
        if((bf && (args.t<=768 || args.t==1024)) || (!bf && args.t<=512))
        {
            if(args.t>=(bf ? 512 : 192))
            {
                if constexpr(bf)
                    launch(GdnOutputKnownHeadsKernel<gdn::GdnOutputStreamingKernel<DataType,true,32,64>,16,64>{},dim3(num_chunks,args.h_v,1));
                else
                    launch(gdn::GdnOutputStreamingKernel<DataType,true,32,64>{},dim3(num_chunks,args.h_v,1));
            }
            else
            {
                if constexpr(bf)
                    launch(GdnOutputKnownHeadsKernel<gdn::GdnOutputCooperativeKernel<DataType,true,32,64>,16,64>{},dim3(num_chunks,args.h_v,1));
                else
                    launch(gdn::GdnOutputCooperativeKernel<DataType,true,32,64>{},dim3(num_chunks,args.h_v,1));
            }
            return;
        }
    }
    if(!args.is_varlen && args.num_sequences==1 && args.h_qk==2 && args.h_v==8 &&
       preshuffled_h && args.t>=256 && args.t%64==0 &&
       (args.t<=4096 || (std::is_same_v<DataType,ck_tile::half_t> && args.t<=8192)))
    {
        // The measured crossover depends on dtype. BF16 keeps the narrow
        // tile below 1024 tokens; FP16 benefits from QK reuse from 512 onward.
        constexpr int kWideValueMinTokens =
            std::is_same_v<DataType, ck_tile::bf16_t> ? 1024 : 512;
        if(args.t >= kWideValueMinTokens)
        {
            if constexpr(std::is_same_v<DataType,ck_tile::bf16_t>)
            {
                if(args.t>=3072)
                    launch(gdn::GdnOutputStreamingKernel<DataType,true,32,64,false,true>{},dim3(num_chunks,args.h_v,1));
                else
                    launch(gdn::GdnOutputCooperativeKernel<DataType,true,32,64>{},dim3(num_chunks,args.h_v,1));
            }
            else
            {
                if(args.t>=3072)
                    launch(GdnOutputKnownHeadsKernel<gdn::GdnOutputStreamingKernel<DataType,true,32,64,false,true>,2,8>{},dim3(num_chunks,args.h_v,1));
                else if(args.t>=1024)
                    launch(gdn::GdnOutputStreamingKernel<DataType,true,32,64>{},dim3(num_chunks,args.h_v,1));
                else
                    launch(gdn::GdnOutputCooperativeKernel<DataType,true,32,64>{},dim3(num_chunks,args.h_v,1));
            }
        }
        else
            launch(gdn::GdnOutputCooperativeKernel<DataType, true>{},
                   dim3(num_chunks, args.h_v, 2));
        return;
    }
    // Pair row epilogues only in the measured small-head range. BF16 long
    // sequences keep the previous schedule; its register timing is better.
    const bool pair_rows = !args.is_varlen && args.num_sequences == 1 &&
        args.h_qk == 2 && args.h_v == 8 && args.t > 256 && args.t <= 4096 &&
        (!args.is_bf16 || args.t <= 2048);
    if(pair_rows)
    {
        if(args.t <= 2048)
        {
            if(preshuffled_h)
                launch(gdn::GdnOutputTiledMmacFwdKernel<DataType, 1, 1, true, true, true>{},
                       dim3(num_chunks, args.h_v, 1));
            else
                launch(gdn::GdnOutputTiledMmacFwdKernel<DataType, 1, 1, false, true, true>{},
                       dim3(num_chunks, args.h_v, 1));
        }
        else
        {
            const auto split = select_group4_value_split(num_chunks, args.h_qk);
            if(split == 2)
            {
                if(preshuffled_h)
                    launch(gdn::GdnOutputTiledMmacFwdKernel<DataType, 4, 2, true, false, true>{},
                           dim3(num_chunks, args.h_qk, 2));
                else
                    launch(gdn::GdnOutputTiledMmacFwdKernel<DataType, 4, 2, false, false, true>{},
                           dim3(num_chunks, args.h_qk, 2));
            }
            else if(preshuffled_h)
                launch(gdn::GdnOutputTiledMmacFwdKernel<DataType, 4, 1, true, false, true>{},
                       dim3(num_chunks, args.h_qk, 1));
            else
                launch(gdn::GdnOutputTiledMmacFwdKernel<DataType, 4, 1, false, false, true>{},
                       dim3(num_chunks, args.h_qk, 1));
        }
        return;
    }
    // Small grids benefit from splitting the existing single-head row tile
    // along V. Larger grids retain score reuse to avoid extra QK work.
    if(parallel_vh && num_chunks >= 4 && args.h_v >= 32)
    {
        // Reuse one score tile across both row halves when the head grid
        // already supplies enough parallel work.
        launch(gdn::GdnOutputTiledMmacFwdKernel<DataType, 1, 1, false, true>{},
               dim3(num_chunks, args.h_v, 1));
    }
    else if(parallel_vh && args.h_v < 16 && num_chunks * args.h_v <= 64)
    {
        using Kernel = gdn::GdnOutputTiledMmacFwdKernel<DataType, 1, 2>;
        launch(Kernel{}, dim3(num_chunks * 2, args.h_v, 2));
    }
    else if(args.t > 256 && args.h_v < 16 &&
            (args.t <= 2048 ||
             (args.is_bf16 && args.t <= 4096 && GroupSize == 4)))
    {
        if(preshuffled_h)
            launch(gdn::GdnOutputTiledMmacFwdKernel<DataType, 1, 1, true, true>{},
                   dim3(num_chunks, args.h_v, 1));
        else
            launch(gdn::GdnOutputTiledMmacFwdKernel<DataType, 1, 1, false, true>{},
                   dim3(num_chunks, args.h_v, 1));
    }
    else if(parallel_vh)
    {
        using Kernel = gdn::GdnOutputTiledMmacFwdKernel<DataType, 1>;
        launch(Kernel{},
               dim3(num_chunks * (Kernel::kChunkSize / Kernel::kRowTile),
                    args.h_v,
                    1));
    }
    else if constexpr(GroupSize == 1)
    {
        if(preshuffled_h)
            launch(gdn::GdnOutputTiledMmacFwdKernel<DataType, 1, 1, true, true>{},
                   dim3(num_chunks, args.h_qk, 1));
        else
            launch(gdn::GdnOutputTiledMmacFwdKernel<DataType, 1, 1, false, true>{},
                   dim3(num_chunks, args.h_qk, 1));
    }
    else if constexpr(GroupSize == 2)
    {
        if(preshuffled_h)
            launch(gdn::GdnOutputTiledMmacFwdKernel<DataType, 2, 1, true>{},
                   dim3(num_chunks, args.h_qk, 1));
        else
            launch(gdn::GdnOutputTiledMmacFwdKernel<DataType, 2, 1, false>{},
                   dim3(num_chunks, args.h_qk, 1));
    }
    else
    {
        const auto value_split =
            select_group4_value_split(num_chunks, args.h_qk);
        if(value_split == 2)
        {
            if(preshuffled_h)
                launch(gdn::GdnOutputTiledMmacFwdKernel<DataType, 4, 2, true>{},
                       dim3(num_chunks, args.h_qk, 2));
            else
                launch(gdn::GdnOutputTiledMmacFwdKernel<DataType, 4, 2>{},
                       dim3(num_chunks, args.h_qk, 2));
        }
        else if(preshuffled_h)
            launch(gdn::GdnOutputTiledMmacFwdKernel<DataType, 4, 1, true>{},
                   dim3(num_chunks, args.h_qk, 1));
        else
            launch(gdn::GdnOutputTiledMmacFwdKernel<DataType, 4, 1>{},
                   dim3(num_chunks, args.h_qk, 1));
    }
}

template <typename DataType>
void dispatch_output(const PrefillArguments& args, hipStream_t stream)
{
    switch(args.h_v / args.h_qk)
    {
    case 1: launch_output_impl<DataType, 1>(args, stream); break;
    case 2: launch_output_impl<DataType, 2>(args, stream); break;
    case 4: launch_output_impl<DataType, 4>(args, stream); break;
    // The prefill runner validates that the head ratio is 1, 2 or 4.
    default: break;
    }
}

void launch_output_bf16(const PrefillArguments& args, hipStream_t stream)
{
    dispatch_output<ck_tile::bf16_t>(args, stream);
}

void launch_output_fp16(const PrefillArguments& args, hipStream_t stream)
{
    dispatch_output<ck_tile::fp16_t>(args, stream);
}

} // namespace gdn_example
