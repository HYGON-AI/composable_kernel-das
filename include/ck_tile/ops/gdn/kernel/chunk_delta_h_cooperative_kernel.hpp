// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/gdn/kernel/chunk_delta_h_scan_kernel.hpp"
#include "ck_tile/ops/gdn/gdn_numeric.hpp"
#include "ck_tile/core/arch/hcu_buffer_addressing.hpp"
#include "ck_tile/ops/gdn/pipeline/chunk_delta_h_scan_policy.hpp"

namespace ck_tile {

// Four waves compute T16 projection slices and K32 update slices. FP32
// state stays in the update GEMM distribution; its rounded LDS image feeds
// the next projection and packed H publication. Disjoint W/K buffers carry
// next-chunk prefetch through the update, without producer-only waves.
// The caller selects aligned, fixed-length, G-only workloads. H publication
// supports both the plain and packed layouts; state accumulation stays FP32.
template <typename DataType, bool PreshuffledH = true>
struct ChunkDeltaHCooperativeKernel
{
    static constexpr index_t SnapshotStores=PreshuffledH?1:2;
    using Kargs = ChunkDeltaHScanFwdKargs<DataType>;
    using Projection = GdnChunkMmacBlockGemm<DataType, 64, 16, 128, 256>;
    using Mmac = std::conditional_t<std::is_same_v<DataType,bf16_t>,
        WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16,
        WarpGemmAttributeMmacImplF16F16F32M16N16K16>;
    using AVec = typename Mmac::AVecType;
    using CVec = typename Mmac::CVecType;
    CK_TILE_DEVICE static auto read_k(DataType* ptr, index_t stage)
    {
        const index_t lane=get_lane_id(), wave=get_warp_id();
        const index_t row=stage*16+lane/4;
        const index_t col=wave*4+lane%4;
        const index_t pos=((row/16*4+row%4)*64+(row%16)/4*16+col)*8;
        // Clang requires a C-style cast for the generic-to-LDS address-space conversion.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
        if constexpr(std::is_same_v<DataType,bf16_t>)
            return bit_cast<uint32x4_t>(__builtin_hcu_ds_read_m32x16_bf16_alt(
                (__attribute__((address_space(3))) short*)(ptr+pos)));
        else
            return bit_cast<uint32x4_t>(__builtin_hcu_ds_read_m32x16_f16_alt(
                (__attribute__((address_space(3))) __fp16*)(ptr+pos)));
#pragma clang diagnostic pop
    }
    CK_TILE_DEVICE static void transpose4(thread_buffer<float,4>& x)
    {
        const index_t lane=get_lane_id(),group=lane/16;
        auto swap=[](float& a,float& b){float t=a;a=b;b=t;};
        float p0=x[number<0>{}],p1=x[number<1>{}],p2=x[number<2>{}],p3=x[number<3>{}];
        if(group&1){swap(p0,p1);swap(p2,p3);}if(group&2){swap(p0,p2);swap(p1,p3);}
        p1=bit_cast<float>(__builtin_amdgcn_ds_bpermute((lane^16)*4,bit_cast<uint32_t>(p1)));
        p2=bit_cast<float>(__builtin_amdgcn_ds_bpermute((lane^32)*4,bit_cast<uint32_t>(p2)));
        p3=bit_cast<float>(__builtin_amdgcn_ds_bpermute((lane^48)*4,bit_cast<uint32_t>(p3)));
        if(group&1){swap(p0,p1);swap(p2,p3);}if(group&2){swap(p0,p2);swap(p1,p3);}
        x(number<0>{})=p0;x(number<1>{})=p1;x(number<2>{})=p2;x(number<3>{})=p3;
    }
    struct LdsStorage
    {
        DataType w[64 * 128];
        DataType k[64 * 128];
        alignas(16) DataType state[16 * 136];
        DataType residual[16 * 68];
    };

    // D2L image [T/16, T%4, (T%16)/4, K]. Each issue writes
    // four wave-contiguous 1KiB regions; views express both W and K^T.
    template <bool Transpose>
    CK_TILE_HOST_DEVICE static constexpr auto operand_descriptor()
    {
        constexpr auto d = make_naive_tensor_descriptor(
            make_tuple(number<4>{}, number<4>{}, number<4>{}, number<128>{}),
            make_tuple(number<2048>{}, number<128>{}, number<512>{}, number<1>{}),
            number<8>{}, number<1>{});
        if constexpr(Transpose)
            return transform_tensor_descriptor(d,
                make_tuple(make_pass_through_transform(number<128>{}),
                           make_merge_transform(make_tuple(number<4>{}, number<4>{}, number<4>{}))),
                make_tuple(sequence<3>{}, sequence<0,1,2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        else
            return transform_tensor_descriptor(d,
                make_tuple(make_merge_transform(make_tuple(number<4>{}, number<4>{}, number<4>{})),
                           make_pass_through_transform(number<128>{})),
                make_tuple(sequence<0,1,2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
    }
    CK_TILE_DEVICE static void transfer(DataType* dst, const DataType* src, index_t stride)
    {
        const index_t wave = get_warp_id(), lane = get_lane_id();
        const auto resource = make_wave_buffer_resource(src);
        static_for<0,4,1>{}([&](auto issue) {
            const index_t row = issue * 16 + wave + (lane / 16) * 4;
            const index_t col = (lane % 16) * 8;
            const uintptr_t addr = reinterpret_cast<uintptr_t>(dst) +
                                   (issue * 2048 + wave * 512) * sizeof(DataType);
            hcu_async_buffer_load_asm_impl<DataType,8>(addr, resource,
                (row * stride + col) * sizeof(DataType), 0);
        });
    }
    template <index_t Pending>
    CK_TILE_DEVICE static void publish()
    {
        asm volatile("s_waitcnt vmcnt(%0) lgkmcnt(0)\n s_barrier" :: "n"(Pending) : "memory");
    }

    template <typename Tile, typename F>
    CK_TILE_DEVICE static void visit(Tile& tile, F f)
    {
        constexpr auto spans = remove_cvref_t<Tile>::get_distributed_spans();
        sweep_tile_span(spans[number<0>{}], [&](auto i) {
            sweep_tile_span(spans[number<1>{}], [&](auto j) {
                constexpr auto idx = make_tuple(i, j);
                const auto x = get_x_indices_from_distributed_indices(
                    tile.get_tile_distribution(), idx);
                f(idx, x.at(number<0>{}), x.at(number<1>{}));
            });
        });
    }

    CK_TILE_DEVICE void operator()(Kargs a) const
    {
        __shared__ LdsStorage sm;
        const index_t seq = blockIdx.y / a.num_value_heads;
        const index_t vh = blockIdx.y % a.num_value_heads;
        const index_t qh = vh / (a.num_value_heads / a.num_qk_heads);
        const index_t vb = blockIdx.x * 16;
        const index_t bos = seq * a.total_tokens;
        const long_index_t state_base =
            (static_cast<long_index_t>(seq) * a.num_value_heads + vh) * 128 * 128;
        const index_t state_k=get_warp_id()*32+(get_lane_id()%16)*2;
        const index_t state_v=get_lane_id()/16;
        thread_buffer<float,4> state[2];
        static_for<0,2,1>{}([&](auto pair) {
            if(a.has_initial_state) {
                const auto off=state_base+(state_k+pair)*128+vb+state_v*4;
                state[pair].template set_as<uint32x4_t>(number<0>{},*reinterpret_cast<const uint32x4_t*>(a.initial_state+off));
                transpose4(state[pair]);
            } else static_for<0,4,1>{}([&](auto j){state[pair](j)=0.f;});
        });

        transfer(sm.w, a.w + (static_cast<long_index_t>(bos) * a.num_value_heads + vh) * 128,
                 a.num_value_heads * 128);

        auto wview = make_tensor_view<address_space_enum::lds>(
            sm.w, operand_descriptor<false>());
        auto hview = make_tensor_view<address_space_enum::lds>(
            sm.state, MakeGdnSimpleLdsDescriptor<16, 128, 8>());
        auto kview = make_tensor_view<address_space_enum::lds>(
            sm.k, operand_descriptor<true>());
        auto rview = make_tensor_view<address_space_enum::lds>(
            sm.residual, MakeGdnSimpleLdsDescriptor<16, 64>());
        auto ww = make_tile_window(wview, make_tuple(number<64>{}, number<128>{}),
                                  multi_index<2>{0, 0});
        auto hw = make_tile_window(hview, make_tuple(number<16>{}, number<128>{}),
                                  multi_index<2>{0, 0});
        auto kw = make_tile_window(kview, make_tuple(number<128>{}, number<64>{}),
                                  multi_index<2>{0, 0});
        auto rw = make_tile_window(rview, make_tuple(number<16>{}, number<64>{}),
                                  multi_index<2>{0, 0});
        for(index_t chunk = 0; chunk < a.num_chunks; ++chunk)
        {
            const index_t token = bos + chunk * 64;
            const bool next = chunk + 1 < a.num_chunks;
            const long_index_t hbase =
                (static_cast<long_index_t>(seq * a.num_chunks + chunk) * a.num_value_heads + vh)
                * 128 * 128;
            static_for<0,4,1>{}([&](auto j) {
                thread_buffer<DataType,2> pair;
                pair(number<0>{})=gdn_type_convert<DataType>(state[0][j]);
                pair(number<1>{})=gdn_type_convert<DataType>(state[1][j]);
                *reinterpret_cast<uint32_t*>(sm.state+(state_v+j*4)*136+state_k)=
                    pair.template get_as<uint32_t>()[number<0>{}];
            });
            const index_t lane = get_lane_id(), group = lane / 16;
            const index_t t = get_warp_id() * 16 + lane % 16;
            const index_t v = group * 4;
            const long_index_t voff =
                (static_cast<long_index_t>(token + t) * a.num_value_heads + vh) * 128 + vb + v;
            uint32_t last_bits,cur_bits;
            uint64_t u_bits;
            auto gr=make_wave_buffer_resource(a.g+vh);
            auto ur=make_wave_buffer_resource(a.u);
            const uint32_t gl_off=(token+63)*a.num_value_heads*4;
            const uint32_t gc_off=(token+t)*a.num_value_heads*4;
            const uint32_t u_off=voff*sizeof(DataType);
            asm volatile("buffer_load_dword %0, %1, %2, 0 offen\n" : "=v"(last_bits) : "v"(gl_off),"s"(gr) : "memory");
            asm volatile("buffer_load_dword %0, %1, %2, 0 offen\n" : "=v"(cur_bits) : "v"(gc_off),"s"(gr) : "memory");
            asm volatile("buffer_load_dwordx2 %0, %1, %2, 0 offen\n" : "=v"(u_bits) : "v"(u_off),"s"(ur) : "memory");
            transfer(sm.k, a.k + (static_cast<long_index_t>(token) * a.num_qk_heads + qh) * 128,
                     a.num_qk_heads * 128);
            // Four current-K D2L operations are younger than carried W.
            // Publish W/state while K remains in flight during projection.
            publish<7>();
            // Packed H uses K8 vectors; all 256 threads publish one vector
            // from the already rounded shared state image.
            if constexpr(PreshuffledH) {
            const index_t snapshot_v = get_thread_id() / 16;
            const index_t snapshot_k = (get_thread_id() % 16) * 8;
            using HVector = uint32x4_t;
            const HVector hp = *reinterpret_cast<const HVector*>(
                sm.state + snapshot_v * 136 + snapshot_k);
            const index_t hoff = (((vb / 16) * 16 + snapshot_k / 8) * 16 + snapshot_v) * 8;
            buffer_store<16>{}(hp, make_wave_buffer_resource(a.h_start + hbase),
                               hoff * sizeof(DataType), 0, 0);
            } else {
            static_for<0,2,1>{}([&](auto ki) {
                const index_t k=get_thread_id()/4+ki*64;
                const index_t v=(get_thread_id()%4)*4;
                thread_buffer<DataType,4> snap;
                static_for<0,4,1>{}([&](auto j) {snap(j)=sm.state[(v+j)*136+k];});
                buffer_store<8>{}(snap,make_wave_buffer_resource(a.h_start+hbase),
                    (k*128+vb+v)*sizeof(DataType),0,0);
            });
            }
            auto projection = Projection::MakeCBlockTile();
            clear_tile(projection);
            Projection{}(projection, ww, hw);
            auto po = Projection{}.MakeOuputLayout(projection);
            // K (four requests) and H store are younger than early G/U.
            // Register dependencies keep their consumers after the wait.
            asm volatile("s_waitcnt vmcnt(%3)\n" : "+v"(last_bits),"+v"(cur_bits),"+v"(u_bits) : "n"(4+SnapshotStores) : "memory");
            asm volatile("s_waitcnt lgkmcnt(0)\n s_barrier" ::: "memory");
            if(next)
                transfer(sm.w, a.w + (static_cast<long_index_t>(token + 64) * a.num_value_heads + vh) * 128,
                         a.num_value_heads * 128);
            const float g_last=bit_cast<float>(last_bits);
            const float decay=__builtin_amdgcn_exp2f(g_last);
            float p0 = po.get_thread_buffer()[number<0>{}];
            float p1 = po.get_thread_buffer()[number<1>{}];
            float p2 = po.get_thread_buffer()[number<2>{}];
            float p3 = po.get_thread_buffer()[number<3>{}];
            auto swap = [](float& x, float& y) { float tmp=x; x=y; y=tmp; };
            if(group & 1) { swap(p0,p1); swap(p2,p3); }
            if(group & 2) { swap(p0,p2); swap(p1,p3); }
            p1 = bit_cast<float>(__builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, bit_cast<uint32_t>(p1)));
            p2 = bit_cast<float>(__builtin_amdgcn_ds_bpermute((lane ^ 32) * 4, bit_cast<uint32_t>(p2)));
            p3 = bit_cast<float>(__builtin_amdgcn_ds_bpermute((lane ^ 48) * 4, bit_cast<uint32_t>(p3)));
            if(group & 1) { swap(p0,p1); swap(p2,p3); }
            if(group & 2) { swap(p0,p2); swap(p1,p3); }
            thread_buffer<DataType,4> u;
            u.template set_as<uint64_t>(number<0>{},u_bits);
            const float g=bit_cast<float>(cur_bits);
            const float scale = __builtin_amdgcn_exp2f(g_last - g);
            thread_buffer<float,4> pr;
            pr(number<0>{})=p0; pr(number<1>{})=p1;
            pr(number<2>{})=p2; pr(number<3>{})=p3;
            thread_buffer<DataType,4> saved;
            static_for<0,4,1>{}([&](auto i) {
                const float value = type_convert<float>(u[i]) - pr[i];
                saved(i)=gdn_type_convert<DataType>(value);
                sm.residual[(v+i)*68+t]=gdn_type_convert<DataType>(value*scale);
            });
            if(a.save_new_value)
                buffer_store<8>{}(saved, make_wave_buffer_resource(a.v_new +
                    (static_cast<long_index_t>(token) * a.num_value_heads + vh) * 128 + vb),
                    (t*a.num_value_heads*128+v)*sizeof(DataType),0,0);
            // K and residual must be resident before Update reads them.
            // Issue the next W only after this wait so it can remain pending.
            if(next) {
                if(a.save_new_value) publish<5+SnapshotStores>(); else publish<4+SnapshotStores>();
            } else {
                if(a.save_new_value) publish<1+SnapshotStores>(); else publish<SnapshotStores>();
            }
            CVec delta0{0.f},delta1{0.f};
            static_for<0,4,1>{}([&](auto stage) {
                thread_buffer<DataType,8> kpack;
                kpack.template set_as<uint32x4_t>(number<0>{},read_k(sm.k,stage));
                thread_buffer<DataType,4> residual;
                residual.template set_as<uint64_t>(number<0>{},
                    *reinterpret_cast<const uint64_t*>(sm.residual+
                        (get_lane_id()%16)*68+stage*16+(get_lane_id()/16)*4));
                const AVec a0=kpack.template get_as<AVec>()[number<0>{}];
                const AVec a1=kpack.template get_as<AVec>()[number<1>{}];
                const AVec b=residual.template get_as<AVec>()[number<0>{}];
                Mmac{}(delta0,a0,b);
                Mmac{}(delta1,a1,b);
            });
            thread_buffer<float,4> d0,d1;
            d0.template set_as<CVec>(number<0>{},delta0);
            d1.template set_as<CVec>(number<0>{},delta1);
            static_for<0,4,1>{}([&](auto j) {
                state[0](j)=state[0][j]*decay+d0[j];
                state[1](j)=state[1][j]*decay+d1[j];
            });
            // All waves have consumed state/residual/K before the next chunk
            // can reuse the corresponding LDS storage.
            __syncthreads();
        }
        if(a.store_final_state)
            static_for<0,2,1>{}([&](auto pair) {
                auto out=state[pair];transpose4(out);
                const auto off=state_base+(state_k+pair)*128+vb+state_v*4;
                buffer_store<16>{}(out,make_wave_buffer_resource(a.final_state),off*sizeof(float),0,0);
            });
    }
};
} // namespace ck_tile
