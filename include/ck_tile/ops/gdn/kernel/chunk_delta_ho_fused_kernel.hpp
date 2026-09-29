// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/gdn/kernel/chunk_delta_h_scan_kernel.hpp"
#include "ck_tile/ops/gdn/gdn_numeric.hpp"
#include "ck_tile/core/arch/hcu_buffer_addressing.hpp"
#include "ck_tile/ops/gdn/pipeline/chunk_delta_h_scan_policy.hpp"

namespace ck_tile {

// Single-chunk H+O fusion, plus a one-token tail for T65. All public H,
// V_new and FP32 final-state buffers remain materialized. ComputeOutput=false
// reuses the guarded state scan without Q/O work. The host
// restricts this path to fixed B1/H2/HV8, G-only, plain-state workloads.
template <typename DataType, bool ComputeOutput = true, bool PackedH = false>
struct ChunkDeltaHOFusedKernel
{
    struct Kargs {
        ChunkDeltaHScanFwdKargs<DataType> state;
        const DataType* q;
        DataType* o;
        float scale;
    };
    using QK = GdnChunkMmacBlockGemm<DataType,64,64,128,256>;
    using PV = GdnChunkMmacBlockGemm<DataType,64,16,64,256>;
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
        float gate[128];
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
    CK_TILE_DEVICE static void transfer(DataType* dst, const DataType* src, index_t stride, index_t valid)
    {
        const index_t wave = get_warp_id(), lane = get_lane_id();
        const auto resource = make_wave_buffer_resource(src,valid*stride*sizeof(DataType));
        static_for<0,4,1>{}([&](auto issue) {
            const index_t row = issue * 16 + wave + (lane / 16) * 4;
            const index_t col = (lane % 16) * 8;
            const uintptr_t addr = reinterpret_cast<uintptr_t>(dst) +
                                   (issue * 2048 + wave * 512) * sizeof(DataType);
            hcu_async_buffer_load_asm_impl<DataType,8>(addr, resource,
                row<valid ? (row * stride + col) * sizeof(DataType) : 0x80000000u, 0);
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

    CK_TILE_DEVICE void operator()(Kargs all) const
    {
        const auto a=all.state;
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
                 a.num_value_heads * 128, a.total_tokens);

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
            const index_t valid_tokens=min(index_t{64},a.total_tokens-chunk*64);
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
            if(valid_tokens==1) {
                // The final singleton preserves all H/V/state contracts, using
                // FP32 dot products instead of running 63 zero MMAC rows.
                publish<0>();
                if constexpr(PackedH) {
                const index_t k=get_thread_id()%16*8,v=get_thread_id()/16;
                thread_buffer<DataType,8> snap;
                static_for<0,8,1>{}([&](auto j){snap(j)=sm.state[v*136+k+j];});
                const index_t off=(((vb/16)*16+k/8)*16+v)*8;
                buffer_store<16>{}(snap,make_wave_buffer_resource(a.h_start+hbase),off*sizeof(DataType),0,0);
            } else {
                static_for<0,2,1>{}([&](auto ki) {
                    const index_t k=get_thread_id()/4+ki*64,v=(get_thread_id()%4)*4;
                    thread_buffer<DataType,4> snap;
                    static_for<0,4,1>{}([&](auto j){snap(j)=sm.state[(v+j)*136+k];});
                    buffer_store<8>{}(snap,make_wave_buffer_resource(a.h_start+hbase),(k*128+vb+v)*sizeof(DataType),0,0);
                });
            }
                const index_t lane=get_lane_id(),wave=get_warp_id();
                const float gate=a.g[static_cast<long_index_t>(token)*a.num_value_heads+vh];
                const float eg=__builtin_amdgcn_exp2f(gate),eng=__builtin_amdgcn_exp2f(-gate);
                thread_buffer<DataType,2> wk,qk,kk;
                const auto qoff=(static_cast<long_index_t>(token)*a.num_qk_heads+qh)*128+state_k;
                const auto woff=(static_cast<long_index_t>(token)*a.num_value_heads+vh)*128+state_k;
                wk.template set_as<uint32_t>(number<0>{},*reinterpret_cast<const uint32_t*>(a.w+woff));
                qk.template set_as<uint32_t>(number<0>{},0u);
                if constexpr(ComputeOutput)
                    qk.template set_as<uint32_t>(number<0>{},*reinterpret_cast<const uint32_t*>(all.q+qoff));
                kk.template set_as<uint32_t>(number<0>{},*reinterpret_cast<const uint32_t*>(a.k+qoff));
                const float w0=type_convert<float>(wk[number<0>{}]),w1=type_convert<float>(wk[number<1>{}]);
                const float q0=type_convert<float>(qk[number<0>{}]),q1=type_convert<float>(qk[number<1>{}]);
                const float k0=type_convert<float>(kk[number<0>{}]),k1=type_convert<float>(kk[number<1>{}]);
                const float qg0=type_convert<float>(gdn_type_convert<DataType>(q0*eg));
                const float qg1=type_convert<float>(gdn_type_convert<DataType>(q1*eg));
                auto reduce16=[&](float x) {
                    static_for<0,4,1>{}([&](auto i){x+=bit_cast<float>(__builtin_amdgcn_ds_bpermute((lane^(1<<i))*4,bit_cast<uint32_t>(x)));});return x;
                };
                float* tmp=reinterpret_cast<float*>(sm.k);
                static_for<0,4,1>{}([&](auto j){
                    const float h0=type_convert<float>(gdn_type_convert<DataType>(state[0][j]));
                    const float h1=type_convert<float>(gdn_type_convert<DataType>(state[1][j]));
                    const float wsum=reduce16(w0*h0+w1*h1),qsum=reduce16(qg0*h0+qg1*h1);
                    if(lane%16==0){
                        tmp[wave*16+state_v+j*4]=wsum;
                        if constexpr(ComputeOutput)tmp[64+wave*16+state_v+j*4]=qsum;
                    }
                });
                if constexpr(ComputeOutput) {
                    const float score_part=reduce16(q0*k0+q1*k1);
                    if(lane==0)tmp[128+wave]=score_part;
                }
                __syncthreads();
                if(wave==0 && lane<16) {
                    const float proj=(tmp[lane]+tmp[16+lane])+(tmp[32+lane]+tmp[48+lane]);
                    const auto off=(static_cast<long_index_t>(token)*a.num_value_heads+vh)*128+vb+lane;
                    const DataType value=gdn_type_convert<DataType>(type_convert<float>(a.u[off])-proj);
                    sm.residual[lane]=value;a.v_new[off]=value;
                    if constexpr(ComputeOutput) {
                        const float hist=(tmp[64+lane]+tmp[80+lane])+(tmp[96+lane]+tmp[112+lane]);
                        const float score=(tmp[128]+tmp[129])+(tmp[130]+tmp[131]);
                        const float weight=type_convert<float>(gdn_type_convert<DataType>(score*eg*eng));
                        all.o[off]=gdn_type_convert<DataType>((hist+weight*type_convert<float>(value))*all.scale);
                    }
                }
                __syncthreads();
                static_for<0,4,1>{}([&](auto j){
                    const float value=type_convert<float>(sm.residual[state_v+j*4]);
                    state[0](j)=state[0][j]*eg+k0*value;
                    state[1](j)=state[1][j]*eg+k1*value;
                });
                continue;
            }
            const index_t lane=get_lane_id(), group=lane/16;
            const index_t t=get_warp_id()*16+lane%16, v=group*4;
            const long_index_t voff=(static_cast<long_index_t>(token+t)*a.num_value_heads+vh)*128+vb+v;
            uint32_t last_bits,cur_bits;uint64_t u_bits;
            const auto gr=make_wave_buffer_resource(a.g+vh,a.total_tokens*a.num_value_heads*4);
            const auto ur=make_wave_buffer_resource(a.u,a.total_tokens*a.num_value_heads*128*sizeof(DataType));
            const uint32_t gl_off=(token+valid_tokens-1)*a.num_value_heads*4;
            const uint32_t gc_off=t<valid_tokens ? (token+t)*a.num_value_heads*4 : 0x80000000u;
            const uint32_t u_off=t<valid_tokens ? voff*sizeof(DataType) : 0x80000000u;
            asm volatile("buffer_load_dword %0, %1, %2, 0 offen\n" : "=v"(last_bits) : "v"(gl_off),"s"(gr) : "memory");
            asm volatile("buffer_load_dword %0, %1, %2, 0 offen\n" : "=v"(cur_bits) : "v"(gc_off),"s"(gr) : "memory");
            asm volatile("buffer_load_dwordx2 %0, %1, %2, 0 offen\n" : "=v"(u_bits) : "v"(u_off),"s"(ur) : "memory");
            transfer(sm.k, a.k + (static_cast<long_index_t>(token) * a.num_qk_heads + qh) * 128,
                     a.num_qk_heads * 128, valid_tokens);
            // Four current-K D2L operations are younger than carried W.
            // Publish W/state while K remains in flight during projection.
            publish<7>();
            if constexpr(PackedH) {
                const index_t k=get_thread_id()%16*8,v=get_thread_id()/16;
                thread_buffer<DataType,8> snap;
                static_for<0,8,1>{}([&](auto j){snap(j)=sm.state[v*136+k+j];});
                const index_t off=(((vb/16)*16+k/8)*16+v)*8;
                buffer_store<16>{}(snap,make_wave_buffer_resource(a.h_start+hbase),off*sizeof(DataType),0,0);
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
            asm volatile("s_waitcnt vmcnt(6)\n" : "+v"(last_bits),"+v"(cur_bits),"+v"(u_bits) :: "memory");
            asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n s_barrier" ::: "memory");
            if constexpr(ComputeOutput)
                transfer(sm.w,all.q+(static_cast<long_index_t>(token)*a.num_qk_heads+qh)*128,
                         a.num_qk_heads*128, valid_tokens);
            const float g_last=bit_cast<float>(last_bits);
            const float decay = __builtin_amdgcn_exp2f(g_last);
            // Raw MMAC columns are {group, group+4, group+8, group+12}.
            // A lane-group transpose gives contiguous V4 for U/V_new traffic.
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
            u.template set_as<uint64_t>(number<0>{}, u_bits);
            const float g=bit_cast<float>(cur_bits);
            const float scale = __builtin_amdgcn_exp2f(g_last - g);
            thread_buffer<float,4> pr;
            pr(number<0>{})=p0; pr(number<1>{})=p1;
            pr(number<2>{})=p2; pr(number<3>{})=p3;
            thread_buffer<DataType,4> saved;
            static_for<0,4,1>{}([&](auto i) {
                const float value = t<valid_tokens ? type_convert<float>(u[i]) - pr[i] : 0.f;
                saved(i)=gdn_type_convert<DataType>(value);
                sm.residual[(v+i)*68+t]=gdn_type_convert<DataType>(value*scale);
            });
            if(a.save_new_value && t<valid_tokens)
                buffer_store<8>{}(saved, make_wave_buffer_resource(a.v_new +
                    (static_cast<long_index_t>(token) * a.num_value_heads + vh) * 128 + vb),
                    (t*a.num_value_heads*128+v)*sizeof(DataType),0,0);
            // K and residual must be resident before Update reads them.
            // Issue the next W only after this wait so it can remain pending.
            asm volatile("s_waitcnt lgkmcnt(0)\n s_barrier" ::: "memory");
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
            if constexpr(ComputeOutput) {
                // K and residual have been consumed by all four update waves.
                // The rounded chunk-start state remains alive for QH.
                __syncthreads();
                static_for<0,4,1>{}([&](auto i) {sm.residual[(v+i)*68+t]=saved[i];});
                if(get_thread_id()<64) {
                    const float g=get_thread_id()<valid_tokens ? a.g[static_cast<long_index_t>(token+get_thread_id())*a.num_value_heads+vh] : 0.f;
                    sm.gate[get_thread_id()]=__builtin_amdgcn_exp2f(g);
                    sm.gate[64+get_thread_id()]=__builtin_amdgcn_exp2f(-g);
                }
                publish<0>();
                auto kv=make_tensor_view<address_space_enum::lds>(sm.k,operand_descriptor<false>());
                auto kwq=make_tile_window(kv,make_tuple(number<64>{},number<128>{}),multi_index<2>{0,0});
                auto scores=QK::MakeCBlockTile();clear_tile(scores);QK{}(scores,ww,kwq);
                auto so=QK{}.MakeOuputLayout(scores);
                __syncthreads();
                // Preserve CK's rounded Q*exp(g) input to QH.
                static_for<0,4,1>{}([&](auto issue) {
                    const index_t row=issue*16+get_warp_id()+(lane/16)*4;
                    const index_t off=issue*2048+get_thread_id()*8;
                    thread_buffer<DataType,8> qvec;
                    qvec.template set_as<uint32x4_t>(number<0>{},*reinterpret_cast<const uint32x4_t*>(sm.w+off));
                    static_for<0,8,1>{}([&](auto j) {qvec(j)=gdn_type_convert<DataType>(type_convert<float>(qvec[j])*sm.gate[row]);});
                    *reinterpret_cast<uint32x4_t*>(sm.w+off)=qvec.template get_as<uint32x4_t>()[number<0>{}];
                });
                __syncthreads();
                auto qhacc=Projection::MakeCBlockTile();clear_tile(qhacc);Projection{}(qhacc,ww,hw);
                auto qhout=Projection{}.MakeOuputLayout(qhacc);
                __syncthreads();
                visit(so,[&](auto i,index_t row,index_t col) {
                    const float val=col<=row && row<valid_tokens && col<valid_tokens ? so[i]*sm.gate[row]*sm.gate[64+col] : 0.f;
                    sm.w[row*68+col]=gdn_type_convert<DataType>(val);
                });
                __syncthreads();
                auto pv=make_tensor_view<address_space_enum::lds>(sm.w,MakeGdnSimpleLdsDescriptor<64,64>());
                auto pw=make_tile_window(pv,make_tuple(number<64>{},number<64>{}),multi_index<2>{0,0});
                auto pvacc=PV::MakeCBlockTile();clear_tile(pvacc);PV{}(pvacc,pw,rw);
                auto pvout=PV{}.MakeOuputLayout(pvacc);
                {
                    float p0=(qhout.get_thread_buffer()[number<0>{}]+pvout.get_thread_buffer()[number<0>{}])*all.scale;
                    float p1=(qhout.get_thread_buffer()[number<1>{}]+pvout.get_thread_buffer()[number<1>{}])*all.scale;
                    float p2=(qhout.get_thread_buffer()[number<2>{}]+pvout.get_thread_buffer()[number<2>{}])*all.scale;
                    float p3=(qhout.get_thread_buffer()[number<3>{}]+pvout.get_thread_buffer()[number<3>{}])*all.scale;
                    if(group&1) {swap(p0,p1);swap(p2,p3);}
                    if(group&2) {swap(p0,p2);swap(p1,p3);}
                    p1=bit_cast<float>(__builtin_amdgcn_ds_bpermute((lane^16)*4,bit_cast<uint32_t>(p1)));
                    p2=bit_cast<float>(__builtin_amdgcn_ds_bpermute((lane^32)*4,bit_cast<uint32_t>(p2)));
                    p3=bit_cast<float>(__builtin_amdgcn_ds_bpermute((lane^48)*4,bit_cast<uint32_t>(p3)));
                    if(group&1) {swap(p0,p1);swap(p2,p3);}
                    if(group&2) {swap(p0,p2);swap(p1,p3);}
                    thread_buffer<DataType,4> result;
                    result(number<0>{})=gdn_type_convert<DataType>(p0);
                    result(number<1>{})=gdn_type_convert<DataType>(p1);
                    result(number<2>{})=gdn_type_convert<DataType>(p2);
                    result(number<3>{})=gdn_type_convert<DataType>(p3);
                    if(t<valid_tokens) buffer_store<8>{}(result,make_wave_buffer_resource(all.o),voff*sizeof(DataType),0,0);
                }
            }
            __syncthreads();
            if(next)
                transfer(sm.w, a.w + (static_cast<long_index_t>(token + 64) * a.num_value_heads + vh) * 128,
                         a.num_value_heads * 128, min(index_t{64},a.total_tokens-(chunk+1)*64));

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
