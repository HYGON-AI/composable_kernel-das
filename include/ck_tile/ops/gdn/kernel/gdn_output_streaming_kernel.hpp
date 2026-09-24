// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/ops/gdn/pipeline/gdn_fwd_output_problem.hpp"
#include "ck_tile/ops/gdn/pipeline/chunk_delta_h_scan_policy.hpp"
namespace gdn {
template<typename DataType,bool PackedH,int Rows=32,int VTile=32,bool PrefetchV=std::is_same_v<DataType,ck_tile::bf16_t>,bool CausalPrune=false>
struct GdnOutputStreamingKernel {
    using Kargs=GdnOutputFwdKargs<DataType>;
    using Mmac=std::conditional_t<std::is_same_v<DataType,ck_tile::bf16_t>,
        ck_tile::WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16,
        ck_tile::WarpGemmAttributeMmacImplF16F16F32M16N16K16>;
    using A=typename Mmac::AVecType;using C=typename Mmac::CVecType;
    static constexpr ck_tile::index_t kBlockSize=256;
    CK_TILE_DEVICE static float shuffle(float x,int lane){return ck_tile::bit_cast<float>(__builtin_amdgcn_ds_bpermute(lane*4,ck_tile::bit_cast<uint32_t>(x)));}
    CK_TILE_DEVICE static void transpose(ck_tile::thread_buffer<float,4>& x){
        using namespace ck_tile;const int lane=get_lane_id(),g=lane/16;
        float a=x[number<0>{}],b=x[number<1>{}],c=x[number<2>{}],d=x[number<3>{}];
        auto sw=[](float& a,float& b){float t=a;a=b;b=t;};
        if(g&1){sw(a,b);sw(c,d);}if(g&2){sw(a,c);sw(b,d);}
        b=shuffle(b,lane^16);c=shuffle(c,lane^32);d=shuffle(d,lane^48);
        if(g&1){sw(a,b);sw(c,d);}if(g&2){sw(a,c);sw(b,d);}
        x(number<0>{})=a;x(number<1>{})=b;x(number<2>{})=c;x(number<3>{})=d;
    }
    CK_TILE_DEVICE void operator()(Kargs a)const{
        using namespace ck_tile;
        constexpr index_t MR=2,NR=VTile/16,Stride=2*VTile+8;
        static_assert(PackedH && Rows == 32 && (VTile == 32 || VTile == 64));
        // The four waves cover 64 rows and 2*VTile value columns. A wider
        // value tile shares QK/score work across the whole 128-column output;
        // the dispatcher retains the narrower tile when the grid is small.
        // Publish V after score consumption; retain the dtype-specific barrier.
        constexpr bool kPrefetchV = PrefetchV;
        constexpr bool kExtraBarrier = std::is_same_v<DataType, bf16_t> ? false : true;

        const index_t wave=get_warp_id(),tid=threadIdx.x;
        const index_t chunk=blockIdx.x,row0=(wave/2)*32,vh=blockIdx.y,vb=blockIdx.z*(2*VTile)+(wave%2)*VTile;
        const index_t key_col=(wave%2)*32;
        const index_t qh=vh/(a.num_value_heads/a.num_qk_heads),token=chunk*64;
        const index_t lane=get_lane_id(),row=row0+lane%16,group=lane/16;
        __shared__ DataType sm[64*Stride];
        const float gate=a.g[static_cast<long_index_t>(token+lane)*a.num_value_heads+vh];
        const float ep=__builtin_amdgcn_exp2f(gate),en=__builtin_amdgcn_exp2f(-gate);
        thread_buffer<float,MR> erow;
        static_for<0,MR,1>{}([&](auto m){erow(m)=shuffle(ep,row+m*16);});
        const auto hbase=(static_cast<long_index_t>(chunk)*a.num_value_heads+vh)*128*128;
        uint32x4_t prefetched_v[VTile/16];
        if constexpr(kPrefetchV) static_for<0,VTile/16,1>{}([&](auto p){
            const index_t t=p*(1024/VTile)+tid/(VTile/4),v=(tid%(VTile/4))*8;
            const auto off=(static_cast<long_index_t>(token+t)*a.num_value_heads+vh)*128+blockIdx.z*(2*VTile)+v;
            prefetched_v[p]=*reinterpret_cast<const uint32x4_t*>(a.v_new+off);
        });
        // Keep only one K32 panel live, reusing Q for both QK and QH.
        C raw[MR][2],hist[MR][NR];
        static_for<0,MR,1>{}([&](auto m){
            static_for<0,2,1>{}([&](auto n){raw[m][n]=C{0.f};});
            static_for<0,NR,1>{}([&](auto n){hist[m][n]=C{0.f};});
        });
        static_for<0,4,1>{}([&](auto p){
            thread_buffer<DataType,8> q[MR],k[2],h[NR];
            static_for<0,MR,1>{}([&](auto m){
                const auto off=(static_cast<long_index_t>(token+row+m*16)*a.num_qk_heads+qh)*128+p*32+group*8;
                q[m].template set_as<uint32x4_t>(number<0>{},*reinterpret_cast<const uint32x4_t*>(a.q+off));
            });
            // Wave 1 covers query rows 0..31 and key columns 32..63:
            // every score is causally masked, so no K load or QK MMAC is needed.
            if(!CausalPrune || wave!=1)
            static_for<0,2,1>{}([&](auto n){
                const auto off=(static_cast<long_index_t>(token+key_col+n*16+lane%16)*a.num_qk_heads+qh)*128+p*32+group*8;
                k[n].template set_as<uint32x4_t>(number<0>{},*reinterpret_cast<const uint32x4_t*>(a.k+off));
            });
            static_for<0,NR,1>{}([&](auto n){
                const index_t v=vb+n*16+lane%16,kc=p*32+group*8;
                const index_t off=(((v/16)*16+kc/8)*16+v%16)*8;
                h[n].template set_as<uint32x4_t>(number<0>{},*reinterpret_cast<const uint32x4_t*>(a.h+hbase+off));
            });
            static_for<0,MR,1>{}([&](auto m){
                if(!CausalPrune || wave!=1)
                static_for<0,2,1>{}([&](auto n){
                    Mmac{}(raw[m][n],q[m].template get_as<A>()[number<0>{}],k[n].template get_as<A>()[number<0>{}]);
                    Mmac{}(raw[m][n],q[m].template get_as<A>()[number<1>{}],k[n].template get_as<A>()[number<1>{}]);
                });
                static_for<0,NR,1>{}([&](auto n){
                    Mmac{}(hist[m][n],q[m].template get_as<A>()[number<0>{}],h[n].template get_as<A>()[number<0>{}]);
                    Mmac{}(hist[m][n],q[m].template get_as<A>()[number<1>{}],h[n].template get_as<A>()[number<1>{}]);
                });
            });
        });
        thread_buffer<DataType,4> score[MR][4];
        static_for<0,2,1>{}([&](auto n){
            static_for<0,MR,1>{}([&](auto m){
                thread_buffer<float,4> tmp;tmp.template set_as<C>(number<0>{},raw[m][n]);
                static_for<0,4,1>{}([&](auto j){
                    const index_t col=key_col+n*16+group+j*4;
                    const float inv=shuffle(en,col);
                    if constexpr(std::is_same_v<DataType,bf16_t>)
                        tmp(j)=col<=row+m*16 ? tmp[j]*erow[m]*inv : 0.f;
                    else
                        tmp(j)=col<=row+m*16 ? type_convert<float>(gdn_type_convert<DataType>(tmp[j]*erow[m]*inv)) : 0.f;
                });
                static_for<0,4,1>{}([&](auto j){
                    sm[(row+m*16)*Stride+key_col+n*16+group+j*4]=gdn_type_convert<DataType>(tmp[j]);
                });
            });
        });
        __syncthreads();
        static_for<0,MR,1>{}([&](auto m){static_for<0,4,1>{}([&](auto n){
            static_for<0,4,1>{}([&](auto j){score[m][n](j)=sm[(row+m*16)*Stride+n*16+group*4+j];});
        });});
        __syncthreads();
        auto publish_v = [&] {
            static_for<0,VTile/16,1>{}([&](auto p){
                const index_t t=p*(1024/VTile)+tid/(VTile/4),v=(tid%(VTile/4))*8;
                if constexpr(kPrefetchV)
                    *reinterpret_cast<uint32x4_t*>(sm+t*Stride+v)=prefetched_v[p];
                else
                {
                    const auto off=(static_cast<long_index_t>(token+t)*a.num_value_heads+vh)*128+blockIdx.z*(2*VTile)+v;
                    *reinterpret_cast<uint32x4_t*>(sm+t*Stride+v)=*reinterpret_cast<const uint32x4_t*>(a.v_new+off);
                }
            });
        };
        // GDN history contribution is exp(g_row) * (Q @ H). Apply the
        // row gate to the FP32 accumulator, avoiding an extra low-precision
        // round of Q. Score and final-output RNE are retained.

        if constexpr(kExtraBarrier) __syncthreads();
        publish_v();
        __syncthreads();
        static_for<0,NR,1>{}([&](auto n){
            C local[MR];static_for<0,MR,1>{}([&](auto m){local[m]=C{0.f};});
            static_for<0,4,1>{}([&](auto p){
                // First-half queries have identically zero scores for
                // the final 32 keys, so those score*V products can be skipped.
                if(!CausalPrune || row0!=0 || p<2) {
                thread_buffer<DataType,4> v;
                static_for<0,4,1>{}([&](auto j){v(j)=sm[(p*16+group*4+j)*Stride+(wave%2)*VTile+n*16+lane%16];});
                static_for<0,MR,1>{}([&](auto m){Mmac{}(local[m],score[m][p].template get_as<A>()[number<0>{}],v.template get_as<A>()[number<0>{}]);});
                }
            });
            static_for<0,MR,1>{}([&](auto m){
                thread_buffer<float,4> hx,lx;hx.template set_as<C>(number<0>{},hist[m][n]);lx.template set_as<C>(number<0>{},local[m]);
                static_for<0,4,1>{}([&](auto j){hx(j)=(hx[j]*erow[m]+lx[j])*a.scale;});transpose(hx);
                thread_buffer<DataType,4> out;
                static_for<0,4,1>{}([&](auto j){out(j)=gdn_type_convert<DataType>(hx[j]);});
                const auto off=(static_cast<long_index_t>(token+row+m*16)*a.num_value_heads+vh)*128+vb+n*16+group*4;
                buffer_store<8>{}(out,make_wave_buffer_resource(a.o),off*sizeof(DataType),0,0);
            });
        });
    }
};
} // namespace gdn
