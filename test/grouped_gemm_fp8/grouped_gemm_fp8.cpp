// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_xdl_cshuffle_tile_loop.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

using namespace ck;
using namespace ck::tensor_operation::device;
using Row = tensor_layout::gemm::RowMajor;
using Col = tensor_layout::gemm::ColumnMajor;
using Pass = tensor_operation::element_wise::PassThrough;
using Mul = tensor_operation::element_wise::MultiplyMultiply;
template <index_t... I> using S = Sequence<I...>;
template <typename T, GemmSpecialization Spec,
          BlockGemmPipelineVersion Pipeline = BlockGemmPipelineVersion::v1>
using Op = DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<
    Row, Col, Tuple<Row, Col>, Row, T, T, float, float, Tuple<float, float>, bhalf_t,
    Pass, Pass, Mul, Spec, 1, 64, 16, 16, 256, 16/sizeof(T), 16/sizeof(T), 16, 16, 1, 1,
    S<16,4,1>, S<1,0,2>, S<1,0,2>, 2, 16/sizeof(T), 16/sizeof(T), 0,
    S<16,4,1>, S<1,0,2>, S<1,0,2>, 2, 16/sizeof(T), 16/sizeof(T), 0,
    1, 1, S<1,16,1,4>, S<4,4,1>,
    BlockGemmPipelineScheduler::Intrawave, Pipeline, T>;

void check(hipError_t e) { if(e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
struct Buffer {
    void* p = nullptr;
    explicit Buffer(size_t n) { check(hipMalloc(&p, std::max(n, size_t{1}))); }
    ~Buffer() { if(p) (void)hipFree(p); }
    void put(const void* s, size_t n) { check(hipMemcpy(p,s,n,hipMemcpyHostToDevice)); }
    void get(void* d, size_t n) { check(hipMemcpy(d,p,n,hipMemcpyDeviceToHost)); }
};

template <typename T>
__global__ void make_args(GroupedGemmKernelArgument<2>* args, const T* a, const T* b,
                          const float* sa, const float* sb, bhalf_t* out,
                          const int* ms, int g, int cap, int n, int k, int stride)
{
    int i = threadIdx.x;
    if(i < g)
        args[i] = {a+i*cap*stride, b+i*n*stride, {sb+i*n, sa+i*cap},
                   out+i*cap*n, ms[i], n, k, stride, stride, {0,0}, n};
}

float decode_fp8(unsigned char u)
{
    int e = (u >> 3) & 15, m = u & 7;
    float x = e ? std::ldexp(1.0f + m/8.0f, e-7) : std::ldexp(float(m), -9);
    return (u & 128) ? -x : x;
}
float decode_bf16(bhalf_t v)
{
    uint16_t u; std::memcpy(&u, &v, 2);
    uint32_t bits = uint32_t(u) << 16; float f; std::memcpy(&f, &bits, 4); return f;
}
template <typename T> T encode(unsigned char u)
{
    T value;
    if constexpr(sizeof(T)==1) std::memcpy(&value, &u, 1);
    else { float f=decode_fp8(u); uint32_t bits; std::memcpy(&bits,&f,4);
           uint16_t b=bits>>16; std::memcpy(&value,&b,2); }
    return value;
}
template <typename T, GemmSpecialization Spec,
          BlockGemmPipelineVersion Pipeline = BlockGemmPipelineVersion::v1>
bool run_case(int n, int k, std::vector<int> ms, bool padded_stride, bool graph)
{
    constexpr int cap=65;
    int g=ms.size(), stride=k+(padded_stride?16:0);
    std::vector<T> a(g*cap*stride), b(g*n*stride);
    std::vector<float> af(a.size()), bf(b.size()), sa(g*cap), sb(g*n);
    // Include E4M3FN's exponent-15 finite values; FNUZ interpretation would fail.
    const unsigned char values[]={0x00,0x30,0xb8,0x3c,0x40,0xc4,0x78,0xfe};
    for(size_t i=0;i<a.size();++i) {auto u=values[(i*7+i/stride)%8]; a[i]=encode<T>(u); af[i]=decode_fp8(u);}
    for(size_t i=0;i<b.size();++i) {auto u=values[(i*3+i/stride+2)%8]; b[i]=encode<T>(u); bf[i]=decode_fp8(u);}
    for(size_t i=0;i<sa.size();++i) sa[i]=(1+i%7)/1024.f;
    for(size_t i=0;i<sb.size();++i) sb[i]=(1+i%11)/1024.f;
    std::vector<bhalf_t> out(g*cap*n);
    Buffer da(a.size()*sizeof(T)),db(b.size()*sizeof(T)),dsa(sa.size()*4),dsb(sb.size()*4),
           dout(out.size()*2), dms(g*4), dargs(g*sizeof(GroupedGemmKernelArgument<2>));
    da.put(a.data(),a.size()*sizeof(T)); db.put(b.data(),b.size()*sizeof(T));
    dsa.put(sa.data(),sa.size()*4); dsb.put(sb.data(),sb.size()*4); dms.put(ms.data(),g*4);
    check(hipMemset(dout.p,0x7f,out.size()*2));
    Op<T,Spec,Pipeline> op;
    std::vector<const void*> pa,pb;
    std::vector<void*> pe;
    std::vector<std::array<const void*,2>> pd;
    // Same all-zero placeholders used by the MSLK Tensor API.
    std::vector<GemmDesc> desc(g, GemmDesc{0,0,0,0,0,0,{0,0}});
    auto arg=op.MakeArgument(pa,pb,pd,pe,desc,Pass{},Pass{},Mul{});
    if(!op.IsSupportedArgument(arg)) throw std::runtime_error("GPU-argument placeholders rejected");
    // Check actual empty descriptors, and rejection of nonempty misaligned shapes.
    std::vector<GemmDesc> checks{{0, n, 17, 0, 0, 0, {0,0}},
                               {1, n, k, stride, stride, n, {0,0}}};
    auto checked=op.MakeArgument(pa,pb,pd,pe,checks,Pass{},Pass{},Mul{});
    if(!op.IsSupportedArgument(checked)) throw std::runtime_error("empty expert rejected");
    checks[1].N_=3;
    if(op.IsSupportedArgument(checked)) throw std::runtime_error("invalid N accepted");
    op.SetDeviceKernelArgs(arg,dargs.p);
    hipStream_t stream; check(hipStreamCreate(&stream));
    auto launch=[&]{
        make_args<<<1,64,0,stream>>>(static_cast<GroupedGemmKernelArgument<2>*>(dargs.p),
            static_cast<T*>(da.p),static_cast<T*>(db.p),static_cast<float*>(dsa.p),
            static_cast<float*>(dsb.p),static_cast<bhalf_t*>(dout.p),static_cast<int*>(dms.p),
            g,cap,n,k,stride);
        op.MakeInvoker().Run(arg,StreamConfig{stream,false});
        check(hipGetLastError());
    };
    launch(); check(hipStreamSynchronize(stream)); // Prime occupancy queries before capture.
    hipGraph_t gr=nullptr; hipGraphExec_t ex=nullptr;
    if(graph) {
        check(hipStreamBeginCapture(stream,hipStreamCaptureModeGlobal)); launch();
        check(hipStreamEndCapture(stream,&gr)); check(hipGraphInstantiate(&ex,gr,nullptr,nullptr,0));
    }
    bool pass=true; double maxerr=0;
    for(int replay=0;replay<(graph?3:1);++replay) {
        if(replay) {std::rotate(ms.begin(),ms.begin()+1,ms.end()); dms.put(ms.data(),g*4);}
        check(hipMemset(dout.p,0x7f,out.size()*2));
        if(graph) check(hipGraphLaunch(ex,stream)); else launch();
        check(hipStreamSynchronize(stream)); dout.get(out.data(),out.size()*2);
        for(int i=0;i<g;++i) for(int m=0;m<cap;++m) for(int j=0;j<n;++j) {
            auto idx=(i*cap+m)*n+j;
            if(m>=ms[i]) {uint16_t v; std::memcpy(&v,&out[idx],2); pass &= v==0x7f7f; continue;}
            float ref=0;
            for(int p=0;p<k;++p) ref+=af[(i*cap+m)*stride+p]*bf[(i*n+j)*stride+p];
            ref=ref*sb[i*n+j]*sa[i*cap+m];
            float got=decode_bf16(out[idx]);
            double err=std::abs(double(got)-ref); maxerr=std::max(maxerr,err);
            if(!std::isfinite(got)||err>0.008*std::abs(ref)+0.0001) pass=false;
        }
    }
    if(ex) check(hipGraphExecDestroy(ex)); if(gr) check(hipGraphDestroy(gr));
    check(hipStreamDestroy(stream));
    std::cout<<(pass?"PASS":"FAIL")<<" dtype="<<(sizeof(T)==1?"fp8":"bf16")
             <<" N="<<n<<" K="<<k<<" pad="<<(Spec==GemmSpecialization::KPadding)
             <<" stride="<<stride<<" graph="<<graph<<" max_abs="<<maxerr<<std::endl;
    return pass;
}
int main()
{
    try {
        bool ok=true;
        for(int n:{16,40,128}) {
            ok &= run_case<f8_t,GemmSpecialization::Default>(n,256,{0,1,17,0,65},false,false);
            ok &= run_case<f8_t,GemmSpecialization::KPadding>(n,272,{0,31,0,33,1},true,true);
        }
        ok &= run_case<f8_t,GemmSpecialization::Default>(32,512,{0,0,0,0,0},false,true);
        ok &= run_case<f8_t,GemmSpecialization::KPadding>(32,16,{1,0,17},false,false);
        ok &= run_case<bhalf_t,GemmSpecialization::Default>(32,256,{0,1,17},false,false);
        ok &= run_case<f8_t,GemmSpecialization::Default,BlockGemmPipelineVersion::v2>(
            40,1024,{0,1,33,0,65},false,false);
        ok &= run_case<f8_t,GemmSpecialization::Default,BlockGemmPipelineVersion::v3>(
            40,1024,{0,1,33,0,65},false,false);
        return ok?0:1;
    } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 2;}
}
