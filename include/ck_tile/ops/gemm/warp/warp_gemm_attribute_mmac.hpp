// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_mmac_impl.hpp"

namespace ck_tile {

// [HCU移植] MMAC（海光 HCU 原生矩阵指令）warp GEMM 的基类。对应上游 github CK 的
// MFMA 版本（WarpGemmAttributeMfmaIterateK）。注意：本基类的 CWarpDstrEncoding 必须
// 维持与 HCU GEMM example（03_gemm/19_grouped_gemm）一致的 C 寄存器布局，FMHA 所需的
// 裸硬件布局/转置布局一律走下面派生的 ...IterateKCRaw / ...IterateKLitLts，切勿改基类。
template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat,
          index_t NRepeat,
          index_t MInterleave,
          index_t NInterleave,
          index_t kKIter>
struct WarpGemmAttributeMmacIterateK
{
    static_assert(kKIter > 0, "wrong!");

    using Impl = remove_cvref_t<WarpGemmAttributeMmacImpl_>;

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    // ext_vector_t<datatype, ext_vector_t>, 2d array 2 x vec_a
    using ABufType = thread_buffer<typename Impl::AVecType, kKIter * MRepeat * MInterleave>;
    using BBufType = thread_buffer<typename Impl::BVecType, kKIter * NRepeat * NInterleave>;
    using CBufType =
        thread_buffer<typename Impl::CVecType, MRepeat * NRepeat * MInterleave * NInterleave>;

    static constexpr index_t kM = Impl::kM * MRepeat * MInterleave;
    static constexpr index_t kN = Impl::kN * NRepeat * NInterleave;
    static constexpr index_t kK = Impl::kK * kKIter;
    static constexpr index_t kKPerThread = Impl::kABKPerLane * kKIter;

    using AWarpDstrEncoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<MRepeat, Impl::kAMLane, MInterleave>,
                                         sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<0, 1>>,
                                   sequence<1, 1, 2>,
                                   sequence<0, 2, 1>>;

    using BWarpDstrEncoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<NRepeat, Impl::kBNLane, NInterleave>,
                                         sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<0, 1>>,
                                   sequence<1, 1, 2>,
                                   sequence<0, 2, 1>>;

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCMLane, MInterleave, Impl::kCMPerLane>,
              sequence<NRepeat, Impl::kCNLane, NInterleave, Impl::kCN0PerLane, Impl::kCN1PerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 1>>,
        sequence<1, 2, 1, 1, 2, 2, 2>, // <MRepeat, NRepeat, MInterleave, MPerLane, N0PerLane,
                                       // NmmacInterleave, N1PerLane>
        sequence<0, 0, 2, 3, 3, 2, 4>>;

    using CWarpOutputDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCMLane, MInterleave, Impl::kCMPerLane>,
              sequence<NRepeat, Impl::kCN0PerLane, Impl::kCNLane, NInterleave * Impl::kCN1PerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<2, 1>>,
        sequence<1, 2, 1, 1, 2, 2>, // <MRepeat, NRepeat, MInterleave, MPerLane, N0PerLane,
                                    // NmmacInterleave, N1PerLane>
        sequence<0, 0, 2, 3, 1, 3>>;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CBufType& c_buf, const ABufType& a_buf, const BBufType& b_buf) const
    {
        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            // 1 mmac issue
                            Impl{}(c_buf(iMRepeat * NRepeat * MInterleave * NInterleave +
                                         iNRepeat * MInterleave * NInterleave +
                                         iMInterleave * NInterleave + iNInterleave),
                                   a_buf[iMRepeat * MInterleave * kKIter + iMInterleave * kKIter +
                                         iKIter],
                                   b_buf[iNRepeat * NInterleave * kKIter + iNInterleave * kKIter +
                                         iKIter]);
                        });
                    });
                });
            });
        });
    }

    // c_vec = a_vec * b_vec
    // AVecType = ext_vector_t<ADataType, AWarpTensor::get_thread_buffer_size()>;
    CK_TILE_DEVICE CBufType operator()(const ABufType& a_buf, const BBufType& b_buf) const
    {
        CBufType c_buf;

        // c += a * b
        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            // 1 mmac issue
                            Impl{}(c_buf(iMRepeat * NRepeat * MInterleave * NInterleave +
                                         iNRepeat * MInterleave * NInterleave +
                                         iMInterleave * NInterleave + iNInterleave),
                                   a_buf[iMRepeat * MInterleave * kKIter + iMInterleave * kKIter +
                                         iKIter],
                                   b_buf[iNRepeat * NInterleave * kKIter + iNInterleave * kKIter +
                                         iKIter]);
                        });
                    });
                });
            });
        });

        return c_buf;
    }

    template <typename CWarpTensor>
    CK_TILE_DEVICE auto MakeCOutputLayout(const CWarpTensor& c_warp_tensor) const
    {
        constexpr auto c_warp_output_distribution =
            make_static_tile_distribution(CWarpOutputDstrEncoding{});
        auto c_warp_output_tensor =
            make_static_distributed_tensor<CDataType>(c_warp_output_distribution);

        static_for<0, MRepeat, 1>{}([&](auto iMR) {
            static_for<0, NRepeat, 1>{}([&](auto iNR) {
                static_for<0, MInterleave, 1>{}([&](auto iMI) {
                    static_for<0, NInterleave, 1>{}([&](auto iNI) {
                        static_for<0, Impl::kCN0PerLane, 1>{}([&](auto iCN0) {
                            c_warp_output_tensor.set_y_sliced_thread_data(
                                sequence<iMR, iNR, iMI, 0, iCN0, iNI>{},      // output_orgin
                                sequence<1, 1, 1, 1, 1, Impl::kCN1PerLane>{}, // output_set_length
                                c_warp_tensor.get_y_sliced_thread_data( // get calculate tensor and
                                                                        // change to output layout
                                    sequence<iMR, iNR, iMI, 0, iNI, iCN0, 0>{},
                                    sequence<1, 1, 1, 1, 1, 1, Impl::kCN1PerLane>{}));
                        });
                    });
                });
            });
        });

        return c_warp_output_tensor;
    }
};

// Vendor-style MMAC traversal for wide M-repeat tiles. The tensor
// distributions and accumulator numbering are identical to IterateK; only
// the compile-time issue order changes so one B vector is reused across all M
// repeats before advancing N.
template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat_,
          index_t NRepeat_,
          index_t MInterleave_,
          index_t NInterleave_,
          index_t kKIter_>
struct WarpGemmAttributeMmacIterateKNOuter
    : public WarpGemmAttributeMmacIterateK<
          WarpGemmAttributeMmacImpl_,
          MRepeat_,
          NRepeat_,
          MInterleave_,
          NInterleave_,
          kKIter_>
{
    using Base = WarpGemmAttributeMmacIterateK<
        WarpGemmAttributeMmacImpl_,
        MRepeat_,
        NRepeat_,
        MInterleave_,
        NInterleave_,
        kKIter_>;
    using Impl = typename Base::Impl;
    using ABufType = typename Base::ABufType;
    using BBufType = typename Base::BBufType;
    using CBufType = typename Base::CBufType;
    using Base::operator();

    CK_TILE_DEVICE void
    operator()(CBufType& c_buf,
               const ABufType& a_buf,
               const BBufType& b_buf) const
    {
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_C_ASM)
        static_assert(MRepeat_ == 8 && NRepeat_ == 4 &&
                          MInterleave_ == 1 && NInterleave_ == 1 &&
                          kKIter_ == 1,
                      "fixed-C MMAC asm is specialized for MIWT8_4 K16");
        static_assert(std::is_same_v<typename Impl::ADataType, bf16_t> &&
                          std::is_same_v<typename Impl::BDataType, bf16_t>,
                      "fixed-C MMAC asm is BF16-only");
        // Bind logical C(m,n) in BLAS' N-major issue order to consecutive
        // physical tuples v0:v127. Keeping all 32 MMACs in one asm region
        // prevents both accumulator reallocation and instruction reordering.
        asm volatile(
            "v_mmac_f32_16x16x16_bf16 %0, %36, %32, %0\n\t"
            "s_setprio 1\n\t"
            "v_mmac_f32_16x16x16_bf16 %1, %37, %32, %1\n\t"
            "v_mmac_f32_16x16x16_bf16 %2, %38, %32, %2\n\t"
            "v_mmac_f32_16x16x16_bf16 %3, %39, %32, %3\n\t"
            "v_mmac_f32_16x16x16_bf16 %4, %40, %32, %4\n\t"
            "v_mmac_f32_16x16x16_bf16 %5, %41, %32, %5\n\t"
            "v_mmac_f32_16x16x16_bf16 %6, %42, %32, %6\n\t"
            "v_mmac_f32_16x16x16_bf16 %7, %43, %32, %7\n\t"
            "v_mmac_f32_16x16x16_bf16 %8, %36, %33, %8\n\t"
            "v_mmac_f32_16x16x16_bf16 %9, %37, %33, %9\n\t"
            "v_mmac_f32_16x16x16_bf16 %10, %38, %33, %10\n\t"
            "v_mmac_f32_16x16x16_bf16 %11, %39, %33, %11\n\t"
            "v_mmac_f32_16x16x16_bf16 %12, %40, %33, %12\n\t"
            "v_mmac_f32_16x16x16_bf16 %13, %41, %33, %13\n\t"
            "v_mmac_f32_16x16x16_bf16 %14, %42, %33, %14\n\t"
            "v_mmac_f32_16x16x16_bf16 %15, %43, %33, %15\n\t"
            "v_mmac_f32_16x16x16_bf16 %16, %36, %34, %16\n\t"
            "v_mmac_f32_16x16x16_bf16 %17, %37, %34, %17\n\t"
            "v_mmac_f32_16x16x16_bf16 %18, %38, %34, %18\n\t"
            "v_mmac_f32_16x16x16_bf16 %19, %39, %34, %19\n\t"
            "v_mmac_f32_16x16x16_bf16 %20, %40, %34, %20\n\t"
            "v_mmac_f32_16x16x16_bf16 %21, %41, %34, %21\n\t"
            "v_mmac_f32_16x16x16_bf16 %22, %42, %34, %22\n\t"
            "v_mmac_f32_16x16x16_bf16 %23, %43, %34, %23\n\t"
            "v_mmac_f32_16x16x16_bf16 %24, %36, %35, %24\n\t"
            "v_mmac_f32_16x16x16_bf16 %25, %37, %35, %25\n\t"
            "v_mmac_f32_16x16x16_bf16 %26, %38, %35, %26\n\t"
            "v_mmac_f32_16x16x16_bf16 %27, %39, %35, %27\n\t"
            "v_mmac_f32_16x16x16_bf16 %28, %40, %35, %28\n\t"
            "v_mmac_f32_16x16x16_bf16 %29, %41, %35, %29\n\t"
            "v_mmac_f32_16x16x16_bf16 %30, %42, %35, %30\n\t"
            "v_mmac_f32_16x16x16_bf16 %31, %43, %35, %31\n\t"
            "s_setprio 0"
            : "+&{v[0:3]}"(c_buf(number<0>{})),
              "+&{v[4:7]}"(c_buf(number<4>{})),
              "+&{v[8:11]}"(c_buf(number<8>{})),
              "+&{v[12:15]}"(c_buf(number<12>{})),
              "+&{v[16:19]}"(c_buf(number<16>{})),
              "+&{v[20:23]}"(c_buf(number<20>{})),
              "+&{v[24:27]}"(c_buf(number<24>{})),
              "+&{v[28:31]}"(c_buf(number<28>{})),
              "+&{v[32:35]}"(c_buf(number<1>{})),
              "+&{v[36:39]}"(c_buf(number<5>{})),
              "+&{v[40:43]}"(c_buf(number<9>{})),
              "+&{v[44:47]}"(c_buf(number<13>{})),
              "+&{v[48:51]}"(c_buf(number<17>{})),
              "+&{v[52:55]}"(c_buf(number<21>{})),
              "+&{v[56:59]}"(c_buf(number<25>{})),
              "+&{v[60:63]}"(c_buf(number<29>{})),
              "+&{v[64:67]}"(c_buf(number<2>{})),
              "+&{v[68:71]}"(c_buf(number<6>{})),
              "+&{v[72:75]}"(c_buf(number<10>{})),
              "+&{v[76:79]}"(c_buf(number<14>{})),
              "+&{v[80:83]}"(c_buf(number<18>{})),
              "+&{v[84:87]}"(c_buf(number<22>{})),
              "+&{v[88:91]}"(c_buf(number<26>{})),
              "+&{v[92:95]}"(c_buf(number<30>{})),
              "+&{v[96:99]}"(c_buf(number<3>{})),
              "+&{v[100:103]}"(c_buf(number<7>{})),
              "+&{v[104:107]}"(c_buf(number<11>{})),
              "+&{v[108:111]}"(c_buf(number<15>{})),
              "+&{v[112:115]}"(c_buf(number<19>{})),
              "+&{v[116:119]}"(c_buf(number<23>{})),
              "+&{v[120:123]}"(c_buf(number<27>{})),
              "+&{v[124:127]}"(c_buf(number<31>{}))
            : "v"(b_buf[number<0>{}]),
              "v"(b_buf[number<1>{}]),
              "v"(b_buf[number<2>{}]),
              "v"(b_buf[number<3>{}]),
              "v"(a_buf[number<0>{}]),
              "v"(a_buf[number<1>{}]),
              "v"(a_buf[number<2>{}]),
              "v"(a_buf[number<3>{}]),
              "v"(a_buf[number<4>{}]),
              "v"(a_buf[number<5>{}]),
              "v"(a_buf[number<6>{}]),
              "v"(a_buf[number<7>{}])
            : "memory");
#elif defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_RELAXED_FULL_MMAC_ASM)
        static_assert(MRepeat_ == 8 && NRepeat_ == 4 &&
                          MInterleave_ == 1 && NInterleave_ == 1 &&
                          kKIter_ == 1,
                      "relaxed full-MMAC asm is specialized for MIWT8_4 K16");
        static_assert(std::is_same_v<typename Impl::ADataType, bf16_t> &&
                          std::is_same_v<typename Impl::BDataType, bf16_t>,
                      "relaxed full-MMAC asm is BF16-only");
        // Keep the entire 32-MMAC issue region at one compiler dependency
        // boundary, but let register allocation choose the physical C/A/B
        // ranges. Unlike BLAS_FIXED_C_ASM this does not pin v0:v175, and the
        // lack of a memory clobber lets the waitcnt pass preserve the newer
        // look-ahead LDS operations.
        asm volatile(
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FUSED_SYNC_AB_MMAC)
            "s_waitcnt lgkmcnt(8)\n\t"
            "s_waitcnt vmcnt(4)\n\t"
            "s_barrier\n\t"
#elif defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FUSED_SYNC_B_MMAC)
            "s_waitcnt lgkmcnt(4)\n\t"
            "s_waitcnt vmcnt(4)\n\t"
            "s_barrier\n\t"
#endif
            "v_mmac_f32_16x16x16_bf16 %0, %36, %32, %0\n\t"
            "s_setprio 1\n\t"
            "v_mmac_f32_16x16x16_bf16 %1, %37, %32, %1\n\t"
            "v_mmac_f32_16x16x16_bf16 %2, %38, %32, %2\n\t"
            "v_mmac_f32_16x16x16_bf16 %3, %39, %32, %3\n\t"
            "v_mmac_f32_16x16x16_bf16 %4, %40, %32, %4\n\t"
            "v_mmac_f32_16x16x16_bf16 %5, %41, %32, %5\n\t"
            "v_mmac_f32_16x16x16_bf16 %6, %42, %32, %6\n\t"
            "v_mmac_f32_16x16x16_bf16 %7, %43, %32, %7\n\t"
            "v_mmac_f32_16x16x16_bf16 %8, %36, %33, %8\n\t"
            "v_mmac_f32_16x16x16_bf16 %9, %37, %33, %9\n\t"
            "v_mmac_f32_16x16x16_bf16 %10, %38, %33, %10\n\t"
            "v_mmac_f32_16x16x16_bf16 %11, %39, %33, %11\n\t"
            "v_mmac_f32_16x16x16_bf16 %12, %40, %33, %12\n\t"
            "v_mmac_f32_16x16x16_bf16 %13, %41, %33, %13\n\t"
            "v_mmac_f32_16x16x16_bf16 %14, %42, %33, %14\n\t"
            "v_mmac_f32_16x16x16_bf16 %15, %43, %33, %15\n\t"
            "v_mmac_f32_16x16x16_bf16 %16, %36, %34, %16\n\t"
            "v_mmac_f32_16x16x16_bf16 %17, %37, %34, %17\n\t"
            "v_mmac_f32_16x16x16_bf16 %18, %38, %34, %18\n\t"
            "v_mmac_f32_16x16x16_bf16 %19, %39, %34, %19\n\t"
            "v_mmac_f32_16x16x16_bf16 %20, %40, %34, %20\n\t"
            "v_mmac_f32_16x16x16_bf16 %21, %41, %34, %21\n\t"
            "v_mmac_f32_16x16x16_bf16 %22, %42, %34, %22\n\t"
            "v_mmac_f32_16x16x16_bf16 %23, %43, %34, %23\n\t"
            "v_mmac_f32_16x16x16_bf16 %24, %36, %35, %24\n\t"
            "v_mmac_f32_16x16x16_bf16 %25, %37, %35, %25\n\t"
            "v_mmac_f32_16x16x16_bf16 %26, %38, %35, %26\n\t"
            "v_mmac_f32_16x16x16_bf16 %27, %39, %35, %27\n\t"
            "v_mmac_f32_16x16x16_bf16 %28, %40, %35, %28\n\t"
            "v_mmac_f32_16x16x16_bf16 %29, %41, %35, %29\n\t"
            "v_mmac_f32_16x16x16_bf16 %30, %42, %35, %30\n\t"
            "v_mmac_f32_16x16x16_bf16 %31, %43, %35, %31\n\t"
            "s_setprio 0"
            : "+&v"(c_buf(number<0>{})),
              "+&v"(c_buf(number<4>{})),
              "+&v"(c_buf(number<8>{})),
              "+&v"(c_buf(number<12>{})),
              "+&v"(c_buf(number<16>{})),
              "+&v"(c_buf(number<20>{})),
              "+&v"(c_buf(number<24>{})),
              "+&v"(c_buf(number<28>{})),
              "+&v"(c_buf(number<1>{})),
              "+&v"(c_buf(number<5>{})),
              "+&v"(c_buf(number<9>{})),
              "+&v"(c_buf(number<13>{})),
              "+&v"(c_buf(number<17>{})),
              "+&v"(c_buf(number<21>{})),
              "+&v"(c_buf(number<25>{})),
              "+&v"(c_buf(number<29>{})),
              "+&v"(c_buf(number<2>{})),
              "+&v"(c_buf(number<6>{})),
              "+&v"(c_buf(number<10>{})),
              "+&v"(c_buf(number<14>{})),
              "+&v"(c_buf(number<18>{})),
              "+&v"(c_buf(number<22>{})),
              "+&v"(c_buf(number<26>{})),
              "+&v"(c_buf(number<30>{})),
              "+&v"(c_buf(number<3>{})),
              "+&v"(c_buf(number<7>{})),
              "+&v"(c_buf(number<11>{})),
              "+&v"(c_buf(number<15>{})),
              "+&v"(c_buf(number<19>{})),
              "+&v"(c_buf(number<23>{})),
              "+&v"(c_buf(number<27>{})),
              "+&v"(c_buf(number<31>{}))
            : "v"(b_buf[number<0>{}]),
              "v"(b_buf[number<1>{}]),
              "v"(b_buf[number<2>{}]),
              "v"(b_buf[number<3>{}]),
              "v"(a_buf[number<0>{}]),
              "v"(a_buf[number<1>{}]),
              "v"(a_buf[number<2>{}]),
              "v"(a_buf[number<3>{}]),
              "v"(a_buf[number<4>{}]),
              "v"(a_buf[number<5>{}]),
              "v"(a_buf[number<6>{}]),
              "v"(a_buf[number<7>{}])
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FUSED_SYNC_AB_MMAC) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FUSED_SYNC_B_MMAC)
            : "memory"
#endif
        );
#elif defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_AB_FENCE_ALL)
        static_assert(MRepeat_ == 8 && NRepeat_ == 4 &&
                          MInterleave_ == 1 && NInterleave_ == 1 &&
                          kKIter_ == 1,
                      "raw A/B operand fence is specialized for MIWT8_4 K16");
        auto a_ready = a_buf;
        auto b_ready = b_buf;
        // Present every current-stage LDS result to the waitcnt pass at one
        // instruction boundary. This should collapse the progressive
        // lgkmcnt(7/6/5/4) waits otherwise emitted at individual MMAC uses.
        asm volatile(""
                     : "+v"(a_ready[number<0>{}]),
                       "+v"(a_ready[number<1>{}]),
                       "+v"(a_ready[number<2>{}]),
                       "+v"(a_ready[number<3>{}]),
                       "+v"(a_ready[number<4>{}]),
                       "+v"(a_ready[number<5>{}]),
                       "+v"(a_ready[number<6>{}]),
                       "+v"(a_ready[number<7>{}]),
                       "+v"(b_ready[number<0>{}]),
                       "+v"(b_ready[number<1>{}]),
                       "+v"(b_ready[number<2>{}]),
                       "+v"(b_ready[number<3>{}])
                     :
                     : "memory");
        static_for<0, NRepeat_, 1>{}([&](auto iNRepeat) {
            static_for<0, MRepeat_, 1>{}([&](auto iMRepeat) {
                constexpr index_t c_idx =
                    decltype(iMRepeat)::value * NRepeat_ +
                    decltype(iNRepeat)::value;
                constexpr index_t a_idx = decltype(iMRepeat)::value;
                constexpr index_t b_idx = decltype(iNRepeat)::value;
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_AB_INLINE_MMAC)
                asm volatile(
                    "v_mmac_f32_16x16x16_bf16 %0, %1, %2, %0"
                    : "+v"(c_buf(c_idx))
                    : "v"(a_ready[a_idx]), "v"(b_ready[b_idx]));
#else
                Impl{}(
                    c_buf(c_idx), a_ready[a_idx], b_ready[b_idx]);
#endif
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO)
                if constexpr(decltype(iNRepeat)::value == 0 &&
                             decltype(iMRepeat)::value == 0)
                {
                    __builtin_amdgcn_s_setprio(1);
                }
                if constexpr(decltype(iNRepeat)::value == NRepeat_ - 1 &&
                             decltype(iMRepeat)::value == MRepeat_ - 1)
                {
                    __builtin_amdgcn_s_setprio(0);
                }
#endif
            });
        });
#elif defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_JIT_PERMUTE)
        static_assert(NInterleave_ == 1 && MInterleave_ == 1 &&
                          kKIter_ == 1,
                      "JIT B64 permutation is specialized for MIWT8_4 K16");
        static_for<0, NRepeat_, 1>{}([&](auto iNRepeat) {
            auto b_operand = b_buf[iNRepeat];
            const auto b1  = b_operand[1];
            b_operand[1]   = b_operand[2];
            b_operand[2]   = b1;
            // Force each two-permute producer to remain adjacent to the eight
            // MMACs that consume it instead of hoisting all four B producers
            // ahead of the complete 32-MMAC interval.
            asm volatile("" : "+v"(b_operand) : : "memory");

            static_for<0, MRepeat_, 1>{}([&](auto iMRepeat) {
                constexpr index_t c_idx =
                    decltype(iMRepeat)::value * NRepeat_ +
                    decltype(iNRepeat)::value;
                constexpr index_t a_idx =
                    decltype(iMRepeat)::value;
                Impl{}(c_buf(c_idx), a_buf[a_idx], b_operand);
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO)
                if constexpr(decltype(iNRepeat)::value == 0 &&
                             decltype(iMRepeat)::value == 0)
                {
                    __builtin_amdgcn_s_setprio(1);
                }
                if constexpr(decltype(iNRepeat)::value ==
                                 NRepeat_ - 1 &&
                             decltype(iMRepeat)::value ==
                                 MRepeat_ - 1)
                {
                    __builtin_amdgcn_s_setprio(0);
                }
#endif
            });
        });
#else
        static_for<0, NRepeat_, 1>{}([&](auto iNRepeat) {
            static_for<0, MRepeat_, 1>{}([&](auto iMRepeat) {
                static_for<0, NInterleave_, 1>{}([&](auto iNInterleave) {
                    static_for<0, MInterleave_, 1>{}([&](auto iMInterleave) {
                        static_for<0, kKIter_, 1>{}([&](auto iKIter) {
                            constexpr index_t c_idx =
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_RAW_C_ORDER)
                                (decltype(iNRepeat)::value * NInterleave_ +
                                 decltype(iNInterleave)::value) *
                                    MRepeat_ * MInterleave_ +
                                decltype(iMRepeat)::value * MInterleave_ +
                                decltype(iMInterleave)::value;
#else
                                decltype(iMRepeat)::value * NRepeat_ *
                                    MInterleave_ * NInterleave_ +
                                decltype(iNRepeat)::value * MInterleave_ *
                                    NInterleave_ +
                                decltype(iMInterleave)::value *
                                    NInterleave_ +
                                decltype(iNInterleave)::value;
#endif
                            constexpr index_t a_idx =
                                decltype(iMRepeat)::value * MInterleave_ *
                                    kKIter_ +
                                decltype(iMInterleave)::value * kKIter_ +
                                decltype(iKIter)::value;
                            constexpr index_t b_idx =
                                decltype(iNRepeat)::value * NInterleave_ *
                                    kKIter_ +
                                decltype(iNInterleave)::value * kKIter_ +
                                decltype(iKIter)::value;
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_AB_INLINE_MMAC)
                            static_assert(
                                std::is_same_v<typename Impl::ADataType,
                                               bf16_t> &&
                                    std::is_same_v<typename Impl::BDataType,
                                                   bf16_t>,
                                "raw A/B inline MMAC gate is BF16-only");
                            asm volatile(
                                "v_mmac_f32_16x16x16_bf16 %0, %1, %2, %0"
                                : "+v"(c_buf(c_idx))
                                : "v"(a_buf[a_idx]), "v"(b_buf[b_idx]));
#else
                            Impl{}(
                                c_buf(c_idx), a_buf[a_idx], b_buf[b_idx]);
#endif
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO)
                            if constexpr(decltype(iNRepeat)::value == 0 &&
                                         decltype(iMRepeat)::value == 0 &&
                                         decltype(iNInterleave)::value == 0 &&
                                         decltype(iMInterleave)::value == 0 &&
                                         decltype(iKIter)::value == 0)
                            {
                                // Match the selected hipBLASLt solution:
                                // raise priority after its first MMAC so the
                                // remaining dependent accumulator chain is
                                // not interrupted by operand-loading waves.
                                __builtin_amdgcn_s_setprio(1);
                            }
                            if constexpr(
                                decltype(iNRepeat)::value == NRepeat_ - 1 &&
                                decltype(iMRepeat)::value == MRepeat_ - 1 &&
                                decltype(iNInterleave)::value ==
                                    NInterleave_ - 1 &&
                                decltype(iMInterleave)::value ==
                                    MInterleave_ - 1 &&
                                decltype(iKIter)::value == kKIter_ - 1)
                            {
                                __builtin_amdgcn_s_setprio(0);
                            }
#endif
                        });
                    });
                });
            });
        });
#endif
    }
};

// [HCU移植] 本结构体为 FMHA 移植新增（上游 github CK 没有）。
// basic MMAC（gfx936/gfx928/gfx92a，非 lit_lts）的 QK/PV C 矩阵在寄存器中的“裸硬件”
// 布局为：lane = m + 16*(n%4), reg = n/4，即 col = c*16 + r*4 + lane/16。FMHA 直接在
// 寄存器里消费该 C tile，必须用这套编码；而 GEMM example 走基类编码。两者不可合并。
// ------------------------------------------------------------------
// FMHA fwd (gfx936/gfx928/gfx92a basic MMAC) C register layout.
//
// The base WarpGemmAttributeMmacIterateK::CWarpDstrEncoding maps the MMAC C registers
// the way the standalone GEMM epilogue (basic_gemm / grouped_gemm) expects, and must NOT
// be changed or those examples break (compile + fp16/bf16 precision). FMHA instead
// consumes the QK / PV C tile directly in-register and needs the *raw* hardware layout:
//   lane = m + 16 * (n % 4), reg = n / 4   (i.e. col = c*16 + r*4 + lane/16).
// This struct provides only that CWarpDstrEncoding override; everything else is inherited
// from the base. Reference it via the dedicated WarpGemmMmac*_FMHA aliases.
template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat_,
          index_t NRepeat_,
          index_t MInterleave_,
          index_t NInterleave_,
          index_t kKIter>
struct WarpGemmAttributeMmacIterateKCRaw
    : public WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImpl_,
                                           MRepeat_,
                                           NRepeat_,
                                           MInterleave_,
                                           NInterleave_,
                                           kKIter>
{
    using Base = WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImpl_,
                                               MRepeat_,
                                               NRepeat_,
                                               MInterleave_,
                                               NInterleave_,
                                               kKIter>;

    using Impl = typename Base::Impl;

    static constexpr index_t MRepeat     = MRepeat_;
    static constexpr index_t NRepeat     = NRepeat_;
    static constexpr index_t MInterleave = MInterleave_;
    static constexpr index_t NInterleave = NInterleave_;

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCMLane, MInterleave, Impl::kCMPerLane>,
              sequence<NRepeat, Impl::kCN0PerLane, Impl::kCNLane, NInterleave * Impl::kCN1PerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<2, 1>>,
        sequence<1, 2, 1, 1, 2, 2>,
        sequence<0, 0, 2, 3, 1, 3>>;
};

template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat_,
          index_t NRepeat_,
          index_t MInterleave_,
          index_t NInterleave_,
          index_t kKIter>
struct WarpGemmAttributeMmacIterateKTransC
    : public WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImpl_,
                                           MRepeat_,
                                           NRepeat_,
                                           MInterleave_,
                                           NInterleave_,
                                           kKIter>
{
    using Base = WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImpl_,
                                               MRepeat_,
                                               NRepeat_,
                                               MInterleave_,
                                               NInterleave_,
                                               kKIter>;

    using Impl = typename Base::Impl;

    using ADataType = typename Base::ADataType;
    using BDataType = typename Base::BDataType;
    using CDataType = typename Base::CDataType;

    using ABufType = typename Base::ABufType;
    using BBufType = typename Base::BBufType;
    using CBufType = typename Base::CBufType;

    static constexpr index_t kM          = Base::kM;
    static constexpr index_t kN          = Base::kN;
    static constexpr index_t kK          = Base::kK;
    static constexpr index_t MRepeat     = MRepeat_;
    static constexpr index_t NRepeat     = NRepeat_;
    static constexpr index_t MInterleave = MInterleave_;
    static constexpr index_t NInterleave = NInterleave_;

    using AWarpDstrEncoding = typename Base::AWarpDstrEncoding;

    using BWarpDstrEncoding = typename Base::BWarpDstrEncoding;

    // clang-format off
    // lane cluster: {Impl::kCMLane, Impl::kCNLane}
    // lane slice: {MRepeat,NRepeat, MInterleave, NInterleave, Impl::kCNPerLane, Impl::kCM0PerLane, Impl::kCM1PerLane}
    // clang-format on
    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane, MInterleave>,
              sequence<NRepeat, Impl::kCNLane, Impl::kCNPerLane, NInterleave>>,
        tuple<sequence<1, 2>>,
        tuple<sequence<2, 1>>,
        sequence<1, 2, 1, 2, 2, 1, 1>,
        sequence<0, 0, 4, 3, 2, 1, 3>>;

    // clang-format off
    // lane cluster: {Impl::kCMLane, Impl::kCNLane}
    // lane slice: {MRepeat, NRepeat, Impl::kCM0PerLane, Impl::kCM1PerLane, MInterleave, Impl::kCNPerLane, NInterleave}
    // clang-format on
    using CWarpPermuteDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane, MInterleave>,
              sequence<NRepeat, Impl::kCNLane, Impl::kCNPerLane, NInterleave>>,
        tuple<sequence<1, 2>>,
        tuple<sequence<2, 1>>,
        sequence<1, 2, 1, 1, 1, 2, 2>,
        sequence<0, 0, 1, 3, 4, 2, 3>>;

    CK_TILE_DEVICE void
    operator()(CBufType& c_buf, const ABufType& a_buf, const BBufType& b_buf) const
    {
        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            // 1 mmac issue
                            Impl{}(c_buf(iMRepeat * NRepeat * MInterleave * NInterleave +
                                         iNRepeat * MInterleave * NInterleave +
                                         iMInterleave * NInterleave + iNInterleave),
                                   a_buf[iMRepeat * MInterleave * kKIter + iMInterleave * kKIter +
                                         iKIter],
                                   b_buf[iNRepeat * NInterleave * kKIter + iNInterleave * kKIter +
                                         iKIter]);
                        });
                    });
                });
            });
        });
    }

    template <typename CWarpTensor>
    CK_TILE_DEVICE static auto GetCWarpPermuteTensor(const CWarpTensor& c_warp_tensor)
    {
        constexpr auto c_warp_permute_dstr =
            make_static_tile_distribution(CWarpPermuteDstrEncoding{});

        auto c_warp_permute_tensor = make_static_distributed_tensor<CDataType>(c_warp_permute_dstr);

        using SFC = space_filling_curve<sequence<MRepeat,
                                                 NRepeat,
                                                 MInterleave,
                                                 NInterleave,
                                                 Impl::kCNPerLane,
                                                 Impl::kCM0PerLane,
                                                 Impl::kCM1PerLane>,
                                        sequence<0, 1, 2, 3, 4, 5, 6>,
                                        sequence<1, 1, 1, 1, 1, 1, 1>>;

        constexpr auto num_access = SFC::get_num_of_access();

        static_for<0, num_access, 1>{}([&](auto access_id) {
            constexpr auto idx = SFC::get_index(access_id);

            constexpr auto iMR  = idx.at(number<0>{});
            constexpr auto iNR  = idx.at(number<1>{});
            constexpr auto iMI  = idx.at(number<2>{});
            constexpr auto iNI  = idx.at(number<3>{});
            constexpr auto iCN  = idx.at(number<4>{});
            constexpr auto iCM0 = idx.at(number<5>{});
            constexpr auto iCM1 = idx.at(number<6>{});

            c_warp_permute_tensor.set_y_sliced_thread_data(
                sequence<iMR, iNR, iCM0, iCM1, iMI, iCN, iNI>{},
                sequence<1, 1, 1, 1, 1, 1, 1>{},
                c_warp_tensor.get_y_sliced_thread_data(
                    sequence<iMR, iNR, iMI, iNI, iCN, iCM0, iCM1>{},
                    sequence<1, 1, 1, 1, 1, 1, 1>{}));
        });

        return c_warp_permute_tensor;
    }
};

// V2 version refers to KIterate stride =  warp_size * vec_a/b， not continuous vec_a/b
template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat_,
          index_t NRepeat_,
          index_t MInterleave_,
          index_t NInterleave_,
          index_t kKIter>
struct WarpGemmAttributeMmacIterateKTransC_v2
    : WarpGemmAttributeMmacIterateKTransC<WarpGemmAttributeMmacImpl_,
                                          MRepeat_,
                                          NRepeat_,
                                          MInterleave_,
                                          NInterleave_,
                                          kKIter>
{
    using Base = WarpGemmAttributeMmacIterateKTransC<WarpGemmAttributeMmacImpl_,
                                                     MRepeat_,
                                                     NRepeat_,
                                                     MInterleave_,
                                                     NInterleave_,
                                                     kKIter>;
    using Impl = typename Base::Impl;
    static constexpr index_t MRepeat     = MRepeat_;
    static constexpr index_t NRepeat     = NRepeat_;
    static constexpr index_t MInterleave = MInterleave_;
    static constexpr index_t NInterleave = NInterleave_;

    using ADataType = typename Base::ADataType;
    using BDataType = typename Base::BDataType;
    using CDataType = typename Base::CDataType;

    using ABufType = typename Base::ABufType;
    using BBufType = typename Base::BBufType;
    using CBufType = typename Base::CBufType;

    using AWarpDstrEncoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<MRepeat_, Impl::kAMLane, MInterleave_>,
                                         sequence<kKIter, Impl::kABKLane, Impl::kABKPerLane>>,
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<1, 1>>,
                                   sequence<1, 1, 2, 2>,
                                   sequence<0, 2, 0, 2>>;
    using BWarpDstrEncoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<NRepeat_, Impl::kBNLane, NInterleave_>,
                                         sequence<kKIter, Impl::kABKLane, Impl::kABKPerLane>>,
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<1, 1>>,
                                   sequence<1, 1, 2, 2>,
                                   sequence<0, 2, 0, 2>>;
    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat_, MInterleave_, Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>,
              sequence<NRepeat_, NInterleave_, Impl::kCNLane, Impl::kCNPerLane>>,
        tuple<sequence<1, 2>>,
        tuple<sequence<3, 2>>,
        sequence<1, 2, 1, 2, 2, 1, 1>,
        sequence<0, 0, 1, 1, 3, 2, 4>>;

    CK_TILE_DEVICE void
    operator()(CBufType& c_buf, const ABufType& a_buf, const BBufType& b_buf) const
    {
        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            // 1 mmac issue
                            Impl{}(c_buf(iMRepeat * NRepeat * MInterleave * NInterleave +
                                         iNRepeat * MInterleave * NInterleave +
                                         iMInterleave * NInterleave + iNInterleave),
                                   a_buf[iMRepeat * MInterleave * kKIter + iMInterleave * kKIter +
                                         iKIter],
                                   b_buf[iNRepeat * NInterleave * kKIter + iNInterleave * kKIter +
                                         iKIter]);
                        });
                    });
                });
            });
        });
    }
};

template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat,
          index_t NRepeat,
          index_t MInterleave,
          index_t NInterleave,
          index_t kKIter>
struct WarpGemmAttributeMmacIterateKTransC_ds128
    : public WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImpl_,
                                           MRepeat,
                                           NRepeat,
                                           MInterleave,
                                           NInterleave,
                                           kKIter>
{
    using Base = WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImpl_,
                                               MRepeat,
                                               NRepeat,
                                               MInterleave,
                                               NInterleave,
                                               kKIter>;

    using Impl = typename Base::Impl;

    using ADataType = typename Base::ADataType;
    using BDataType = typename Base::BDataType;
    using CDataType = typename Base::CDataType;

    using ABufType = typename Base::ABufType;
    using BBufType = typename Base::BBufType;
    using CBufType = typename Base::CBufType;

    static constexpr index_t kM = Base::kM;
    static constexpr index_t kN = Base::kN;
    static constexpr index_t kK = Base::kK;

    // ds_128
    static constexpr index_t kABKPerLane  = Impl::kABKPerLane * kKIter;
    static constexpr index_t kABKPerLane1 = 16 / sizeof(ADataType);
    static constexpr index_t kABKPerLane0 = kABKPerLane / kABKPerLane1;

    // static_assert(kABKPerLane % kABKPerLane1 == 0);

    using AWarpDstrEncoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<MRepeat, Impl::kAMLane, MInterleave>,
                                         sequence<kABKPerLane0, Impl::kABKLane, kABKPerLane1>>,
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<1, 1>>,
                                   sequence<1, 1, 2, 2>,
                                   sequence<0, 2, 0, 2>>;

    using BWarpDstrEncoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<NRepeat, Impl::kBNLane, NInterleave>,
                                         sequence<kABKPerLane0, Impl::kABKLane, kABKPerLane1>>,
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<1, 1>>,
                                   sequence<1, 1, 2, 2>,
                                   sequence<0, 2, 0, 2>>;

    // clang-format off
    // lane cluster: {Impl::kCMLane, Impl::kCNLane}
    // lane slice: {MRepeat,NRepeat, MInterleave, NInterleave, Impl::kCNPerLane, Impl::kCM0PerLane, Impl::kCM1PerLane}
    // clang-format on
    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane, MInterleave>,
              sequence<NRepeat, Impl::kCNLane, Impl::kCNPerLane, NInterleave>>,
        tuple<sequence<1, 2>>,
        tuple<sequence<2, 1>>,
        sequence<1, 2, 1, 2, 2, 1, 1>,
        sequence<0, 0, 4, 3, 2, 1, 3>>;

    // FIXME: compatible
    using CWarpPermuteDstrEncoding = CWarpDstrEncoding;
        
    CK_TILE_DEVICE void
    operator()(CBufType& c_buf, const ABufType& a_buf, const BBufType& b_buf) const
    {
        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            // 1 mmac issue
                            Impl{}(c_buf(iMRepeat * NRepeat * MInterleave * NInterleave +
                                         iNRepeat * MInterleave * NInterleave +
                                         iMInterleave * NInterleave + iNInterleave),
                                   a_buf[iMRepeat * MInterleave * kKIter + iMInterleave * kKIter +
                                         iKIter],
                                   b_buf[iNRepeat * NInterleave * kKIter + iNInterleave * kKIter +
                                         iKIter]);
                        });
                    });
                });
            });
        });
    }
};

// MoE, for loading Matix B into swizzled LDS in the 2nd GEMM of MOE. K larger than 64
template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat_,
          index_t NRepeat_,
          index_t MInterleave_,
          index_t NInterleave_,
          index_t kKIter>
struct WarpGemmAttributeMmacIterateKTransC_v3
    : WarpGemmAttributeMmacIterateKTransC<WarpGemmAttributeMmacImpl_,
                                          MRepeat_,
                                          NRepeat_,
                                          MInterleave_,
                                          NInterleave_,
                                          kKIter>
{
    static_assert((kKIter > 0) && (kKIter % 2 == 0), "kKIter wrong!");

    // V2 version refers to KIterate stride =  warp_size * vec_a/b， not continuous vec_a/b
    using Base = WarpGemmAttributeMmacIterateKTransC<WarpGemmAttributeMmacImpl_,
                                                     MRepeat_,
                                                     NRepeat_,
                                                     MInterleave_,
                                                     NInterleave_,
                                                     kKIter>;
    using Impl = typename Base::Impl;
    static constexpr index_t MRepeat     = MRepeat_;
    static constexpr index_t NRepeat     = NRepeat_;
    static constexpr index_t MInterleave = MInterleave_;
    static constexpr index_t NInterleave = NInterleave_;

    using ADataType = typename Base::ADataType;
    using BDataType = typename Base::BDataType;
    using CDataType = typename Base::CDataType;

    using ABufType = typename Base::ABufType;
    using BBufType = typename Base::BBufType;
    using CBufType = typename Base::CBufType;

    using AWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat_, Impl::kAMLane, MInterleave_>,
              sequence<kKIter / 2, Impl::kABKLane, Impl::kABKPerLane * 2>>, // multiply by 2:  make
                                                                            // ds_read_b64 to
                                                                            // ds_read_b128
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 1>>,
        sequence<1, 1, 2, 2>,
        sequence<0, 2, 0, 2>>;
    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<NRepeat_, Impl::kBNLane, NInterleave_>,
              sequence<kKIter / 2, Impl::kABKLane, Impl::kABKPerLane * 2>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 1>>,
        sequence<1, 1, 2, 2>,
        sequence<0, 2, 0, 2>>;
    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat_, MInterleave_, Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>,
              sequence<NRepeat_, NInterleave_, Impl::kCNLane, Impl::kCNPerLane>>,
        tuple<sequence<1, 2>>,
        tuple<sequence<3, 2>>,
        sequence<1, 2, 1, 2, 2, 1, 1>,
        sequence<0, 0, 1, 1, 3, 2, 4>>;

    CK_TILE_DEVICE void
    operator()(CBufType& c_buf, const ABufType& a_buf, const BBufType& b_buf) const
    {
        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            // 1 mmac issue
                            Impl{}(c_buf(iMRepeat * NRepeat * MInterleave * NInterleave +
                                         iNRepeat * MInterleave * NInterleave +
                                         iMInterleave * NInterleave + iNInterleave),
                                   a_buf[iMRepeat * MInterleave * kKIter + iMInterleave * kKIter +
                                         iKIter],
                                   b_buf[iNRepeat * NInterleave * kKIter + iNInterleave * kKIter +
                                         iKIter]);
                        });
                    });
                });
            });
        });
    }
};

// [HCU移植] 本结构体为 FMHA 移植新增（上游 github CK 没有）。仅用于 gfx938/gfx946。
// 这些新架构上 MMAC 的“操作数交换”不像 AMD MFMA 那样能转置 C 布局，因此改用带 lts
// (lane transpose sum) 的 _lit_lts 内建函数得到转置后的 C，并配套这套 CWarpDstrEncoding。
// ------------------------------------------------------------------
// LitLts: Hygon HCU gfx938+ __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(lit=true,lts=true).
// Hardware C register layout: row = 4*(lane/16)+r, col = c*16+lane%16.
// Each lane holds kCM1PerLane=4 consecutive M elements and kCNPerLane=1 N element.
// Lane distribution: kCMLane=4 (M lanes), kCNLane=16 (N lanes).
// Uses TransC-style CWarpDstrEncoding which correctly handles 4×16 lane grouping.
// Key difference from AMD TransC: M elements are consecutive (kCM0PerLane=1,kCM1PerLane=4),
// not strided (kCM0PerLane=4,kCM1PerLane=1).
template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat_,
          index_t NRepeat_,
          index_t MInterleave_,
          index_t NInterleave_,
          index_t kKIter>
struct WarpGemmAttributeMmacIterateKLitLts
    : public WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImpl_,
                                           MRepeat_,
                                           NRepeat_,
                                           MInterleave_,
                                           NInterleave_,
                                           kKIter>
{
    using Base = WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImpl_,
                                               MRepeat_,
                                               NRepeat_,
                                               MInterleave_,
                                               NInterleave_,
                                               kKIter>;

    using Impl = typename Base::Impl;

    using ADataType = typename Base::ADataType;
    using BDataType = typename Base::BDataType;
    using CDataType = typename Base::CDataType;

    using ABufType = typename Base::ABufType;
    using BBufType = typename Base::BBufType;
    using CBufType = typename Base::CBufType;

    static constexpr index_t kM          = Base::kM;
    static constexpr index_t kN          = Base::kN;
    static constexpr index_t kK          = Base::kK;
    static constexpr index_t MRepeat     = MRepeat_;
    static constexpr index_t NRepeat     = NRepeat_;
    static constexpr index_t MInterleave = MInterleave_;
    static constexpr index_t NInterleave = NInterleave_;

    using AWarpDstrEncoding = typename Base::AWarpDstrEncoding;
    using BWarpDstrEncoding = typename Base::BWarpDstrEncoding;

    // CWarpDstrEncoding: base structure + Ps2RHss swapped from <2,1> to <1,2>.
    // Hardware lane layout for LitLts: lane = M_lane*16 + N_lane (P[0]→M_lane, P[1]→N_lane).
    // Base encoding uses P[0]→N_lane, P[1]→M_lane which is transposed vs hardware,
    // causing a transposed (M,N) mapping. Swapping Ps2RHss fixes this.
    // Uses kCMPerLane/kCN0PerLane/kCN1PerLane (base sub-dims), same as base encoding.
    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCMLane, MInterleave, Impl::kCMPerLane>,
              sequence<NRepeat, Impl::kCNLane, NInterleave, Impl::kCN0PerLane, Impl::kCN1PerLane>>,
        tuple<sequence<1, 2>>,
        tuple<sequence<1, 1>>,
        sequence<1, 2, 1, 1, 2, 2, 2>,
        sequence<0, 0, 2, 3, 3, 2, 4>>;

    CK_TILE_DEVICE void
    operator()(CBufType& c_buf, const ABufType& a_buf, const BBufType& b_buf) const
    {
        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            Impl{}(c_buf(iMRepeat * NRepeat * MInterleave * NInterleave +
                                         iNRepeat * MInterleave * NInterleave +
                                         iMInterleave * NInterleave + iNInterleave),
                                   a_buf[iMRepeat * MInterleave * kKIter + iMInterleave * kKIter +
                                         iKIter],
                                   b_buf[iNRepeat * NInterleave * kKIter + iNInterleave * kKIter +
                                         iKIter]);
                        });
                    });
                });
            });
        });
    }
};

template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat_,
          index_t NRepeat_,
          index_t MInterleave_,
          index_t NInterleave_,
          index_t kKIter>
struct WarpGemmAttributeMmacIterateKShuffle
    : public WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImpl_,
                                           MRepeat_,
                                           NRepeat_,
                                           MInterleave_,
                                           NInterleave_,
                                           kKIter>
{
    static_assert(kKIter > 0, "wrong!");

    using Impl = remove_cvref_t<WarpGemmAttributeMmacImpl_>;
    using Base = WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImpl_,
                                               MRepeat_,
                                               NRepeat_,
                                               MInterleave_,
                                               NInterleave_,
                                               kKIter>;

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    // ext_vector_t<datatype, ext_vector_t>, 2d array 2 x vec_a
    using ABufType = thread_buffer<typename Impl::AVecType, kKIter * MRepeat_ * MInterleave_>;
    using BBufType = thread_buffer<typename Impl::BVecType, kKIter * NRepeat_ * NInterleave_>;
    using CBufType =
        thread_buffer<typename Impl::CVecType, MRepeat_ * NRepeat_ * MInterleave_ * NInterleave_>;

    static constexpr index_t kM = Impl::kM * MRepeat_ * MInterleave_;
    static constexpr index_t kN = Impl::kN * NRepeat_ * NInterleave_;
    static constexpr index_t kK = Impl::kK * kKIter;

    using AWarpDstrEncoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<MRepeat_, Impl::kAMLane, MInterleave_>,
                                         sequence<kKIter/2, Impl::kABKLane, Impl::kABKPerLane*2>>,
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<1, 1>>,
                                   sequence<1, 1, 2, 2>,
                                   sequence<0, 2, 0, 2>>;

    using BWarpDstrEncoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<NRepeat_, Impl::kBNLane, NInterleave_>,
                                         sequence<kKIter/2,Impl::kABKLane, Impl::kABKPerLane * 2>>,
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<1, 1>>,
                                   sequence<1, 1, 2, 2>,
                                   sequence<0, 2, 0, 2>>;

    using CWarpDstrEncoding = typename Base::CWarpDstrEncoding;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CBufType& c_buf, const ABufType& a_buf, const BBufType& b_buf) const
    {
        static_for<0, MRepeat_, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat_, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave_, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave_, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter/2, 1>{}([&](auto iKIter) {
                            // 1 mmac issue
                            Impl{}(c_buf(iMRepeat * NRepeat_ * MInterleave_ * NInterleave_ +
                                         iNRepeat * MInterleave_ * NInterleave_ +
                                         iMInterleave * NInterleave_ + iNInterleave),
                                   a_buf[iMRepeat * MInterleave_ * kKIter + iMInterleave * kKIter +
                                         iKIter * 2],
                                   b_buf[iNRepeat * NInterleave_ * kKIter + iNInterleave * kKIter +
                                         iKIter * 2]);
                            Impl{}(c_buf(iMRepeat * NRepeat_ * MInterleave_ * NInterleave_ +
                                         iNRepeat * MInterleave_ * NInterleave_ +
                                         iMInterleave * NInterleave_ + iNInterleave),
                                   a_buf[iMRepeat * MInterleave_ * kKIter + iMInterleave * kKIter +
                                         iKIter * 2 + 1],
                                   b_buf[iNRepeat * NInterleave_ * kKIter + iNInterleave * kKIter +
                                         iKIter * 2 + 1]);
                        });
                    });
                });
            });
        });
    }

    // c_vec = a_vec * b_vec
    // AVecType = ext_vector_t<ADataType, AWarpTensor::get_thread_buffer_size()>;
    CK_TILE_DEVICE CBufType operator()(const ABufType& a_buf, const BBufType& b_buf) const
    {
        CBufType c_buf;

        // c += a * b
        static_for<0, MRepeat_, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat_, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave_, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave_, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter/2, 1>{}([&](auto iKIter) {
                            // 1 mmac issue
                            Impl{}(c_buf(iMRepeat * NRepeat_ * MInterleave_ * NInterleave_ +
                                         iNRepeat * MInterleave_ * NInterleave_ +
                                         iMInterleave * NInterleave_ + iNInterleave),
                                   a_buf[iMRepeat * MInterleave_ * kKIter + iMInterleave * kKIter +
                                         iKIter * 2],
                                   b_buf[iNRepeat * NInterleave_ * kKIter + iNInterleave * kKIter +
                                         iKIter * 2]);
                            Impl{}(c_buf(iMRepeat * NRepeat_ * MInterleave_ * NInterleave_ +
                                         iNRepeat * MInterleave_ * NInterleave_ +
                                         iMInterleave * NInterleave_ + iNInterleave),
                                   a_buf[iMRepeat * MInterleave_ * kKIter + iMInterleave * kKIter +
                                         iKIter * 2 + 1],
                                   b_buf[iNRepeat * NInterleave_ * kKIter + iNInterleave * kKIter +
                                         iKIter * 2 + 1]);
                        });
                    });
                });
            });
        });

        return c_buf;
    }
};


template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat_,
          index_t NRepeat_,
          index_t MInterleave_,
          index_t NInterleave_,
          index_t kKIter>
struct WarpGemmAttributeMmacIterateKTransC_Shuffle
    : public WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImpl_,
                                           MRepeat_,
                                           NRepeat_,
                                           MInterleave_,
                                           NInterleave_,
                                           kKIter>
{
    using Base = WarpGemmAttributeMmacIterateKTransC<WarpGemmAttributeMmacImpl_,
                                                    MRepeat_,
                                                    NRepeat_,
                                                    MInterleave_,
                                                    NInterleave_,
                                                    kKIter>;

    using Impl = typename Base::Impl;

    using ADataType = typename Base::ADataType;
    using BDataType = typename Base::BDataType;
    using CDataType = typename Base::CDataType;

    using ABufType = typename Base::ABufType;
    using BBufType = typename Base::BBufType;
    using CBufType = typename Base::CBufType;

    static constexpr index_t kM          = Base::kM;
    static constexpr index_t kN          = Base::kN;
    static constexpr index_t kK          = Base::kK;
    static constexpr index_t MRepeat     = MRepeat_;
    static constexpr index_t NRepeat     = NRepeat_;
    static constexpr index_t MInterleave = MInterleave_;
    static constexpr index_t NInterleave = NInterleave_;

    using AWarpDstrEncoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<MRepeat_, Impl::kAMLane, MInterleave_>,   // <1, 16, 1>
                                         sequence<kKIter/2, Impl::kABKLane, Impl::kABKPerLane*2>>,  // <8/2, 4, 4*2>
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<1, 1>>,
                                   sequence<1, 1, 2, 2>,
                                   sequence<0, 2, 0, 2>>;

    using BWarpDstrEncoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<NRepeat_, Impl::kBNLane, NInterleave_>,   // <2, 16, 1>
                                         sequence<kKIter/2, Impl::kABKLane, Impl::kABKPerLane * 2>>, // <8/2, 4, 4*2>
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<1, 1>>,
                                   sequence<1, 1, 2, 2>,
                                   sequence<0, 2, 0, 2>>;

    // clang-format off
    // lane cluster: {Impl::kCMLane, Impl::kCNLane}
    // lane slice: {MRepeat,NRepeat, MInterleave, NInterleave, Impl::kCNPerLane, Impl::kCM0PerLane, Impl::kCM1PerLane}
    // clang-format on
    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane, MInterleave>,  
              sequence<NRepeat, Impl::kCNLane, Impl::kCNPerLane, NInterleave>>,     // <2,16,1,1>
        tuple<sequence<1, 2>>,
        tuple<sequence<2, 1>>,
        sequence<1, 2, 1, 2, 2, 1, 1>,
        sequence<0, 0, 4, 3, 2, 1, 3>>;

    CK_TILE_DEVICE void
    operator()(CBufType& c_buf, const ABufType& a_buf, const BBufType& b_buf) const
    {
        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter/2, 1>{}([&](auto iKIter) {
                            // 1 mmac issue
                            Impl{}(c_buf(iMRepeat * NRepeat * MInterleave * NInterleave +
                                         iNRepeat * MInterleave * NInterleave +
                                         iMInterleave * NInterleave + iNInterleave),
                                   a_buf[iMRepeat * MInterleave * kKIter + iMInterleave * kKIter +
                                         iKIter * 2],
                                   b_buf[iNRepeat * NInterleave * kKIter + iNInterleave * kKIter +
                                         iKIter * 2]);
                            Impl{}(c_buf(iMRepeat * NRepeat * MInterleave * NInterleave +
                                        iNRepeat * MInterleave * NInterleave +
                                        iMInterleave * NInterleave + iNInterleave),
                                    a_buf[iMRepeat * MInterleave * kKIter + iMInterleave * kKIter +
                                        iKIter * 2 + 1],
                                    b_buf[iNRepeat * NInterleave * kKIter + iNInterleave * kKIter +
                                        iKIter * 2 + 1]);

                        });
                    });
                });
            });
        });
    }
};

template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat,
          index_t NRepeat,
          index_t MInterleave,
          index_t NInterleave,
          index_t kKIter>
struct WarpGemmAttributeInt8MmacIterateK
{
    static_assert(kKIter > 0, "wrong!");

    using Impl = remove_cvref_t<WarpGemmAttributeMmacImpl_>;

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    using AComputeDataType = typename Impl::AComputeDataType;
    using BComputeDataType = typename Impl::BComputeDataType;
    using CComputeDataType = typename Impl::CComputeDataType;
    

    // ext_vector_t<datatype, ext_vector_t>, 2d array 2 x vec_a
    // Impl::AvecType is Vi2=2*int32=2*4*int8
    static_assert(vector_traits<typename Impl::AInt8VecType>::vector_size ==8,"-----------------------21------------------");
    static_assert(vector_traits<typename Impl::BInt8VecType>::vector_size ==8,"-----------------------22------------------");
    static_assert(vector_traits<typename Impl::CInt32VecType>::vector_size ==4,"-----------------------23------------------");
    using AVecType =
        ext_vector_t<ADataType, vector_traits<typename Impl::AInt8VecType>::vector_size * kKIter * MRepeat * MInterleave>;
    using BVecType =
        ext_vector_t<BDataType, vector_traits<typename Impl::BInt8VecType>::vector_size * kKIter * NRepeat * NInterleave>;
    using CVecType = 
        ext_vector_t<CDataType, vector_traits<typename Impl::CInt32VecType>::vector_size * MRepeat * NRepeat * MInterleave * NInterleave>;
    
    // using ABufType = thread_buffer<typename Impl::AVecType, kKIter * MRepeat * MInterleave>;
    // using BBufType = thread_buffer<typename Impl::BVecType, kKIter * NRepeat * NInterleave>;
    // using CBufType = thread_buffer<typename Impl::CVecType, MRepeat * NRepeat * MInterleave * NInterleave>;

    // static_assert(kKIter*MRepeat*MInterleave==1*2*1,"-----------------------31------------------");
    // static_assert(kKIter*NRepeat*NInterleave==1*1*4,"-----------------------32------------------");
    // static_assert(MRepeat*NRepeat*MInterleave*NInterleave==2*1*1*4,"-----------------------33------------------");
    static constexpr index_t kM = Impl::kM * MRepeat * MInterleave;
    static constexpr index_t kN = Impl::kN * NRepeat * NInterleave;
    static constexpr index_t kK = Impl::kK * kKIter;
    static constexpr index_t kKPerThread = Impl::kABKPerLane * kKIter;

    using AWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kAMLane, MInterleave>, sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 1>>,
        sequence<1, 1, 2>,
        sequence<0, 2, 1>>;

    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<NRepeat, Impl::kBNLane, NInterleave>, sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 1>>,
        sequence<1, 1, 2>,
        sequence<0, 2, 1>>;
    
    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCMLane, MInterleave, Impl::kCMPerLane>,
              sequence<NRepeat, Impl::kCNLane, NInterleave, Impl::kCN0PerLane, Impl::kCN1PerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 1>>,
        sequence<1, 2, 1, 1, 2, 2, 2>, // <MRepeat, NRepeat, MInterleave, MPerLane, N0PerLane,
                                       // NmmacInterleave, N1PerLane>
        sequence<0, 0, 2, 3, 3, 2, 4>>;

    using CWarpOutputDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCMLane, MInterleave, Impl::kCMPerLane>,
              sequence<NRepeat, Impl::kCN0PerLane, Impl::kCNLane, NInterleave * Impl::kCN1PerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<2, 1>>,
        sequence<1, 2, 1, 1, 2, 2>, // <MRepeat, NRepeat, MInterleave, MPerLane, N0PerLane,
                                    // NmmacInterleave, N1PerLane>
        sequence<0, 0, 2, 3, 1, 3>>;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        using buf_a = thread_buffer<typename Impl::AInt8VecType, kKIter * MRepeat * MInterleave>;
        using buf_b = thread_buffer<typename Impl::BInt8VecType, kKIter * NRepeat * NInterleave>;
        using buf_c = thread_buffer<typename Impl::CInt32VecType, MRepeat * NRepeat * MInterleave * NInterleave>;
	    // static_assert(MRepeat*NRepeat*MInterleave*NInterleave==2*1*1*4,"-------------------------------41----------------------");

        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            //1 mmac issue
                            Impl{}(reinterpret_cast<buf_c&>(c_vec)
                                       .template get_as<typename Impl::CVecType>()[iMRepeat * NRepeat * MInterleave * NInterleave + 
                                                                                   iNRepeat * MInterleave * NInterleave +
                                                                                   iMInterleave * NInterleave + iNInterleave], 
                                   reinterpret_cast<const buf_a&>(a_vec)
                                       .template get_as<typename Impl::AVecType>()[iMRepeat * MInterleave * kKIter + 
                                                                                   iMInterleave * kKIter + iKIter],
                                   reinterpret_cast<const buf_b&>(b_vec)
                                       .template get_as<typename Impl::BVecType>()[iNRepeat * NInterleave * kKIter + 
                                                                                   iNInterleave * kKIter + iKIter]);
                        });
                    });
                });
            });
        });
    }
    
    // c_vec += a_vec * b_vec
    // CK_TILE_DEVICE void
    // operator()(CBufType& c_buf, const ABufType& a_buf, const BBufType& b_buf) const
    // {
    //     // using buf_a = thread_buffer<typename Impl::AInt8VecType, kKIter * MRepeat * MInterleave>;
    //     // using buf_b = thread_buffer<typename Impl::BInt8VecType, kKIter * NRepeat * NInterleave>;
    //     // using buf_c = thread_buffer<typename Impl::CInt32VecType, MRepeat * NRepeat * MInterleave * NInterleave>;
	//     // static_assert(MRepeat*NRepeat*MInterleave*NInterleave==2*1*1*4,"-------------------------------41----------------------");

    //     static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
    //         static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
    //             static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
    //                 static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
    //                     static_for<0, kKIter, 1>{}([&](auto iKIter) {
    //                         //1 mmac issue
    //                         Impl{}(c_buf(iMRepeat * NRepeat * MInterleave * NInterleave + 
    //                                     iNRepeat * MInterleave * NInterleave +
    //                                     iMInterleave * NInterleave + iNInterleave), 
    //                                a_buf(iMRepeat * MInterleave * kKIter + 
    //                                      iMInterleave * kKIter + iKIter),
    //                                b_buf(iNRepeat * NInterleave * kKIter + 
    //                                      iNInterleave * kKIter + iKIter));
    //                     });
    //                 });
    //             });
    //         });
    //     });
    // }

    // c_vec = a_vec * b_vec
    // AVecType = ext_vector_t<ADataType, AWarpTensor::get_thread_buffer_size()>;
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        using buf_a = thread_buffer<typename Impl::AVecType, kKIter * MRepeat * MInterleave>;
        using buf_b = thread_buffer<typename Impl::BVecType, kKIter * NRepeat * NInterleave>;
        using buf_c = thread_buffer<typename Impl::CVecType, MRepeat * NRepeat * MInterleave * NInterleave>;

        buf_c c_vec;

        // c += a * b
        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            Impl{}(c_vec.template get_as<typename Impl::CVecType>()[iMRepeat * NRepeat * MInterleave * NInterleave + 
                                                                                   iNRepeat * MInterleave * NInterleave +
                                                                                   iMInterleave * NInterleave + 
                                                                                   iNInterleave], 
                                   reinterpret_cast<const buf_a&>(a_vec)
                                       .template get_as<typename Impl::AVecType>()[iMRepeat * MInterleave * kKIter + 
                                                                                   iMInterleave * kKIter + 
                                                                                   iKIter],
                                   reinterpret_cast<const buf_b&>(b_vec)
                                       .template get_as<typename Impl::BVecType>()[iNRepeat * NInterleave * kKIter + 
                                                                                   iNInterleave * kKIter + 
                                                                                   iKIter]);
                        });
                    });
                });
            });
        });

        return reinterpret_cast<CVecType>(c_vec);
    }

    template <typename CWarpTensor>
    CK_TILE_DEVICE auto MakeCOutputLayout(const CWarpTensor& c_warp_tensor) const
    {
        constexpr auto c_warp_output_distribution = make_static_tile_distribution(CWarpOutputDstrEncoding{});
        auto c_warp_output_tensor = make_static_distributed_tensor<CDataType>(c_warp_output_distribution);
        
        static_for<0, MRepeat, 1>{}([&](auto iMR){
            static_for<0, NRepeat, 1>{}([&](auto iNR){
                static_for<0, MInterleave, 1>{} ([&](auto iMI){
                    static_for<0, NInterleave, 1>{}([&](auto iNI){
                        static_for<0, Impl::kCN0PerLane, 1>{}([&](auto iCN0){
                            
                            c_warp_output_tensor.set_y_sliced_thread_data(
                                sequence<iMR, iNR, iMI, 0, iCN0, iNI>{}, //output_orgin
                                sequence<1, 1, 1, 1, 1, Impl::kCN1PerLane>{}, //output_set_length
                                c_warp_tensor.get_y_sliced_thread_data( //get calculate tensor and change to output layout
                                    sequence<iMR, iNR, iMI, 0, iNI, iCN0, 0>{},
                                    sequence<1, 1, 1, 1, 1, 1, Impl::kCN1PerLane>{}));

                        });

                    });
                });
            });
        });

        return c_warp_output_tensor;
    }
};

template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat_,
          index_t NRepeat_,
          index_t MInterleave_,
          index_t NInterleave_,
          index_t kKIter_>
struct WarpGemmAttributeInt8ScaleChannelMmacIterateK
{
    static_assert(kKIter_ > 0, "wrong!");

    using Impl = remove_cvref_t<WarpGemmAttributeMmacImpl_>;
    using Base = WarpGemmAttributeInt8MmacIterateK<WarpGemmAttributeMmacImpl_,
                                                   MRepeat_,
                                                   NRepeat_,
                                                   MInterleave_,
                                                   NInterleave_,
                                                   kKIter_>;

    static constexpr index_t MRepeat     = MRepeat_;
    static constexpr index_t NRepeat     = NRepeat_;
    static constexpr index_t MInterleave = MInterleave_;
    static constexpr index_t NInterleave = NInterleave_;
    static constexpr index_t kKIter      = kKIter_;

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    using AScaleType = typename Impl::AScaleType;
    using BScaleType = typename Impl::BScaleType;

    // ext_vector_t<datatype, ext_vector_t>, 2d array 2 x vec_a
    // Impl::AvecType is Vi2=2*int32=2*4*int8
    static_assert(vector_traits<typename Impl::AInt8VecType>::vector_size ==8,"Int8VecType vector size is not 8");
    static_assert(vector_traits<typename Impl::BInt8VecType>::vector_size ==8,"Int8VecType vector size is not 8");
    static_assert(vector_traits<typename Impl::CFloat32VecType>::vector_size ==4,"Float32VecType vector size is not 4");
    using AVecType =
        ext_vector_t<ADataType, vector_traits<typename Impl::AInt8VecType>::vector_size * kKIter * MRepeat * MInterleave>;
    using BVecType =
        ext_vector_t<BDataType, vector_traits<typename Impl::BInt8VecType>::vector_size * kKIter * NRepeat * NInterleave>;
    using CVecType = 
        ext_vector_t<CDataType, vector_traits<typename Impl::CInt32VecType>::vector_size * MRepeat * NRepeat * MInterleave * NInterleave>;

    using AScaleVec = ext_vector_t<AScaleType, MRepeat * MInterleave>;
    using BScaleVec = ext_vector_t<BScaleType, NRepeat * NInterleave>;
    
    using ABufType = thread_buffer<typename Impl::AVecType, kKIter * MRepeat * MInterleave>;
    using BBufType = thread_buffer<typename Impl::BVecType, kKIter * NRepeat * NInterleave>;
    using CBufType = thread_buffer<typename Impl::CVecType, MRepeat * NRepeat * MInterleave * NInterleave>;
    using AScaleBufType = thread_buffer<AScaleType, MRepeat * MInterleave>;
    using BScaleBufType = thread_buffer<BScaleType, NRepeat * NInterleave>;


    static constexpr index_t kM = Impl::kM * MRepeat * MInterleave;
    static constexpr index_t kN = Impl::kN * NRepeat * NInterleave;
    static constexpr index_t kK = Impl::kK * kKIter;
    static constexpr index_t kKPerThread = Impl::kABKPerLane * kKIter;

    static constexpr index_t kKIteration = kKIter;
    static constexpr index_t kABKPerLane = Impl::kABKPerLane;
    static constexpr index_t kCN0PerLane = Impl::kCN0PerLane;

    using AWarpDstrEncoding = typename Base::AWarpDstrEncoding;
    using BWarpDstrEncoding = typename Base::BWarpDstrEncoding;
    using CWarpDstrEncoding = typename Base::CWarpDstrEncoding;


    static_assert(MInterleave == 1, "MInterleave must be 1 for scale channel");
    static_assert(MRepeat == 1, "MRepeat must be 1 for scale channel");

    using AScaleWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kAMLane, MInterleave>,
              sequence<Impl::kABKLane, 1>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 1>>,
        sequence<1, 1, 2>,
        sequence<0, 2, 1>>;

    using BScaleWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<NRepeat, Impl::kBNLane, NInterleave>,
              sequence<Impl::kABKLane, 1>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 1>>,
        sequence<1, 1, 2>,
        sequence<0, 2, 1>>;

  
    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CBufType& c_buf, const ABufType& a_buf, const BBufType& b_buf, const AScaleBufType& a_scale_buf, const BScaleBufType& b_scale_buf) const
    {
        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            //1 mmac issue
                            Impl{}(c_buf(iMRepeat * NRepeat * MInterleave * NInterleave + 
                                         iNRepeat * MInterleave * NInterleave +
                                         iMInterleave * NInterleave + iNInterleave),
                                   a_buf[iMRepeat * MInterleave * kKIter + iMInterleave * kKIter + 
                                         iKIter],
                                   b_buf[iNRepeat * NInterleave * kKIter + iNInterleave * kKIter + 
                                         iKIter],
                                   a_scale_buf[iMRepeat * MInterleave + iMInterleave],
                                   b_scale_buf[iNRepeat * NInterleave + iNInterleave]);
                        });
                    });
                });
            });
        });
    }

    // CK_TILE_DEVICE void
    // operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec, const AScaleVec& a_scale_vec, const BScaleVec& b_scale_vec) const
    // {
    //     using buf_a = thread_buffer<typename Impl::AInt8VecType, kKIter * MRepeat * MInterleave>;
    //     using buf_b = thread_buffer<typename Impl::BInt8VecType, kKIter * NRepeat * NInterleave>;
    //     using buf_a_scale = thread_buffer<AScaleType, MRepeat * MInterleave>;
    //     using buf_b_scale = thread_buffer<BScaleType, NRepeat * NInterleave>;
    //     using buf_c = thread_buffer<typename Impl::CFloat32VecType, MRepeat * NRepeat * MInterleave * NInterleave>;
    //     //using buf_c = thread_buffer<typename Impl::CVecType, MRepeat * NRepeat * MInterleave * NInterleave>;
        
    //     static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
    //         static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
    //             static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
    //                 static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
    //                     AScaleType a_scale = reinterpret_cast<const buf_a_scale&>(a_scale_vec)
    //                                            .template get_as<AScaleType>()[iMRepeat * MInterleave + iMInterleave];
    //                     BScaleType b_scale = reinterpret_cast<const buf_b_scale&>(b_scale_vec)
    //                                            .template get_as<BScaleType>()[iNRepeat * NInterleave + iNInterleave];

    //                     static_for<0, kKIter, 1>{}([&](auto iKIter) {
    //                         //1 mmac issue
    //                         Impl{}(reinterpret_cast<buf_c&>(c_vec)
    //                                    .template get_as<typename Impl::CVecType>()[iMRepeat * NRepeat * MInterleave * NInterleave + 
    //                                                                                iNRepeat * MInterleave * NInterleave +
    //                                                                                iMInterleave * NInterleave + 
    //                                                                                iNInterleave],
    //                                reinterpret_cast<const buf_a&>(a_vec)
    //                                    .template get_as<typename Impl::AVecType>()[iMRepeat * MInterleave * kKIter + 
    //                                                                                iMInterleave * kKIter + 
    //                                                                                iKIter],
    //                                reinterpret_cast<const buf_b&>(b_vec)
    //                                    .template get_as<typename Impl::BVecType>()[iNRepeat * NInterleave * kKIter + 
    //                                                                                iNInterleave * kKIter + 
    //                                                                                iKIter],
    //                                a_scale,
    //                                b_scale);
    //                     });
    //                 });
    //             });
    //         });
    //     });
    // }

    // c_vec = a_vec * b_vec
    // AVecType = ext_vector_t<ADataType, AWarpTensor::get_thread_buffer_size()>;
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        using buf_a = thread_buffer<typename Impl::AVecType, kKIter * MRepeat * MInterleave>;
        using buf_b = thread_buffer<typename Impl::BVecType, kKIter * NRepeat * NInterleave>;
        using buf_c = thread_buffer<typename Impl::CVecType, MRepeat * NRepeat * MInterleave * NInterleave>;

        buf_c c_vec;

        // c += a * b
        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            Impl{}(c_vec.template get_as<typename Impl::CVecType>()[iMRepeat * NRepeat * MInterleave * NInterleave + 
                                                                                   iNRepeat * MInterleave * NInterleave +
                                                                                   iMInterleave * NInterleave + 
                                                                                   iNInterleave], 
                                   reinterpret_cast<const buf_a&>(a_vec)
                                       .template get_as<typename Impl::AVecType>()[iMRepeat * MInterleave * kKIter + 
                                                                                   iMInterleave * kKIter + 
                                                                                   iKIter],
                                   reinterpret_cast<const buf_b&>(b_vec)
                                       .template get_as<typename Impl::BVecType>()[iNRepeat * NInterleave * kKIter + 
                                                                                   iNInterleave * kKIter + 
                                                                                   iKIter]);
                        });
                    });
                });
            });
        });

        return reinterpret_cast<CVecType>(c_vec);
    }

};

template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat,
          index_t NRepeat,
          index_t MInterleave,
          index_t NInterleave,
          index_t kKIter>
struct WarpGemmAttributeInt8MmacIterateKShuffle
        : public WarpGemmAttributeInt8MmacIterateK<WarpGemmAttributeMmacImpl_,
                                                    MRepeat,
                                                    NRepeat,
                                                    MInterleave,
                                                    NInterleave,
                                                    kKIter>
{
    static_assert(kKIter > 0 && kKIter % 2 == 0, "kKIter must be even and greater than 0!");

    using Impl = remove_cvref_t<WarpGemmAttributeMmacImpl_>;
    using Base = WarpGemmAttributeInt8MmacIterateK<WarpGemmAttributeMmacImpl_,
                                                    MRepeat,
                                                    NRepeat,
                                                    MInterleave,
                                                    NInterleave,
                                                    kKIter>;

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    using AComputeDataType = typename Impl::AComputeDataType;
    using BComputeDataType = typename Impl::BComputeDataType;
    using CComputeDataType = typename Impl::CComputeDataType;
    
    using AVecType =
        ext_vector_t<ADataType, vector_traits<typename Impl::AInt8VecType>::vector_size * kKIter * MRepeat * MInterleave>;
    using BVecType =
        ext_vector_t<BDataType, vector_traits<typename Impl::BInt8VecType>::vector_size * kKIter * NRepeat * NInterleave>;
    using CVecType = 
        ext_vector_t<CDataType, vector_traits<typename Impl::CInt32VecType>::vector_size * MRepeat * NRepeat * MInterleave * NInterleave>;


    using AWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kAMLane, MInterleave>, sequence<kKIter / 2, Impl::kABKLane, Impl::kABKPerLane * 2>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 1>>,
        sequence<1, 1, 2, 2>,
        sequence<0, 2, 0, 2>>;

    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<NRepeat, Impl::kBNLane, NInterleave>, sequence<kKIter / 2, Impl::kABKLane, Impl::kABKPerLane * 2>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 1>>,
        sequence<1, 1, 2, 2>,
        sequence<0, 2, 0, 2>>;
    
    using CWarpDstrEncoding = typename Base::CWarpDstrEncoding;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        using buf_a = thread_buffer<typename Impl::AInt8VecType, kKIter * MRepeat * MInterleave>;
        using buf_b = thread_buffer<typename Impl::BInt8VecType, kKIter * NRepeat * NInterleave>;
        using buf_c = thread_buffer<typename Impl::CInt32VecType, MRepeat * NRepeat * MInterleave * NInterleave>;
	    // static_assert(MRepeat*NRepeat*MInterleave*NInterleave==2*1*1*4,"-------------------------------41----------------------");

        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter/2, 1>{}([&](auto iKIter) {
                            //1 mmac issue
                            Impl{}(reinterpret_cast<buf_c&>(c_vec)
                                       .template get_as<typename Impl::CVecType>()[iMRepeat * NRepeat * MInterleave * NInterleave + 
                                                                                   iNRepeat * MInterleave * NInterleave +
                                                                                   iMInterleave * NInterleave + 
                                                                                   iNInterleave], 
                                   reinterpret_cast<const buf_a&>(a_vec)
                                       .template get_as<typename Impl::AVecType>()[iMRepeat * MInterleave * kKIter + 
                                                                                   iMInterleave * kKIter + 
                                                                                   iKIter * 2],
                                   reinterpret_cast<const buf_b&>(b_vec)
                                       .template get_as<typename Impl::BVecType>()[iNRepeat * NInterleave * kKIter + 
                                                                                   iNInterleave * kKIter + 
                                                                                   iKIter * 2]);
                            //2 mmac issue
                            Impl{}(reinterpret_cast<buf_c&>(c_vec)
                                       .template get_as<typename Impl::CVecType>()[iMRepeat * NRepeat * MInterleave * NInterleave + 
                                                                                   iNRepeat * MInterleave * NInterleave +
                                                                                   iMInterleave * NInterleave + 
                                                                                   iNInterleave], 
                                   reinterpret_cast<const buf_a&>(a_vec)
                                       .template get_as<typename Impl::AVecType>()[iMRepeat * MInterleave * kKIter + 
                                                                                   iMInterleave * kKIter + 
                                                                                   iKIter * 2 + 1],
                                   reinterpret_cast<const buf_b&>(b_vec)
                                       .template get_as<typename Impl::BVecType>()[iNRepeat * NInterleave * kKIter + 
                                                                                   iNInterleave * kKIter + 
                                                                                   iKIter * 2 + 1]);
                        });
                    });
                });
            });
        });
    }

    // c_vec = a_vec * b_vec
    // AVecType = ext_vector_t<ADataType, AWarpTensor::get_thread_buffer_size()>;
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        using buf_a = thread_buffer<typename Impl::AVecType, kKIter * MRepeat * MInterleave>;
        using buf_b = thread_buffer<typename Impl::BVecType, kKIter * NRepeat * NInterleave>;
        using buf_c = thread_buffer<typename Impl::CVecType, MRepeat * NRepeat * MInterleave * NInterleave>;

        buf_c c_vec;

        // c += a * b
        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter/2, 1>{}([&](auto iKIter) {
                            Impl{}(c_vec.template get_as<typename Impl::CVecType>()[iMRepeat * NRepeat * MInterleave * NInterleave + 
                                                                                   iNRepeat * MInterleave * NInterleave +
                                                                                   iMInterleave * NInterleave + 
                                                                                   iNInterleave], 
                                   reinterpret_cast<const buf_a&>(a_vec)
                                       .template get_as<typename Impl::AVecType>()[iMRepeat * MInterleave * kKIter + 
                                                                                   iMInterleave * kKIter + 
                                                                                   iKIter * 2],
                                   reinterpret_cast<const buf_b&>(b_vec)
                                       .template get_as<typename Impl::BVecType>()[iNRepeat * NInterleave * kKIter + 
                                                                                   iNInterleave * kKIter + 
                                                                                   iKIter * 2]);

                            Impl{}(c_vec.template get_as<typename Impl::CVecType>()[iMRepeat * NRepeat * MInterleave * NInterleave + 
                                                                                    iNRepeat * MInterleave * NInterleave +
                                                                                    iMInterleave * NInterleave + 
                                                                                    iNInterleave], 
                                    reinterpret_cast<const buf_a&>(a_vec)
                                        .template get_as<typename Impl::AVecType>()[iMRepeat * MInterleave * kKIter + 
                                                                                    iMInterleave * kKIter + 
                                                                                    iKIter * 2 + 1],
                                    reinterpret_cast<const buf_b&>(b_vec)
                                        .template get_as<typename Impl::BVecType>()[iNRepeat * NInterleave * kKIter + 
                                                                                    iNInterleave * kKIter + 
                                                                                    iKIter * 2 + 1]);
                        });
                    });
                });
            });
        });

        return reinterpret_cast<CVecType>(c_vec);
    }


};

template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat_,
          index_t NRepeat_,
          index_t MInterleave_,
          index_t NInterleave_,
          index_t kKIter>
struct WarpGemmAttributeInt8MmacIterateKTransC
    : public WarpGemmAttributeInt8MmacIterateK<WarpGemmAttributeMmacImpl_,
                                                MRepeat_,
                                                NRepeat_,
                                                MInterleave_,
                                                NInterleave_,
                                                kKIter>
{
    using Base = WarpGemmAttributeInt8MmacIterateK<WarpGemmAttributeMmacImpl_,
                                                    MRepeat_,
                                                    NRepeat_,
                                                    MInterleave_,
                                                    NInterleave_,
                                                    kKIter>;

    using Impl = typename Base::Impl;

    using ADataType = typename Base::ADataType;
    using BDataType = typename Base::BDataType;
    using CDataType = typename Base::CDataType;

    using AComputeDataType = typename Base::AComputeDataType;
    using BComputeDataType = typename Base::BComputeDataType;
    using CComputeDataType = typename Base::CComputeDataType;
    
    using AVecType = typename Base::AVecType;
    using BVecType = typename Base::BVecType;
    using CVecType = typename Base::CVecType;
    
    static constexpr index_t kM = Base::kM;
    static constexpr index_t kN = Base::kN;
    static constexpr index_t kK = Base::kK;
    static constexpr index_t MRepeat     = MRepeat_;
    static constexpr index_t NRepeat     = NRepeat_;
    static constexpr index_t MInterleave = MInterleave_;
    static constexpr index_t NInterleave = NInterleave_;

    using AWarpDstrEncoding = typename Base::AWarpDstrEncoding;

    using BWarpDstrEncoding = typename Base::BWarpDstrEncoding;

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane, MInterleave>, // <1, 4, 4, 1, 1>
              sequence<NRepeat, Impl::kCNLane, Impl::kCNPerLane, NInterleave>>,         // <2, 16, 1, 1> 
        tuple<sequence<1, 2>>,
        tuple<sequence<2, 1>>,
        sequence<1, 2, 1, 2, 2, 1, 1>,
        sequence<0, 0, 4, 3, 2, 1, 3>>;


    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        using buf_a = thread_buffer<typename Impl::AInt8VecType, kKIter * MRepeat * MInterleave>;
        using buf_b = thread_buffer<typename Impl::BInt8VecType, kKIter * NRepeat * NInterleave>;
        using buf_c = thread_buffer<typename Impl::CInt32VecType, MRepeat * NRepeat * MInterleave * NInterleave>;

        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            //1 mmac issue
                            Impl{}(reinterpret_cast<buf_c&>(c_vec)
                                       .template get_as<typename Impl::CVecType>()[iMRepeat * NRepeat * MInterleave * NInterleave + 
                                                                                   iNRepeat * MInterleave * NInterleave +
                                                                                   iMInterleave * NInterleave + iNInterleave], 
                                   reinterpret_cast<const buf_a&>(a_vec)
                                       .template get_as<typename Impl::AVecType>()[iMRepeat * MInterleave * kKIter + 
                                                                                   iMInterleave * kKIter + iKIter],
                                   reinterpret_cast<const buf_b&>(b_vec)
                                       .template get_as<typename Impl::BVecType>()[iNRepeat * NInterleave * kKIter + 
                                                                                   iNInterleave * kKIter + iKIter]);
                        });
                    });
                });
            });
        });
    }

};


template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat_,
          index_t NRepeat_,
          index_t MInterleave_,
          index_t NInterleave_,
          index_t kKIter>
struct WarpGemmAttributeInt8MmacIterateKTransC_Shuffle
    : public WarpGemmAttributeInt8MmacIterateK<WarpGemmAttributeMmacImpl_,
                                               MRepeat_,
                                               NRepeat_,
                                               MInterleave_,
                                               NInterleave_,
                                               kKIter>
{
    using Base = WarpGemmAttributeInt8MmacIterateKTransC<WarpGemmAttributeMmacImpl_,
                                                        MRepeat_,
                                                        NRepeat_,
                                                        MInterleave_,
                                                        NInterleave_,
                                                        kKIter>;

    using Impl = typename Base::Impl;

    using ADataType = typename Base::ADataType;
    using BDataType = typename Base::BDataType;
    using CDataType = typename Base::CDataType;

    using AComputeDataType = typename Base::AComputeDataType;
    using BComputeDataType = typename Base::BComputeDataType;
    using CComputeDataType = typename Base::CComputeDataType;
    
    using AVecType = typename Base::AVecType;
    using BVecType = typename Base::BVecType;
    using CVecType = typename Base::CVecType;
    
    static constexpr index_t kM = Base::kM;
    static constexpr index_t kN = Base::kN;
    static constexpr index_t kK = Base::kK;
    static constexpr index_t MRepeat     = MRepeat_;
    static constexpr index_t NRepeat     = NRepeat_;
    static constexpr index_t MInterleave = MInterleave_;
    static constexpr index_t NInterleave = NInterleave_;

    using AWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kAMLane, MInterleave>,
            sequence<kKIter/2, Impl::kABKLane, Impl::kABKPerLane * 2>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 1>>,
        sequence<1, 1, 2, 2>,
        sequence<0, 2, 0, 2>>;

    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<NRepeat, Impl::kBNLane, NInterleave>,
            sequence<kKIter/2, Impl::kABKLane, Impl::kABKPerLane * 2>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 1>>,
        sequence<1, 1, 2, 2>,
        sequence<0, 2, 0, 2>>;

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane, MInterleave>, // <1, 4, 4, 1, 1>
              sequence<NRepeat, Impl::kCNLane, Impl::kCNPerLane, NInterleave>>,         // <2, 16, 1, 1> 
        tuple<sequence<1, 2>>,
        tuple<sequence<2, 1>>,
        sequence<1, 2, 1, 2, 2, 1, 1>,
        sequence<0, 0, 4, 3, 2, 1, 3>>;


    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        using buf_a = thread_buffer<typename Impl::AInt8VecType, kKIter * MRepeat * MInterleave>;
        using buf_b = thread_buffer<typename Impl::BInt8VecType, kKIter * NRepeat * NInterleave>;
        using buf_c = thread_buffer<typename Impl::CInt32VecType, MRepeat * NRepeat * MInterleave * NInterleave>;

        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter/2, 1>{}([&](auto iKIter) {
                            //1 mmac issue
                            Impl{}(reinterpret_cast<buf_c&>(c_vec)
                                       .template get_as<typename Impl::CVecType>()[iMRepeat * NRepeat * MInterleave * NInterleave + 
                                                                                   iNRepeat * MInterleave * NInterleave +
                                                                                   iMInterleave * NInterleave + iNInterleave], 
                                   reinterpret_cast<const buf_a&>(a_vec)
                                       .template get_as<typename Impl::AVecType>()[iMRepeat * MInterleave * kKIter + 
                                                                                   iMInterleave * kKIter + iKIter * 2],
                                   reinterpret_cast<const buf_b&>(b_vec)
                                       .template get_as<typename Impl::BVecType>()[iNRepeat * NInterleave * kKIter + 
                                                                                   iNInterleave * kKIter + iKIter * 2]);
                            //2 mmac issue
                            Impl{}(reinterpret_cast<buf_c&>(c_vec)
                                       .template get_as<typename Impl::CVecType>()[iMRepeat * NRepeat * MInterleave * NInterleave + 
                                                                                   iNRepeat * MInterleave * NInterleave +
                                                                                   iMInterleave * NInterleave + iNInterleave], 
                                   reinterpret_cast<const buf_a&>(a_vec)
                                       .template get_as<typename Impl::AVecType>()[iMRepeat * MInterleave * kKIter + 
                                                                                   iMInterleave * kKIter + iKIter * 2 + 1],
                                   reinterpret_cast<const buf_b&>(b_vec)
                                       .template get_as<typename Impl::BVecType>()[iNRepeat * NInterleave * kKIter + 
                                                                                   iNInterleave * kKIter + iKIter * 2 + 1]);
                        });
                    });
                });
            });
        });
    }

};

template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat,
          index_t NRepeat,
          index_t MInterleave,
          index_t NInterleave,
          index_t kKIter>
struct WarpGemmAttributeInt4MmacIterateK
{
    static_assert(kKIter > 0, "wrong!");

    using Impl = remove_cvref_t<WarpGemmAttributeMmacImpl_>;

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    using AComputeDataType = typename Impl::AComputeDataType;
    using BComputeDataType = typename Impl::BComputeDataType;
    using CComputeDataType = typename Impl::CComputeDataType;

    using AVecType =
        ext_vector_t<int8_t, vector_traits<typename Impl::AInt4VecType>::vector_size * kKIter *
                                 MRepeat * MInterleave>;
    using BVecType =
        ext_vector_t<int8_t, vector_traits<typename Impl::BInt4VecType>::vector_size * kKIter *
                                 NRepeat * NInterleave>;
    using CVecType =
        ext_vector_t<CDataType,
                     vector_traits<typename Impl::CVecType>::vector_size * MRepeat * NRepeat *
                         MInterleave * NInterleave>;

    static constexpr index_t kM = Impl::kM * MRepeat * MInterleave;
    static constexpr index_t kN = Impl::kN * NRepeat * NInterleave;
    static constexpr index_t kK = Impl::kK * kKIter;
    static constexpr index_t kKPerThread = Impl::kABKPerLane * kKIter;

    using AWarpDstrEncoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<MRepeat, Impl::kAMLane, MInterleave>,
                                         sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<0, 1>>,
                                   sequence<1, 1, 2>,
                                   sequence<0, 2, 1>>;

    using BWarpDstrEncoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<NRepeat, Impl::kBNLane, NInterleave>,
                                         sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<0, 1>>,
                                   sequence<1, 1, 2>,
                                   sequence<0, 2, 1>>;

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCMLane, MInterleave, Impl::kCMPerLane>,
              sequence<NRepeat, Impl::kCNLane, NInterleave, Impl::kCN0PerLane, Impl::kCN1PerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 1>>,
        sequence<1, 2, 1, 1, 2, 2, 2>,
        sequence<0, 0, 2, 3, 3, 2, 4>>;

    using CWarpOutputDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCMLane, MInterleave, Impl::kCMPerLane>,
              sequence<NRepeat, Impl::kCN0PerLane, Impl::kCNLane, NInterleave * Impl::kCN1PerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<2, 1>>,
        sequence<1, 2, 1, 1, 2, 2>,
        sequence<0, 0, 2, 3, 1, 3>>;

    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        using buf_a =
            thread_buffer<typename Impl::AInt4VecType, kKIter * MRepeat * MInterleave>;
        using buf_b =
            thread_buffer<typename Impl::BInt4VecType, kKIter * NRepeat * NInterleave>;
        using buf_c = thread_buffer<typename Impl::CVecType,
                                    MRepeat * NRepeat * MInterleave * NInterleave>;

        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            Impl{}(reinterpret_cast<buf_c&>(c_vec)
                                       .template get_as<typename Impl::CVecType>()
                                           [iMRepeat * NRepeat * MInterleave * NInterleave +
                                            iNRepeat * MInterleave * NInterleave +
                                            iMInterleave * NInterleave + iNInterleave],
                                   reinterpret_cast<const buf_a&>(a_vec)
                                       .template get_as<typename Impl::AVecType>()
                                           [iMRepeat * MInterleave * kKIter +
                                            iMInterleave * kKIter + iKIter],
                                   reinterpret_cast<const buf_b&>(b_vec)
                                       .template get_as<typename Impl::BVecType>()
                                           [iNRepeat * NInterleave * kKIter +
                                            iNInterleave * kKIter + iKIter]);
                        });
                    });
                });
            });
        });
    }

    template <typename CWarpTensor>
    CK_TILE_DEVICE auto MakeCOutputLayout(const CWarpTensor& c_warp_tensor) const
    {
        constexpr auto c_warp_output_distribution =
            make_static_tile_distribution(CWarpOutputDstrEncoding{});
        auto c_warp_output_tensor =
            make_static_distributed_tensor<CDataType>(c_warp_output_distribution);

        static_for<0, MRepeat, 1>{}([&](auto iMR) {
            static_for<0, NRepeat, 1>{}([&](auto iNR) {
                static_for<0, MInterleave, 1>{}([&](auto iMI) {
                    static_for<0, NInterleave, 1>{}([&](auto iNI) {
                        static_for<0, Impl::kCN0PerLane, 1>{}([&](auto iCN0) {
                            c_warp_output_tensor.set_y_sliced_thread_data(
                                sequence<iMR, iNR, iMI, 0, iCN0, iNI>{},
                                sequence<1, 1, 1, 1, 1, Impl::kCN1PerLane>{},
                                c_warp_tensor.get_y_sliced_thread_data(
                                    sequence<iMR, iNR, iMI, 0, iNI, iCN0, 0>{},
                                    sequence<1, 1, 1, 1, 1, 1, Impl::kCN1PerLane>{}));
                        });
                    });
                });
            });
        });

        return c_warp_output_tensor;
    }
};

template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat,
          index_t NRepeat,
          index_t MInterleave,
          index_t NInterleave,
          index_t kKIter>
struct WarpGemmAttributeFp8Bf8MmacIterateK
{
    static_assert(kKIter > 0, "wrong!");

    using Impl = remove_cvref_t<WarpGemmAttributeMmacImpl_>;

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    using AScaleType = typename Impl::AScaleType;
    using BScaleType = typename Impl::BScaleType;

    using AComputeDataType = typename Impl::AComputeDataType;
    using BComputeDataType = typename Impl::BComputeDataType;
    using CComputeDataType = typename Impl::CComputeDataType;
    

    using AVecType =
        ext_vector_t<ADataType, vector_traits<typename Impl::AFp8Bf8VecType>::vector_size * kKIter * MRepeat * MInterleave>;
    using BVecType =
        ext_vector_t<BDataType, vector_traits<typename Impl::BFp8Bf8VecType>::vector_size * kKIter * NRepeat * NInterleave>;
    using CVecType =
        ext_vector_t<CDataType, vector_traits<typename Impl::CVecType>::vector_size * MRepeat * NRepeat * MInterleave * NInterleave>;

    using ABufType = thread_buffer<typename Impl::AVecType, kKIter * MRepeat * MInterleave>;
    using BBufType = thread_buffer<typename Impl::BVecType, kKIter * NRepeat * NInterleave>;
    using CBufType = thread_buffer<typename Impl::CVecType, MRepeat * NRepeat * MInterleave * NInterleave>;
    using AScaleBufType = thread_buffer<AScaleType, MRepeat * MInterleave>;
    using BScaleBufType = thread_buffer<BScaleType, NRepeat * NInterleave>;

    static constexpr index_t kM = Impl::kM * MRepeat * MInterleave;
    static constexpr index_t kN = Impl::kN * NRepeat * NInterleave;
    static constexpr index_t kK = Impl::kK * kKIter;
    static constexpr index_t kKPerThread = Impl::kABKPerLane * kKIter;

    using AWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kAMLane, MInterleave>, sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 1>>,
        sequence<1, 1, 2>,
        sequence<0, 2, 1>>;

    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<NRepeat, Impl::kBNLane, NInterleave>, sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 1>>,
        sequence<1, 1, 2>,
        sequence<0, 2, 1>>;
    
    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCMLane, MInterleave, Impl::kCMPerLane>,
              sequence<NRepeat, Impl::kCNLane, NInterleave, Impl::kCN0PerLane, Impl::kCN1PerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 1>>,
        sequence<1, 2, 1, 1, 2, 2, 2>, // <MRepeat, NRepeat, MInterleave, MPerLane, N0PerLane,
                                       // NmmacInterleave, N1PerLane>
        sequence<0, 0, 2, 3, 3, 2, 4>>;

    using CWarpOutputDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCMLane, MInterleave, Impl::kCMPerLane>,
              sequence<NRepeat, Impl::kCN0PerLane, Impl::kCNLane, NInterleave * Impl::kCN1PerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<2, 1>>,
        sequence<1, 2, 1, 1, 2, 2>, // <MRepeat, NRepeat, MInterleave, MPerLane, N0PerLane,
                                    // NmmacInterleave, N1PerLane>
        sequence<0, 0, 2, 3, 1, 3>>;

    using AScaleWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kAMLane, MInterleave>,
              sequence<Impl::kABKLane, 1>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 1>>,
        sequence<1, 1, 2>,
        sequence<0, 2, 1>>;

    using BScaleWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<NRepeat, Impl::kBNLane, NInterleave>,
              sequence<Impl::kABKLane, 1>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 1>>,
        sequence<1, 1, 2>,
        sequence<0, 2, 1>>;

    
    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CBufType& c_buf, const ABufType& a_buf, const BBufType& b_buf, const AScaleBufType& a_scale_buf, const BScaleBufType& b_scale_buf) const
    {
        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            //1 mmac issue
                            Impl{}(c_buf(iMRepeat * NRepeat * MInterleave * NInterleave + 
                                         iNRepeat * MInterleave * NInterleave +
                                         iMInterleave * NInterleave + iNInterleave),
                                   a_buf[iMRepeat * MInterleave * kKIter + iMInterleave * kKIter + 
                                         iKIter],
                                   b_buf[iNRepeat * NInterleave * kKIter + iNInterleave * kKIter + 
                                         iKIter],
                                   a_scale_buf[iMRepeat * MInterleave + iMInterleave],
                                   b_scale_buf[iNRepeat * NInterleave + iNInterleave]);
                        });
                    });
                });
            });
        });
    }

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        using buf_a = thread_buffer<typename Impl::AFp8Bf8VecType, kKIter * MRepeat * MInterleave>;
        using buf_b = thread_buffer<typename Impl::BFp8Bf8VecType, kKIter * NRepeat * NInterleave>;
        using buf_c = thread_buffer<typename Impl::CVecType, MRepeat * NRepeat * MInterleave * NInterleave>;

        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            //1 mmac issue
                            Impl{}(reinterpret_cast<buf_c&>(c_vec)
                                       .template get_as<typename Impl::CVecType>()[iMRepeat * NRepeat * MInterleave * NInterleave +
                                                                                   iNRepeat * MInterleave * NInterleave +
                                                                                   iMInterleave * NInterleave + iNInterleave], 
                                   reinterpret_cast<const buf_a&>(a_vec)
                                       .template get_as<typename Impl::AVecType>()[iMRepeat * MInterleave * kKIter +        // AVecType -> Impl::AFp8Bf8VecType perf model上性能相仿
                                                                                   iMInterleave * kKIter + iKIter],
                                   reinterpret_cast<const buf_b&>(b_vec)
                                       .template get_as<typename Impl::BVecType>()[iNRepeat * NInterleave * kKIter +
                                                                                   iNInterleave * kKIter + iKIter]);
                        });
                    });
                });
            });
        });
    }

    // c_vec = a_vec * b_vec
    // AVecType = ext_vector_t<ADataType, AWarpTensor::get_thread_buffer_size()>;
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        using buf_a = thread_buffer<typename Impl::AVecType, kKIter * MRepeat * MInterleave>;
        using buf_b = thread_buffer<typename Impl::BVecType, kKIter * NRepeat * NInterleave>;
        using buf_c = thread_buffer<typename Impl::CVecType, MRepeat * NRepeat * MInterleave * NInterleave>;

        buf_c c_vec;

        // c += a * b
        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            Impl{}(c_vec.template get_as<typename Impl::CVecType>()[iMRepeat * NRepeat * MInterleave * NInterleave +
                                                                                   iNRepeat * MInterleave * NInterleave +
                                                                                   iMInterleave * NInterleave +
                                                                                   iNInterleave], 
                                   reinterpret_cast<const buf_a&>(a_vec)
                                       .template get_as<typename Impl::AVecType>()[iMRepeat * MInterleave * kKIter +    // AVecType -> Impl::AFp8Bf8VecType perf model上性能相仿
                                                                                   iMInterleave * kKIter +
                                                                                   iKIter],
                                   reinterpret_cast<const buf_b&>(b_vec)
                                       .template get_as<typename Impl::BVecType>()[iNRepeat * NInterleave * kKIter +
                                                                                   iNInterleave * kKIter +
                                                                                   iKIter]);
                        });
                    });
                });
            });
        });

        return reinterpret_cast<CVecType>(c_vec);
    }

};


} // namespace ck_tile
