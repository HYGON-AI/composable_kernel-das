// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include <ck_tile/core.hpp>
#include "fused_moegemm.hpp"
#include "fused_moegemm_api_traits.hpp"
#include "fused_moegemm_api_internal.hpp"

// Test-only writer-count instance. Production builds leave it disabled.
#ifndef AITER_CK_MOE_ENABLE_OWNER_DEBUG
#define AITER_CK_MOE_ENABLE_OWNER_DEBUG 0
#endif

// #include "fused_moegemm_fp16_m32.cpp"

// Note: this internal API only declare, not define here, otherwise will block `make -j`
template <typename Traits_>
float fused_moegemm_(const ck_tile::stream_config& s, fused_moegemm_args a);


enum struct SolutionType {
    ONE_STAGE = 0,
    TWO_STAGE = 1,
};

enum struct WeightLoadType {
    DUMMY = 0,
    ALL_REGISTERS = 1,
    ALL_LDS = 2,
    GU_LDS_D_REGS = 3,
    GU_REGS_D_LDS = 4,
};

//represent: block_m0, block_n0, block_k0, block_m1, block_n1, block_k1
enum struct OneStageBlockConfig{
    DUMMY = 0,
    CFG_16_128_128_16_128_128 = 1,
    CFG_16_128_128_16_64_128 = 2,
    CFG_16_64_128_16_128_64 = 3,

    CFG_32_256_32_32_64_256 = 20,
    CFG_32_256X6_OWNER_32_32_64_256 = 22,
#if AITER_CK_MOE_ENABLE_OWNER_DEBUG
    CFG_32_256X6_OWNER_DEBUG_32_32_64_256 = 23,
#endif
};

// solution ID mapping:
// 32 bits: [0~3]:quant_type, [4~7]:weight_load_type, [8~15]:block_config, [16~29]:reserved, [30~31]: solutionType
int get_fused_moe_sol_ID(int quant_type, WeightLoadType ld_type, OneStageBlockConfig block_config, SolutionType sol_type) {
    return (quant_type & 0xF) | ((static_cast<int>(ld_type) & 0xF) << 4) |
           ((static_cast<int>(block_config) & 0xFF) << 8) |
           ((static_cast<int>(sol_type) & 0x3) << 30);
}

OneStageBlockConfig get_block_config_from_sol_ID(int sol_id) {
    int cfg = (sol_id >> 8) & 0xFF;
    switch (cfg) {
    case 1:
        return OneStageBlockConfig::CFG_16_128_128_16_128_128;
    case 2:
        return OneStageBlockConfig::CFG_16_128_128_16_64_128;
    case 3:
        return OneStageBlockConfig::CFG_16_64_128_16_128_64;
    case 20:
        return OneStageBlockConfig::CFG_32_256_32_32_64_256;
    case 22:
        return OneStageBlockConfig::CFG_32_256X6_OWNER_32_32_64_256;
#if AITER_CK_MOE_ENABLE_OWNER_DEBUG
    case 23:
        return OneStageBlockConfig::CFG_32_256X6_OWNER_DEBUG_32_32_64_256;
#endif
    default:
        return OneStageBlockConfig::CFG_16_128_128_16_128_128;
    }
}
WeightLoadType get_weight_ld_type_from_sol_ID(int sol_id) {
    int ld = (sol_id >> 4) & 0xF;
    switch (ld) {
    case 1:
        return WeightLoadType::ALL_REGISTERS;
    case 2:
        return WeightLoadType::ALL_LDS;
    case 3:
        return WeightLoadType::GU_LDS_D_REGS;
    case 4:
        return WeightLoadType::GU_REGS_D_LDS;
    default:
        return WeightLoadType::ALL_REGISTERS;
    }
}



#define W8A8_FUSE_MOE_DISPATCH_(wt_shuffle, prec_i_, block_shape_n_)                 \
    using t_ = fmoe_<prec_i_, ck_tile::int8_t, float, float, float, float, float, ck_tile::int8_t,  \
        S<16, 128, 128>, S<1, 4, 1>, S<16, 32, 128>, S<16, 128, 128>, S<16, 32, 128>, S<4, 4, 4, 4>,  S<2>, S<8, 8, 8, 8>, 1, 2, 1, act_, go_, 4, block_shape_n_, 128, false, true>;   \
        if (wt_shuffle) {                                    \
            return fused_moegemm_wt_shuffle_quant<t_>(s, a); \
        } else {                                             \
            return fused_moegemm_wt_w8a8_quant<t_>(s, a);    \
        }


#define W8A8_QUANT_BLOCK_N_DISPATCH_(wt_shuffle, prec_i_, block_shape_n_)       \
    if (block_shape_n_ == 128) {                                    \
        W8A8_FUSE_MOE_DISPATCH_(wt_shuffle, prec_i_, 128)                       \
    } else if (block_shape_n_ == 64) {                              \
        W8A8_FUSE_MOE_DISPATCH_(wt_shuffle, prec_i_, 64)                        \
    } else if (block_shape_n_ == 32) {                              \
        W8A8_FUSE_MOE_DISPATCH_(wt_shuffle, prec_i_, 32)                        \
    }

#define W8A8_QUANT_PREC_DISPATCH_(wt_shuffle, prec_i_, block_shape_n_)               \
    if (prec_i_ == "fp16") {                                                         \
        W8A8_QUANT_BLOCK_N_DISPATCH_(wt_shuffle, ck_tile::fp16_t, block_shape_n_);   \
    } else if (prec_i_ == "bf16") {                                                  \
        W8A8_QUANT_BLOCK_N_DISPATCH_(wt_shuffle, ck_tile::bf16_t, block_shape_n_);   \
    }

    
#define W4A8_FUSE_MOE_DISPATCH_(prec_i_, block_shape_n_, block_shape_k_)                 \
    using t_ = fmoe_<prec_i_, ck_tile::int8_t, float, float, prec_i_, float, float, ck_tile::int8_t,  \
        S<16, 64, 64>, S<1, 4, 1>, S<16, 16, 64>, S<16, 128, 64>, S<16, 32, 64>, S<2, 2, 2, 2>,  S<1>, S<8, 8, 8, 8>, 1, 1, 1, act_, go_, 5, block_shape_n_, block_shape_k_, false, false>;   \
    return fused_moegemm_wt_w4a8_quant<t_>(s, a);

#define W4A8_QUANT_BLOCK_N_DISPATCH_(prec_i_, block_shape_n_, block_shape_k_)       \
    if (block_shape_k_ == 64) {                                                     \
        W4A8_FUSE_MOE_DISPATCH_(prec_i_, block_shape_n_, 64)                        \
    }

#define W4A8_QUANT_PREC_DISPATCH_(prec_i_, block_shape_n_, block_shape_k_)          \
    if (prec_i_ == "fp16") {                                                        \
        W4A8_QUANT_BLOCK_N_DISPATCH_(ck_tile::fp16_t, block_shape_n_, block_shape_k_);   \
    } else if (prec_i_ == "bf16") {                                                 \
        W4A8_QUANT_BLOCK_N_DISPATCH_(ck_tile::bf16_t, block_shape_n_, block_shape_k_);   \
    }


// #define W4A16_FUSE_MOE_DISPATCH_(prec_i_, block_shape_k_)           \
//     using t_ = fmoe_<prec_i_, ck_tile::int8_t, float, float, prec_i_, float, float, ck_tile::int8_t,    \
//         S<16, 128, 128>, S<1, 4, 1>, S<16, 32, 128>, S<16, 128, 128>, S<16, 32, 128>, S<4, 4, 4, 4>,  S<2>, S<8, 8, 8, 8>, 1, 2, 1, act_, go_, 3, 1, block_shape_k_, false, false>; \
//     return fused_moegemm_wt_w4a16_quant<t_>(s, a);

#define W4A16_FUSE_MOE_DISPATCH_(prec_i_, block_shape_k_)           \
    using t_ = fmoe_<prec_i_, ck_tile::int8_t, float, float, prec_i_, float, float, ck_tile::int8_t,    \
        S<16, 64, 128>, S<1, 4, 1>, S<16, 16, 128>, S<16, 128, 64>, S<16, 32, 64>, S<4, 4, 4, 2>,  S<1>, S<8, 8, 8, 8>, 1, 1, 1, act_, go_, 3, 1, block_shape_k_, false, false>; \
    return fused_moegemm_wt_w4a16_quant<t_>(s, a);

#define W4A16_QUANT_BLOCK_K_DISPATCH_(prec_i_, block_shape_k_)      \
    if (block_shape_k_ == 64) {                                     \
        W4A16_FUSE_MOE_DISPATCH_(prec_i_, 64)                       \
    }

#define W4A16_QUANT_PREC_DISPATCH_(prec_i_, block_shape_k_)               \
    if (prec_i_ == "fp16") {                                              \
        W4A16_QUANT_BLOCK_K_DISPATCH_(ck_tile::fp16_t, block_shape_k_);   \
    } else if (prec_i_ == "bf16") {                                       \
        W4A16_QUANT_BLOCK_K_DISPATCH_(ck_tile::bf16_t, block_shape_k_);   \
    }


template <ck_tile::index_t... Is>
using S = ck_tile::sequence<Is...>;

template <ck_tile::index_t act_>
float fused_moegemm_impl(fused_moegemm_traits t,
                         fused_moegemm_args a,
                         const ck_tile::stream_config& s)
{
//     printf("[fused_moegemm]: 1. t.prec_i = %s, t.prec_w = %s, t.prec_o = %s, t.prec_st = %s, t.prec_sw = %s, t.prec_sq = %s, t.prec_kw = %s, t.prec_zp = %s\n",
//            t.prec_i.c_str(), t.prec_w.c_str(), t.prec_o.c_str(), t.prec_st.c_str(), t.prec_sw.c_str(), t.prec_sq.c_str(), t.prec_kw.c_str(), t.prec_zp.c_str());

    // clang-format off
    float r = -1;

    // 1. quant_type = 0, no quant
    if (t.prec_i == "fp16" && t.prec_w == "fp16" && t.prec_o == "fp32" && t.prec_st == "fp32" &&
           t.prec_sw == "fp32" && t.prec_sq == "fp32" && t.prec_kw == "fp32" && t.gate_only == 0)
    {
        constexpr ck_tile::index_t go_  = 0;

        if (t.block_m == 32) {
            const auto block_config = get_block_config_from_sol_ID(t.solution_id);
            const auto weight_load = get_weight_ld_type_from_sol_ID(t.solution_id);
            if(t.solution_id == 0 ||
               (block_config == OneStageBlockConfig::CFG_32_256_32_32_64_256 &&
                weight_load == WeightLoadType::GU_REGS_D_LDS))
            {
                using t_ = fmoe_<ck_tile::fp16_t, ck_tile::fp16_t, float, float, float, float, float, int, S<32, 256, 32>, S<1, 4, 1>, S<32, 64, 32>, S<32, 64, 256>, S<32, 16, 256>, S<2, 4, 4, 4>,  S<4>, S<8, 8, 8, 4>, 1, 4, 1, act_, go_, 0, 1, 1, false, false>;
                r = fused_moegemm_<t_>(s, a);
            }
            else if(block_config ==
                        OneStageBlockConfig::CFG_32_256X6_OWNER_32_32_64_256 &&
                    weight_load == WeightLoadType::GU_REGS_D_LDS)
            {
                // Topk=1 serial-I owner: a single CTA owns one sorted-token tile
                // and visits all six 256-wide intermediate chunks in fixed order.
                using t_ = fmoe_<ck_tile::fp16_t, ck_tile::fp16_t, float, float, float, float, float, int, S<32, 256, 32>, S<1, 4, 1>, S<32, 64, 32>, S<32, 64, 256>, S<32, 16, 256>, S<2, 4, 4, 4>,  S<4>, S<8, 8, 8, 4>, 1, 4, 1, act_, go_, 0, 1, 1, false, false, 6>;
                r = fused_moegemm_<t_>(s, a);
            }
#if AITER_CK_MOE_ENABLE_OWNER_DEBUG
            else if(block_config ==
                        OneStageBlockConfig::CFG_32_256X6_OWNER_DEBUG_32_32_64_256 &&
                    weight_load == WeightLoadType::GU_REGS_D_LDS)
            {
                // Debug-only owner instance: the final-write planning point counts
                // every valid output element and the AITER boundary requires count=1.
                using t_ = fmoe_<ck_tile::fp16_t, ck_tile::fp16_t, float, float, float, float, float, int, S<32, 256, 32>, S<1, 4, 1>, S<32, 64, 32>, S<32, 64, 256>, S<32, 16, 256>, S<2, 4, 4, 4>,  S<4>, S<8, 8, 8, 4>, 1, 4, 1, act_, go_, 0, 1, 1, false, false, 6, true>;
                r = fused_moegemm_<t_>(s, a);
            }
#endif
            else
            {
                printf("[fused_moegemm][block_m=32]: unsupported solution_id=%d\n",
                       t.solution_id);
            }

        } else if (t.block_m == 16) {

            // 0. no solution_id, use default
            if (t.solution_id == 0) {
                // printf("[fused_moegemm]: 0. use default solution.\n");
                using t_ = fmoe_<ck_tile::fp16_t, ck_tile::fp16_t, float, float, float, float, float, int, S<16, 128, 128>, S<1, 4, 1>, S<16, 32, 128>, S<16, 64, 128>, S<16, 16, 128>, S<4, 16, 16, 4>,  S<2>, S<8, 8, 8, 8>, 1, 2, 1, act_, go_, 0, 1, 1, true, false>;
                r = fused_moegemm_swizzled_lds_<t_>(s, a);

            } else if (get_block_config_from_sol_ID(t.solution_id) == OneStageBlockConfig::CFG_16_128_128_16_128_128 &&
                get_weight_ld_type_from_sol_ID(t.solution_id) == WeightLoadType::ALL_REGISTERS) {
                // printf("[fused_moegemm]: 1. use solution_id = %d, block_config = CFG_16_128_128_16_128_128, weight_ld_type = ALL_REGISTERS\n", t.solution_id);

                using t_ = fmoe_<ck_tile::fp16_t, ck_tile::fp16_t, float, float, float, float, float, int, S<16, 128, 128>, S<1, 4, 1>, S<16, 32, 128>, S<16, 128, 128>, S<16, 32, 128>, S<4, 16, 16, 16>,  S<2>, S<8, 8, 8, 8>, 1, 2, 1, act_, go_, 0, 1, 1, false, false>;
                r = fused_moegemm_wreg_<t_>(s, a);

            }  else if (get_block_config_from_sol_ID(t.solution_id) == OneStageBlockConfig::CFG_16_128_128_16_64_128 &&
                get_weight_ld_type_from_sol_ID(t.solution_id) == WeightLoadType::GU_REGS_D_LDS) {
                // printf("[fused_moegemm]: 2. use solution_id = %d, block_config = CFG_16_128_128_16_64_128, weight_ld_type = GU_REGS_D_LDS\n", t.solution_id);

                using t_ = fmoe_<ck_tile::fp16_t, ck_tile::fp16_t, float, float, float, float, float, int, S<16, 128, 128>, S<1, 4, 1>, S<16, 32, 128>, S<16, 64, 128>, S<16, 16, 128>, S<4, 16, 16, 4>,  S<2>, S<8, 8, 8, 8>, 1, 2, 1, act_, go_, 0, 1, 1, true, false>;
                r = fused_moegemm_swizzled_lds_<t_>(s, a);

            } else {
                printf("[fused_moegemm][block_m=16]: 0. unsupported solution_id=%d\n", t.solution_id);
            }
        } else {
            printf("[fused_moegemm][block_m=16]: 0. unsupported solution_id=%d\n", t.solution_id);
        }
        
        // 4. shuffle gemm0 weight
        // using t_ = fmoe_<ck_tile::fp16_t, ck_tile::fp16_t, float, float, float, float, float, int, S<16, 128, 128>, S<1, 4, 1>, S<16, 32, 128>, S<16, 128, 128>, S<16, 32, 128>, S<4, 4, 4, 16>,  S<2>, S<8, 8, 8, 8>, 1, 2, 1, act_, go_, 0, 1, 1, false, true>;
        // r = fused_moegemm_wreg_shuffle_<t_>(s, a);

        // 5. shuffle gemm0 and gemm1 weight
        // using t_ = fmoe_<ck_tile::fp16_t, ck_tile::fp16_t, float, float, float, float, float, int, S<16, 128, 128>, S<1, 4, 1>, S<16, 32, 128>, S<16, 128, 128>, S<16, 32, 128>, S<4, 4, 4, 4>,  S<2>, S<8, 8, 8, 8>, 1, 2, 1, act_, go_, 0, 1, 1, false, true>;
        // r = fused_moegemm_wt_bd_shuffle_<t_>(s, a);
     }

     else if (t.prec_w == "int8" && t.prec_o == "fp32" && t.prec_st == "fp32" &&
              t.prec_sw == "fp32" && t.prec_sq == "fp32" && t.prec_kw == "fp32" &&
              t.fused_quant == 4 && t.block_m == 16 && t.gate_only == 0)
     {
        constexpr ck_tile::index_t go_  = 0;

        // 1. use_int8_w8a8_block
        if ((128 % a.block_shape_n == 0) && (a.block_shape_k == 128)) {
            W8A8_QUANT_PREC_DISPATCH_(t.use_wt_shuffle, t.prec_i, a.block_shape_n);
        }

        printf("[fused_moegemm]: 0. unsupported fused_quant=%d, block_shape_n=%d, block_shape_k=%d\n", t.fused_quant, a.block_shape_n, a.block_shape_k);

     } else if (t.prec_w == "uint8" && t.prec_o == "fp32" && t.prec_st == "fp32" &&
        t.prec_sq == "fp32" && t.prec_kw == "fp32" && t.prec_zp == "uint8" && t.fused_quant == 3 &&
        t.block_m == 16 && t.gate_only == 0)
    {
        constexpr ck_tile::index_t go_  = 0;

        // 1. use_int8_w4a16_block
        if (a.block_shape_n == 1 && t.prec_i == t.prec_sw) {
            W4A16_QUANT_PREC_DISPATCH_(t.prec_i, a.block_shape_k);
        }

        printf("[fused_moegemm]: 1. unsupported fused_quant=%d, block_shape_n=%d, block_shape_k=%d, t.prec_i=%s, t.prec_sw=%s\n",
            t.fused_quant, a.block_shape_n, a.block_shape_k, t.prec_i.c_str(), t.prec_sw.c_str());

    } else if (t.prec_w == "uint8" && t.prec_o == "fp32" && t.prec_st == "fp32" &&
        t.prec_sq == "fp32" && t.prec_kw == "fp32" && t.prec_zp == "uint8" && t.fused_quant == 5 &&
        t.block_m == 16 && t.gate_only == 0)
    {
        constexpr ck_tile::index_t go_  = 0;

        // printf("[fused_moegemm]: 2. fused_quant=%d, block_shape_n=%d, block_shape_k=%d, t.prec_i=%s, t.prec_sw=%s\n",
        //     t.fused_quant, a.block_shape_n, a.block_shape_k, t.prec_i.c_str(), t.prec_sw.c_str());

        // 1. use_int4_w4a8_block
        if (a.block_shape_n == 1) {
            W4A8_QUANT_PREC_DISPATCH_(t.prec_i, 1, a.block_shape_k);
        }

        printf("[fused_moegemm]: 2. unsupported fused_quant=%d, block_shape_n=%d, block_shape_k=%d, t.prec_i=%s, t.prec_sw=%s\n",
            t.fused_quant, a.block_shape_n, a.block_shape_k, t.prec_i.c_str(), t.prec_sw.c_str());

    }
    else {
        printf("[fused_moegemm]: unsupported fused_moegemm traits, prec_i = %s, prec_w = %s, prec_o = %s, prec_st = %s, prec_sw = %s, prec_sq = %s,"
            " prec_kw = %s, prec_zp = %s, fused_quant=%d, block_m = %d, gate_only = %d, activation = %d\n",
            t.prec_i.c_str(), t.prec_w.c_str(), t.prec_o.c_str(), t.prec_st.c_str(), t.prec_sw.c_str(), t.prec_sq.c_str(), t.prec_kw.c_str(),
            t.prec_zp.c_str(), t.fused_quant, t.block_m, t.gate_only, t.activation);
     }

    // clang-format on
    return r;
}


float fused_moegemm(fused_moegemm_traits t,
                     fused_moegemm_args a,
                     const ck_tile::stream_config& s)
{
    // CK fused MoE traits: 0=GELU, 1=SiLU.
    if(t.activation == 0)
    {
        return fused_moegemm_impl<0>(t, a, s);
    }
    if(t.activation == 1)
    {
        return fused_moegemm_impl<1>(t, a, s);
    }

    printf("[fused_moegemm]: unsupported activation=%d, expected 0 (GELU) or 1 (SiLU)\n",
           t.activation);
    return -1;
}



void fused_moe_find_solutions(fused_moegemm_traits t, fused_moegemm_args a, int* sol_data, int* sol_size)
{
    (void)a;

    if (sol_size == nullptr) {
        printf("[fused_moe_find_solutions]: Error: sol_size is nullptr, return directly!\n");
        return;
    }

    if (t.activation != 0 && t.activation != 1) {
        *sol_size = 0;
        printf("[fused_moe_find_solutions]: unsupported activation=%d, expected 0 (GELU) or 1 (SiLU)\n",
               t.activation);
        return;
    }

    // 1. no quant
    if (t.prec_i == "fp16" && t.prec_w == "fp16" && t.prec_o == "fp32" && t.prec_st == "fp32" &&
           t.prec_sw == "fp32" && t.prec_sq == "fp32" && t.prec_kw == "fp32" && t.gate_only == 0)
    {
        if (t.block_m == 32) {
            if (sol_data == nullptr) {
                *sol_size = 1;
            } else {
                int idx = 0;
                sol_data[idx++] = get_fused_moe_sol_ID(0, WeightLoadType::GU_REGS_D_LDS, OneStageBlockConfig::CFG_32_256_32_32_64_256, SolutionType::ONE_STAGE);
                *sol_size = idx;
            }

        } else if (t.block_m == 16) {
             if (sol_data == nullptr) {
                *sol_size = 2;
            } else {
                int idx = 0;
                // 1. no quant
                sol_data[idx++] = get_fused_moe_sol_ID(0, WeightLoadType::ALL_REGISTERS, OneStageBlockConfig::CFG_16_128_128_16_128_128, SolutionType::ONE_STAGE);
                sol_data[idx++] = get_fused_moe_sol_ID(0, WeightLoadType::GU_REGS_D_LDS, OneStageBlockConfig::CFG_16_128_128_16_64_128, SolutionType::ONE_STAGE);
                *sol_size = idx;
            }
        }
       
    }

    // 2. use_int8_w8a8_block
    else if (t.prec_i == "fp16" && t.prec_w == "int8" && t.prec_o == "fp32" && t.prec_st == "fp32" &&
           t.prec_sw == "fp32" && t.prec_sq == "fp32" && t.prec_kw == "fp32" && t.fused_quant == 4 &&
           t.gate_only == 0){

        if (t.block_m == 16) {
            if (a.block_shape_k == 64) {
                *sol_size = 1;  // 2 stage MoE solution

                if (sol_data != nullptr) {
                    // two stage MoE solution
                    sol_data[0] = get_fused_moe_sol_ID(0, WeightLoadType::DUMMY, OneStageBlockConfig::DUMMY, SolutionType::TWO_STAGE);
                }
            }
            else if (a.block_shape_k == 128) {
                *sol_size = 2;  // 1 stage MoE solutions and 2 stage MoE solution

                if (sol_data != nullptr) {
                    // one stage MoE solution
                    sol_data[0] = get_fused_moe_sol_ID(0, WeightLoadType::DUMMY, OneStageBlockConfig::CFG_16_128_128_16_128_128, SolutionType::ONE_STAGE);
                    // two stage MoE solution
                    sol_data[1] = get_fused_moe_sol_ID(0, WeightLoadType::DUMMY, OneStageBlockConfig::DUMMY, SolutionType::TWO_STAGE);
                }
            } else {
                *sol_size = 0;
                printf("[fused_moe_find_solutions]: unsupported block_shape_k=%d for fused_quant=4.\n", a.block_shape_k);
            }

        } else {
            *sol_size = 1;  // 2 stage MoE solution
            if (sol_data != nullptr) {
                sol_data[0] = get_fused_moe_sol_ID(0, WeightLoadType::DUMMY, OneStageBlockConfig::DUMMY, SolutionType::TWO_STAGE);
            }
        }

    }
    else {
        *sol_size = 0;
        printf("[fused_moe_find_solutions]: unsupported fused_moegemm traits, prec_i = %s, prec_w = %s, prec_o = %s, prec_st = %s, prec_sw = %s, prec_sq = %s,"
            " prec_kw = %s, prec_zp = %s, gate_only = %d, activation = %d\n",
            t.prec_i.c_str(), t.prec_w.c_str(), t.prec_o.c_str(), t.prec_st.c_str(), t.prec_sw.c_str(), t.prec_sq.c_str(), t.prec_kw.c_str(),
            t.prec_zp.c_str(), t.gate_only, t.activation);
    }
}
