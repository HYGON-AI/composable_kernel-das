#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "fused_moe_quant_mode.hpp"

#include <string>


struct fused_moe_stage1_args
{
    const void* a_ptr;                 // [m, k], input token
    const void* a_scale_ptr;           // [m, 1], token scale
    const void* g_ptr;                 // [e, 2*n, k]
    const void* g_scale_ptr;           // [e, 1, n], gate(up) scale
    const void* g_zp_ptr;              // [e, 2*n, k/group], gate(up) zero-point
    const void* local_expert_mask_ptr; // [e], local_expert_mask_ptr for EP
    void* o_ptr;                       // [m, topk, 2*n]

    void* sorted_token_ids_ptr;  // [max_num_tokens_padded]
    void* sorted_weight_ptr;     // [max_num_tokens_padded]
    void* sorted_expert_ids_ptr; // [(max_num_tokens_padded + block_size - 1) / block_size]
    void* tokens_positions_per_expert_ptr;     // [num_experts*2], represents number of tokens assigned to each expert
    void* num_sorted_tiles_ptr;  // [1], represents total number of tokens after padding

    // void* tmp_out;              // tempary output

    ck_tile::index_t block_m;           // block_m, used to devide the input
    ck_tile::index_t hidden_size;       // k
    ck_tile::index_t intermediate_size; // n / TP, for Gate. and Up, Down is also this value
    ck_tile::index_t num_tokens;        // input number of tokens for current iteration
    ck_tile::index_t num_experts;       // number of groups
    ck_tile::index_t topk;              // need this?

    ck_tile::index_t stride_token;      // for input, stride for each row, should >= hidden_size
    
    ck_tile::index_t block_shape_n;     // quant block n size
    ck_tile::index_t block_shape_k;     // quant block k size
};

struct fused_moe_stage2_args
{
    const void* a_ptr;                 // [m * topk, n], input states from stage1
    const void* a_scale_ptr;           // [m, 1], token scale
    const void* g_ptr;                 // [e, 2*n, k]
    const void* g_scale_ptr;           // [e, 1, n], gate(up) scale
    const void* g_zp_ptr;              // [e, 2*n, k/group], gate(up) zero-point
    const void* local_expert_mask_ptr; // [e], local_expert_mask_ptr for EP
    void* o_ptr;                       // [m, topk, 2*n]

    void* sorted_token_ids_ptr;  // [max_num_tokens_padded]
    void* sorted_weight_ptr;     // [max_num_tokens_padded]
    void* sorted_expert_ids_ptr; // [(max_num_tokens_padded + block_size - 1) / block_size]
    void* tokens_positions_per_expert_ptr;     // [num_experts*2], represents number of tokens assigned to each expert
    void* num_sorted_tiles_ptr;  // [1], represents total number of tokens after padding

    // void* tmp_out;              // tempary output

    ck_tile::index_t block_m;           // block_m, used to devide the input
    ck_tile::index_t hidden_size;       // k
    ck_tile::index_t intermediate_size; // n / TP, for Gate. and Up, Down is also this value
    ck_tile::index_t num_tokens;        // input number of tokens for current iteration
    ck_tile::index_t num_experts;       // number of groups
    ck_tile::index_t topk;              // need this?

    ck_tile::index_t stride_token;      // for output, stride for each row, should >= hidden_size
    
    ck_tile::index_t block_shape_n;     // quant block n size
    ck_tile::index_t block_shape_k;     // quant block k size
};


struct fused_moe_2stage_traits
{
    std::string prec_i;  // input precision
    std::string prec_w;  // weight precision
    std::string prec_o;  // output precision
    std::string prec_st; // token scale data type
    std::string prec_sw; // weight scale data type
    std::string prec_sq; // smooth quant scale
    std::string prec_zp; // zero points data type
    int block_m;
    int activation;  // 0:gelu, 1:silu
    int gate_only;   // 0:g1u1, 1:g1u0
    fused_quant_mode fused_quant; // controls fused quantization strategy
    int solution_id;

    bool local_expert_masking; // if mask experts as local expert
};



float fused_moe_stage1(fused_moe_2stage_traits, fused_moe_stage1_args, const ck_tile::stream_config&);

float fused_moe_stage2(fused_moe_2stage_traits, fused_moe_stage2_args, const ck_tile::stream_config&);