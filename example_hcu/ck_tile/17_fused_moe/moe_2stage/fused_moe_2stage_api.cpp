#include "fused_moe_2stage.hpp"
#include "fused_moegemm_2stage.hpp"


float fused_moe_stage1(fused_moe_2stage_traits traits, fused_moe_stage1_args args, const ck_tile::stream_config& s)
{
    auto t1 = fused_moegemm_2stage_traits{traits.prec_i,
                                        traits.prec_w,
                                        traits.prec_o,
                                        traits.prec_st,
                                        traits.prec_sw,
                                        traits.prec_sq,
                                        traits.prec_zp,
                                        traits.block_m,
                                        traits.activation,
                                        traits.gate_only,
                                        traits.fused_quant,
                                        traits.solution_id};
    
    auto a1 = fused_moegemm_stage1_args{
        args.a_ptr,                 // const void* a_ptr;
        args.a_scale_ptr,           // const void* a_scale_ptr;
        args.g_ptr,                 // const void* g_ptr;
        args.g_scale_ptr,           // const void* g_scale_ptr;
        args.g_zp_ptr,              // const void* g_zp_ptr;
        args.local_expert_mask_ptr, // const void* local_expert_mask_ptr;
        args.o_ptr,                 // void* o_ptr;
        args.sorted_token_ids_ptr,  // void* sorted_token_ids_ptr;  
        args.sorted_weight_ptr,     // void* sorted_weight_ptr;     
        args.sorted_expert_ids_ptr, // void* sorted_expert_ids_ptr; 
        args.tokens_positions_per_expert_ptr,     // void* tokens_positions_per_expert_ptr;
        args.num_sorted_tiles_ptr,  // void* num_sorted_tiles_ptr;  
        args.block_m,               // index_t block_m;           
        args.hidden_size,           // index_t hidden_size;       
        args.intermediate_size,     // index_t intermediate_size; 
        args.num_tokens,            // index_t num_tokens;        
        args.num_experts,           // index_t num_experts;       
        args.topk,                  // index_t topk;              
        args.stride_token,          // index_t stride_token;      
        args.block_shape_n,         // index_t quant block n size;     
        args.block_shape_k          // index_t quant block k size;     
    };


    auto s_sub = ck_tile::stream_config{s.stream_id_, false, s.log_level_, 0, 1};

    float r0 = -1;
    float r = ck_tile::launch_kernel(
        s,
        [=, &r0](const ck_tile::stream_config&) { r0 = fused_moegemm_stage1(t1, a1, s_sub); });

    return r;
}


float fused_moe_stage2(fused_moe_2stage_traits traits, fused_moe_stage2_args args, const ck_tile::stream_config& s)
{
    auto t1 = fused_moegemm_2stage_traits{traits.prec_i,
                                        traits.prec_w,
                                        traits.prec_o,
                                        traits.prec_st,
                                        traits.prec_sw,
                                        traits.prec_sq,
                                        traits.prec_zp,
                                        traits.block_m,
                                        traits.activation,
                                        traits.gate_only,
                                        traits.fused_quant,
                                        traits.solution_id};
    
    auto a1 = fused_moegemm_stage2_args{
        args.a_ptr,                 // const void* a_ptr;
        args.a_scale_ptr,           // const void* a_scale_ptr;
        args.g_ptr,                 // const void* g_ptr;
        args.g_scale_ptr,           // const void* g_scale_ptr;
        args.g_zp_ptr,              // const void* g_zp_ptr;
        args.local_expert_mask_ptr, // const void* local_expert_mask_ptr;
        args.o_ptr,                 // void* o_ptr;
        args.sorted_token_ids_ptr,  // void* sorted_token_ids_ptr;
        args.sorted_weight_ptr,     // void* sorted_weight_ptr;
        args.sorted_expert_ids_ptr, // void* sorted_expert_ids_ptr;
        args.tokens_positions_per_expert_ptr,     // void* tokens_positions_per_expert_ptr;
        args.num_sorted_tiles_ptr,  // void* num_sorted_tiles_ptr;
        args.block_m,               // index_t block_m;
        args.hidden_size,           // index_t hidden_size;
        args.intermediate_size,     // index_t intermediate_size;
        args.num_tokens,            // index_t num_tokens;
        args.num_experts,           // index_t num_experts;
        args.topk,                  // index_t topk;
        args.stride_token,          // index_t stride_token;
        args.block_shape_n,         // index_t quant block n size;
        args.block_shape_k          // index_t quant block k size;
    };


    auto s_sub = ck_tile::stream_config{s.stream_id_, false, s.log_level_, 0, 1};

    float r0 = -1;
    float r = ck_tile::launch_kernel(
        s,
        [=, &r0](const ck_tile::stream_config&) { r0 = fused_moegemm_stage2(t1, a1, s_sub); });

    return r;
}