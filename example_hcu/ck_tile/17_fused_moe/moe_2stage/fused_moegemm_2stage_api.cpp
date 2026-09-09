// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include <ck_tile/core.hpp>
#include "fused_moegemm_2stage.hpp"
#include "fused_moe_groupgemm.hpp"


// TODO: configure kernel parameters based on traits
float fused_moegemm_stage1(fused_moegemm_2stage_traits t, fused_moegemm_stage1_args args, const ck_tile::stream_config& s)
{
    float ave_time = 0.001;

    using Row   = ck_tile::tensor_layout::gemm::RowMajor;
    using Col   = ck_tile::tensor_layout::gemm::ColumnMajor;

    if(t.fused_quant == fused_quant_mode::none && t.prec_i == "fp16" && t.prec_w == "fp16" && t.prec_o == "fp16")
    {
        using GemmConfigHalf = GemmConfigComputeV4<ck_tile::half_t>;
        using Types = Gemm1TypeConfigHalf;

        static_assert(GemmConfigHalf::Persistent == true,
                      "GemmConfigHalf::Persistent should be true");


        int group_count = args.num_experts;
        bool splitk     = false;

        ave_time = moe_grouped_gemm1_tileloop<GemmConfigHalf,
                                            Row,
                                            Col,
                                            Row,
                                            Types::ADataType,
                                            Types::BDataType,
                                            Types::AccDataType,
                                            Types::CDataType,
                                            Types::AScaleDataType,
                                            Types::BScaleDataType>(s, group_count, args, splitk);

    }
    else if (t.fused_quant == fused_quant_mode::fp8_w8a8_block && t.prec_i == "fp8" && t.prec_w == "fp8" && t.prec_o == "fp16")
    {
        // printf("########## fused_moegemm_stage1 using fp8_w8a8_block fused quantization.\n");
        
        using GemmConfigfp8 = GemmConfigComputeV5<ck_tile::fp8_t>;
        using Types = Gemm1TypeConfigFp8W8A8;

        int group_count = args.num_experts;
        bool splitk     = false;

        ave_time = moe_grouped_gemm1_tileloop<GemmConfigfp8,
                                            Row,
                                            Col,
                                            Row,
                                            Types::ADataType,
                                            Types::BDataType,
                                            Types::AccDataType,
                                            Types::CDataType,
                                            Types::AScaleDataType,
                                            Types::BScaleDataType>(s, group_count, args, splitk);
    }
    else if (t.fused_quant == fused_quant_mode::int8_w8a8_block && t.prec_w == "int8" && t.prec_o == "fp16")
    {
        // printf("########## fused_moegemm_stage1 using int8_w8a8_block fused quantization.\n");

        //1. 外部quantization，输入数据为int8
        if (t.prec_i == "int8") {

            using Types = Gemm1TypeConfigInt8W8A8;
            int group_count = args.num_experts;
            bool splitk     = false;

            if (args.block_shape_k == 64) {
                using GemmConfigInt8 = GemmConfigComputeBlockShape64<ck_tile::int8_t>;
                ave_time = moe_grouped_gemm1_tileloop<GemmConfigInt8,
                                                Row,
                                                Col,
                                                Row,
                                                Types::ADataType,
                                                Types::BDataType,
                                                Types::AccDataType,
                                                Types::CDataType,
                                                Types::AScaleDataType,
                                                Types::BScaleDataType>(s, group_count, args, splitk);
            } else {
                using GemmConfigInt8 = GemmConfigComputeV6<ck_tile::int8_t>;
                ave_time = moe_grouped_gemm1_tileloop<GemmConfigInt8,
                                                Row,
                                                Col,
                                                Row,
                                                Types::ADataType,
                                                Types::BDataType,
                                                Types::AccDataType,
                                                Types::CDataType,
                                                Types::AScaleDataType,
                                                Types::BScaleDataType>(s, group_count, args, splitk);
            }

        } else {
            printf("### fused_moegemm_stage1 unsupported input precision for int8_w8a8_block fused quantization.\n");
        }
        
    }

    // printf("### fused_moegemm_stage1 is called with time %f.\n", ave_time);
    return ave_time;
}


float fused_moegemm_stage2(fused_moegemm_2stage_traits t, fused_moegemm_stage2_args args, const ck_tile::stream_config& s)
{
    float ave_time = -0.001;

    using Row   = ck_tile::tensor_layout::gemm::RowMajor;
    using Col   = ck_tile::tensor_layout::gemm::ColumnMajor;

    if(t.fused_quant == fused_quant_mode::none && t.prec_i == "fp16" && t.prec_w == "fp16" && t.prec_o == "fp32")
    {
        using GemmConfigHalf = GemmConfigComputeV4<ck_tile::half_t>;
        using Types = GemmTypeConfig<ck_tile::half_t>;

        int group_count = args.num_experts;
        bool splitk     = false;

        ave_time = moe_grouped_gemm2_tileloop<GemmConfigHalf,
                                            Row,
                                            Col,
                                            Row,
                                            Types::ADataType,
                                            Types::BDataType,
                                            Types::AccDataType,
                                            Types::CDataType,
                                            Types::AScaleDataType,
                                            Types::BScaleDataType>(s, group_count, args, splitk);

    } else if (t.fused_quant == fused_quant_mode::fp8_w8a8_block && t.prec_i == "fp8" && t.prec_w == "fp8" && t.prec_o == "fp32")
    {
        // printf("########## fused_moegemm_stage2 using fp8_w8a8_block fused quantization.\n");
        
        using GemmConfigfp8 = GemmConfigComputeV5<ck_tile::fp8_t>;
        using Types = Gemm2TypeConfigFp8W8A8;

        int group_count = args.num_experts;
        bool splitk     = false;

        ave_time = moe_grouped_gemm2_tileloop<GemmConfigfp8,
                                            Row,
                                            Col,
                                            Row,
                                            Types::ADataType,
                                            Types::BDataType,
                                            Types::AccDataType,
                                            Types::CDataType,
                                            Types::AScaleDataType,
                                            Types::BScaleDataType>(s, group_count, args, splitk);
    }
    else if (t.fused_quant == fused_quant_mode::int8_w8a8_block && t.prec_w == "int8" && t.prec_o == "fp32")
    {
        // printf("########## fused_moegemm_stage2 using int8_w8a8_block fused quantization.\n");
        
        //1. 外部quantization，输入数据为int8
        if (t.prec_i == "int8") {

            using Types = Gemm2TypeConfigInt8W8A8;
            int group_count = args.num_experts;
            bool splitk     = false;

            if (args.block_shape_n == 64) {
                using GemmConfigInt8 = GemmConfigComputeBlockShape64<ck_tile::int8_t>;
                ave_time = moe_grouped_gemm2_tileloop<GemmConfigInt8,
                                                    Row,
                                                    Col,
                                                    Row,
                                                    Types::ADataType,
                                                    Types::BDataType,
                                                    Types::AccDataType,
                                                    Types::CDataType,
                                                    Types::AScaleDataType,
                                                    Types::BScaleDataType>(s, group_count, args, splitk);

            } else {
                using GemmConfigInt8 = GemmConfigComputeV6<ck_tile::int8_t>;
                ave_time = moe_grouped_gemm2_tileloop<GemmConfigInt8,
                                                    Row,
                                                    Col,
                                                    Row,
                                                    Types::ADataType,
                                                    Types::BDataType,
                                                    Types::AccDataType,
                                                    Types::CDataType,
                                                    Types::AScaleDataType,
                                                    Types::BScaleDataType>(s, group_count, args, splitk);
            }


        } else {
            printf("### fused_moegemm_stage2 unsupported input precision for int8_w8a8_block fused quantization.\n");
        }

    }

    // printf("### fused_moegemm_stage2 is called with time %f.\n", ave_time);
    return ave_time;
}
