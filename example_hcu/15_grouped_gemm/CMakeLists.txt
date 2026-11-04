# Copyright (c) 2026 Hygon Info Technologies Ltd.
# SPDX-License-Identifier: MIT
# Modified by Hygon Information Technology Co., Ltd.
# hcu grouped gemm tile loop examples
add_example_executable(example_hcu_grouped_gemm_multiple_d_xdl_fp16
                       grouped_gemm_multiple_d_xdl_fp16.cpp)

add_test(NAME example_hcu_grouped_gemm_multiple_d_xdl_fp16_smoke
    COMMAND $<TARGET_FILE:example_hcu_grouped_gemm_multiple_d_xdl_fp16>
            1 1 0 64,127,255 252,252,252 4608,4608,4608 4608,4608,4608 4608,4608,4608 252,252,252
)
set_tests_properties(example_hcu_grouped_gemm_multiple_d_xdl_fp16_smoke
    PROPERTIES LABELS "hcu;grouped_gemm;multi_d;smoke"
)

add_example_executable(example_hcu_grouped_gemm_xdl_splitk_fp16 grouped_gemm_xdl_splitk_fp16.cpp)
set_tests_properties(example_hcu_grouped_gemm_xdl_splitk_fp16
    PROPERTIES LABELS "hcu;grouped_gemm_xdl_1103")
add_example_executable(example_hcu_grouped_gemm_multiple_d_splitk_xdl_fp16
    grouped_gemm_multiple_d_splitk_xdl_fp16.cpp)
set_tests_properties(example_hcu_grouped_gemm_multiple_d_splitk_xdl_fp16
    PROPERTIES LABELS "hcu;grouped_gemm_xdl_1103")
add_example_executable_no_testing(example_hcu_grouped_gemm_splitk_contract
    grouped_gemm_splitk_contract.cpp)
foreach(dtype fp16 bf16 fp32)
    add_test(NAME example_hcu_grouped_gemm_splitk_contract_${dtype}
        COMMAND $<TARGET_FILE:example_hcu_grouped_gemm_splitk_contract> ${dtype})
    set_tests_properties(example_hcu_grouped_gemm_splitk_contract_${dtype}
        PROPERTIES LABELS "hcu;grouped_gemm_xdl_1103")
endforeach()

foreach(batch 1 2)
    add_test(NAME example_hcu_grouped_gemm_splitk_batch_${batch}
        COMMAND $<TARGET_FILE:example_hcu_grouped_gemm_xdl_splitk_fp16> ${batch})
    set_tests_properties(example_hcu_grouped_gemm_splitk_batch_${batch}
        PROPERTIES LABELS "hcu;grouped_gemm_xdl_1103")
endforeach()
foreach(batch 1 128)
    add_test(NAME example_hcu_grouped_gemm_splitk_multi_d_batch_${batch}
        COMMAND $<TARGET_FILE:example_hcu_grouped_gemm_multiple_d_splitk_xdl_fp16> ${batch})
    set_tests_properties(example_hcu_grouped_gemm_splitk_multi_d_batch_${batch}
        PROPERTIES LABELS "hcu;grouped_gemm_xdl_1103")
endforeach()
add_test(NAME example_hcu_grouped_gemm_splitk_timing
    COMMAND $<TARGET_FILE:example_hcu_grouped_gemm_xdl_splitk_fp16> 4 timing)
set_tests_properties(example_hcu_grouped_gemm_splitk_timing
    PROPERTIES LABELS "hcu;grouped_gemm_xdl_1103")
add_example_executable(example_hcu_grouped_gemm_splitk_no_timing
    grouped_gemm_splitk_no_timing.cpp)
# Legacy timing helpers leave callback parameters unused with CK_TIME_KERNEL=0.
# Keep the exception local to the timing-disabled contract test.
target_compile_options(example_hcu_grouped_gemm_splitk_no_timing PRIVATE -Wno-unused-parameter)
set_tests_properties(example_hcu_grouped_gemm_splitk_no_timing
    PROPERTIES LABELS "hcu;grouped_gemm_xdl_1103")
