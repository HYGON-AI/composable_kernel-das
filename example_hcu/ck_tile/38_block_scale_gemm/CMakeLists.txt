# Copyright (c) 2026 Hygon Info Technologies Ltd.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx938|gfx946")
    set(HCU_QUANT_GEMM_ARCH_DEFINITIONS)
    if(GPU_TARGETS MATCHES "gfx938")
        list(APPEND HCU_QUANT_GEMM_ARCH_DEFINITIONS CK_TILE_HCU_QUANT_GEMM_GFX938=1)
    endif()
    if(GPU_TARGETS MATCHES "gfx946")
        list(APPEND HCU_QUANT_GEMM_ARCH_DEFINITIONS CK_TILE_HCU_QUANT_GEMM_GFX946=1)
    endif()

    add_example_executable(tile_example_gemm_quant_tensor gemm_quant_tensor.cpp)
    target_compile_definitions(tile_example_gemm_quant_tensor
                               PRIVATE ${HCU_QUANT_GEMM_ARCH_DEFINITIONS})
    add_example_executable(tile_example_gemm_quant_rowcol gemm_quant_rowcol.cpp)
    target_compile_definitions(tile_example_gemm_quant_rowcol
                               PRIVATE ${HCU_QUANT_GEMM_ARCH_DEFINITIONS})
    add_example_executable(tile_example_gemm_quant_blockwise gemm_quant_blockwise.cpp)
    target_compile_definitions(tile_example_gemm_quant_blockwise
                               PRIVATE ${HCU_QUANT_GEMM_ARCH_DEFINITIONS})
else()
    message(DEBUG "Skipping HCU quant GEMM examples for current target")
endif()
