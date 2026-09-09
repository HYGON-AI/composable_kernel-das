# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd.

# ckProfiler
set(PROFILER_SOURCES
    profiler.cpp
    profile_grouped_conv_fwd.cpp
    profile_grouped_conv_bwd_data.cpp
    profile_grouped_conv_bwd_weight.cpp
    profile_layout_transform.cpp
    profile_grouped_conv_fwd_bias_relu.cpp
    profile_grouped_conv_fwd_bias_add_relu.cpp
)

set(PROFILER_EXECUTABLE ckProfiler)

add_executable(${PROFILER_EXECUTABLE} ${PROFILER_SOURCES})
target_compile_options(${PROFILER_EXECUTABLE} PRIVATE -Wno-global-constructors)

target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE utility)

if(TARGET device_operations)
    target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_operations)
else()
    foreach(PROFILER_INSTANCE_TARGET
        device_layout_transform_instance
        device_grouped_conv2d_fwd_gfx928_instance
        device_grouped_conv2d_bwd_gfx928_instance
        device_grouped_conv2d_wrw_gfx928_instance
        device_grouped_conv2d_fwd_gfx936_instance
        device_grouped_conv2d_bwd_gfx936_instance
        device_grouped_conv3d_bwd_gfx936_instance
        device_grouped_conv2d_wrw_gfx936_instance
        device_grouped_conv3d_wrw_gfx936_instance
        device_fused_conv2d_cba_gfx928_instance
        device_fused_conv2d_cbra_gfx928_instance
        device_fused_conv2d_cba_gfx936_instance
        device_fused_conv2d_cbra_gfx936_instance
        device_grouped_conv2d_fwd_bias_relu_gfx928_instance
        device_grouped_conv2d_fwd_bias_add_relu_gfx928_instance
        device_grouped_conv3d_fwd_gfx928_instance
        device_fused_conv2d_cba_gfx938_instance
        device_fused_conv2d_cbra_gfx938_instance)
        if(TARGET ${PROFILER_INSTANCE_TARGET})
            target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE ${PROFILER_INSTANCE_TARGET})
        endif()
    endforeach()
endif()
rocm_install(TARGETS ${PROFILER_EXECUTABLE} COMPONENT profiler)
