# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

add_instance_library_with_arch(device_softmax_gfx936_instance
    "gfx936"
    device_softmax_f16_f16_instance_rank3_reduce1.cpp
    device_softmax_f16_f16_instance_rank3_reduce2.cpp
    device_softmax_f16_f16_instance_rank3_reduce3.cpp
    device_softmax_f16_f16_instance_rank4_reduce1.cpp
    device_softmax_f16_f16_instance_rank4_reduce2.cpp
    device_softmax_f16_f16_instance_rank4_reduce3.cpp
    device_softmax_f16_f16_instance_rank4_reduce4.cpp
    device_softmax_f32_f32_instance_rank3_reduce1.cpp
    device_softmax_f32_f32_instance_rank3_reduce2.cpp
    device_softmax_f32_f32_instance_rank3_reduce3.cpp
    device_softmax_f32_f32_instance_rank4_reduce1.cpp
    device_softmax_f32_f32_instance_rank4_reduce2.cpp
    device_softmax_f32_f32_instance_rank4_reduce3.cpp
    device_softmax_f32_f32_instance_rank4_reduce4.cpp)
