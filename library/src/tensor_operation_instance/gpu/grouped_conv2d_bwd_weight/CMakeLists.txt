# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_instance_library(device_grouped_conv2d_bwd_weight_instance
    device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_f16_instance.cpp
    device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_f32_instance.cpp
    device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_bf16_instance.cpp
)

