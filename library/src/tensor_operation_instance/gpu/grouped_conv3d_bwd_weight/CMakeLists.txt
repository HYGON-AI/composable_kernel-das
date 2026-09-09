# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Modified by Hygon Information Technology Co., Ltd.

add_instance_library(device_grouped_conv3d_bwd_weight_instance
    device_grouped_conv3d_bwd_weight_xdl_gndhwc_gkzyxc_gndhwk_f16_instance.cpp
    device_grouped_conv3d_bwd_weight_xdl_gndhwc_gkzyxc_gndhwk_f32_instance.cpp
    device_grouped_conv3d_bwd_weight_xdl_gndhwc_gkzyxc_gndhwk_bf16_instance.cpp
)
