# Copyright (c) 2026 Hygon Info Technologies Ltd.
# SPDX-License-Identifier: MIT

add_instance_library_with_arch(device_fused_conv2d_cbra_gfx928_instance
  "gfx928;gfx92a;gfx936;gfx938"
  nhwgc_gkyxc_nhwgk/fused_conv2d_v1_f16_bias_f16_f1x1s1p0_instance.cpp
)

if(TARGET device_fused_conv2d_cbra_gfx928_instance)
  target_compile_options(device_fused_conv2d_cbra_gfx928_instance PRIVATE -mls-ooo)
endif()
