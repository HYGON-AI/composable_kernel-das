# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

add_custom_target(example_hcu_grouped_conv_fwd_multiple_d)
foreach(dtype f16 bf16 f32)
    set(target example_hcu_grouped_conv_fwd_multiple_d_${dtype})
    add_example_executable(${target} grouped_conv_fwd_${dtype}.cpp)
    add_dependencies(example_hcu_grouped_conv_fwd_multiple_d ${target})
endforeach()
