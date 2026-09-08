# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_instance_library_with_arch(device_permute_scale_gfx936_instance
	"gfx936"
	device_permute_scale_1d_fp16_instances.cpp
	device_permute_scale_2d_fp16_instances.cpp
	device_permute_scale_3d_fp16_instances.cpp
	device_permute_scale_4d_fp16_instances.cpp
	device_permute_scale_5d_fp16_instances.cpp
	device_permute_scale_6d_fp16_instances.cpp
	device_permute_scale_1d_fp32_instances.cpp
	device_permute_scale_2d_fp32_instances.cpp
	device_permute_scale_3d_fp32_instances.cpp
	device_permute_scale_4d_fp32_instances.cpp
	device_permute_scale_5d_fp32_instances.cpp
	device_permute_scale_6d_fp32_instances.cpp
	device_permute_scale_6d_fp32_fp8_instances.cpp)

if (TARGET device_permute_scale_gfx936_instance)
	target_include_directories(device_permute_scale_gfx936_instance BEFORE PRIVATE
			${PROJECT_BINARY_DIR}/include
	)
endif ()


