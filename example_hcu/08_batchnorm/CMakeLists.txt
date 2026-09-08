# Copyright (c) 2026 Hygon Info Technologies Ltd.
# SPDX-License-Identifier: MIT

# The NHWC runners live in example/34_batchnorm. Give HCU enrollment unique target names so
# they can coexist with the NCHW runners below and with the regular example build graph.
add_example_executable(
    example_batchnorm_forward_inferring_nhwc
    ${PROJECT_SOURCE_DIR}/example/34_batchnorm/batchnorm_forward_inferring_nhwc.cpp)

add_example_executable(
    example_batchnorm_forward_training_nhwc
    ${PROJECT_SOURCE_DIR}/example/34_batchnorm/batchnorm_forward_training_nhwc.cpp)
add_example_executable(
    example_batchnorm_backward_nhwc
    ${PROJECT_SOURCE_DIR}/example/34_batchnorm/batchnorm_backward_nhwc.cpp)
add_example_executable(example_batchnorm_backward batchnorm_backward_nchw.cpp)
add_example_executable(example_batchnorm_forward_training batchnorm_forward_training_nchw.cpp)
