/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef CK_CONFIG_H_IN
#define CK_CONFIG_H_IN

// 为 header-only 的 ck4inductor 安装路径保留并提交此文件。
//
// 常规 CMake 构建会从 config.h.in 生成 config.h，并放到 build/install 的
// include 目录。但 PyTorch TorchInductor 通过 `pip install .` 使用 CK 时，
// 会直接从 Python 包里的 header 编译 CK template，这条路径不会运行 CMake。
// 如果不提交此文件，生成的 template 会在 `#include "ck/config.h"` 处失败。
// 下面的默认宏用于补齐 packaged CK headers 需要的 CMake 默认配置，其中
// CK_EXPERIMENTAL_BIT_INT_EXTENSION 用来暴露 ck::f8_t、ck::bf8_t 和 ck::pk_i4_t。
#ifndef CK_EXPERIMENTAL_BIT_INT_EXTENSION
#define CK_EXPERIMENTAL_BIT_INT_EXTENSION
#endif

#ifndef CK_ENABLE_INT8
#define CK_ENABLE_INT8 "ON"
#endif

#ifndef CK_ENABLE_FP8
#define CK_ENABLE_FP8 "ON"
#endif

#ifndef CK_ENABLE_BF8
#define CK_ENABLE_BF8 "ON"
#endif

#ifndef CK_ENABLE_FP16
#define CK_ENABLE_FP16 "ON"
#endif

#ifndef CK_ENABLE_BF16
#define CK_ENABLE_BF16 "ON"
#endif

#ifndef CK_ENABLE_FP32
#define CK_ENABLE_FP32 "ON"
#endif

#ifndef CK_ENABLE_FP64
#define CK_ENABLE_FP64 "ON"
#endif

#endif // CK_CONFIG_H_IN
