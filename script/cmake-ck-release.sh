#!/bin/bash
# Copyright (c) 2026 Hygon Info Technologies Ltd.
# SPDX-License-Identifier: MIT

rm -f CMakeCache.txt
rm -f *.cmake
rm -rf CMakeFiles

MY_PROJECT_SOURCE=$1

CK_EXTRA_CMAKE_ARGS=()
CK_USING_AICC=OFF

if [ -e /opt/dtk/bin/aicc ]; then
	CK_CMAKE_PREFIX_PATH=/opt/dtk
	CK_CMAKE_CXX_COMPILER=/opt/dtk/bin/aicc
	CK_USING_AICC=ON
	# aicc 使用独立 aillvm；但 DTK 的 hip-config.cmake 默认从 /opt/dtk/llvm
	# 搜索 HIP_CLANG_INCLUDE_PATH，可能误选 hipcc 的 clang17 resource dir，
	# 进而链接错误版本的 libclang_rt.builtins，导致 host _Float16 转换错误。
	# 因此 aicc 分支显式使用 aicc 自己的 resource dir；hipcc 分支保持 HIP 默认策略。
	CK_AICC_RESOURCE_DIR=$(${CK_CMAKE_CXX_COMPILER} -print-resource-dir 2>/dev/null || true)
	if [ -n "${CK_AICC_RESOURCE_DIR}" ] && [ -d "${CK_AICC_RESOURCE_DIR}/include" ]; then
		CK_EXTRA_CMAKE_ARGS+=("-DHIP_CLANG_INCLUDE_PATH=${CK_AICC_RESOURCE_DIR}/include")
	fi
elif [ -e /opt/dtk/llvm/bin/hipcc ]; then
	CK_CMAKE_PREFIX_PATH=/opt/dtk
	CK_CMAKE_CXX_COMPILER=/opt/dtk/llvm/bin/hipcc
else
	CK_CMAKE_PREFIX_PATH=/opt/rocm
	CK_CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc
fi

CK_SAVE_TEMPS=${CK_SAVE_TEMPS:-OFF}
CK_VERBOSE_MAKEFILE=${CK_VERBOSE_MAKEFILE:-ON}
CK_GPU_TARGETS="${CK_GPU_TARGETS:-gfx936;gfx938}"
CK_CODE_OBJECT_VERSION=${CK_CODE_OBJECT_VERSION:-}
CK_EXTRA_CXX_FLAGS=""
if [ "${CK_SAVE_TEMPS}" = "ON" ]; then
	CK_EXTRA_CXX_FLAGS="${CK_EXTRA_CXX_FLAGS} -save-temps=$PWD"
fi
if [ -n "${CK_CODE_OBJECT_VERSION}" ]; then
	CK_EXTRA_CXX_FLAGS="${CK_EXTRA_CXX_FLAGS} -mcode-object-version=${CK_CODE_OBJECT_VERSION}"
fi

# Some DTK packages report a stale HIP CMake package version. Use the installed
# public header for CK's generated version macros without modifying the toolchain.
CK_HIP_VERSION_HEADER=${CK_CMAKE_PREFIX_PATH}/include/hip/hip_version.h
if [ -f "${CK_HIP_VERSION_HEADER}" ]; then
	CK_HIP_VERSION_MAJOR=$(awk '$2 == "HIP_VERSION_MAJOR" {print $3; exit}' "${CK_HIP_VERSION_HEADER}")
	CK_HIP_VERSION_MINOR=$(awk '$2 == "HIP_VERSION_MINOR" {print $3; exit}' "${CK_HIP_VERSION_HEADER}")
	CK_HIP_VERSION_PATCH=$(awk '$2 == "HIP_VERSION_PATCH" {print $3; exit}' "${CK_HIP_VERSION_HEADER}")
	if [ -n "${CK_HIP_VERSION_MAJOR}" ] && [ -n "${CK_HIP_VERSION_MINOR}" ] && \
	   [ -n "${CK_HIP_VERSION_PATCH}" ]; then
		CK_EXTRA_CMAKE_ARGS+=("-DCK_OVERRIDE_HIP_VERSION_MAJOR=${CK_HIP_VERSION_MAJOR}")
		CK_EXTRA_CMAKE_ARGS+=("-DCK_OVERRIDE_HIP_VERSION_MINOR=${CK_HIP_VERSION_MINOR}")
		CK_EXTRA_CMAKE_ARGS+=("-DCK_OVERRIDE_HIP_VERSION_PATCH=${CK_HIP_VERSION_PATCH}")
	fi
fi

# aicc may select an older GCC runtime than the active host compiler. The host
# compiler's libstdc++ linker-script directory supplies the matching nonshared ABI objects.
CK_CXX_STANDARD_LIBRARIES=${CK_CXX_STANDARD_LIBRARIES:-}
if [ "${CK_USING_AICC}" = "ON" ] && [ -z "${CK_CXX_STANDARD_LIBRARIES}" ]; then
	CK_HOST_CXX=${CK_HOST_CXX:-g++}
	CK_HOST_LIBSTDCXX=$(${CK_HOST_CXX} -print-file-name=libstdc++.so 2>/dev/null || true)
	if [ -n "${CK_HOST_LIBSTDCXX}" ] && [ -f "${CK_HOST_LIBSTDCXX}" ]; then
		CK_CXX_STANDARD_LIBRARIES="-L$(dirname "${CK_HOST_LIBSTDCXX}") -lstdc++"
	fi
fi
if [ -n "${CK_CXX_STANDARD_LIBRARIES}" ]; then
	CK_EXTRA_CMAKE_ARGS+=("-DCMAKE_CXX_STANDARD_LIBRARIES=${CK_CXX_STANDARD_LIBRARIES}")
fi

export PYTHONDONTWRITEBYTECODE=1

cmake                                                                                             \
"-DCMAKE_PREFIX_PATH=${CK_CMAKE_PREFIX_PATH}"                                                     \
"-DCMAKE_CXX_COMPILER=${CK_CMAKE_CXX_COMPILER}"                                                   \
"-DCMAKE_CXX_FLAGS=-std=c++17 -O3 -ftemplate-backtrace-limit=0 -fPIE -Wno-gnu-line-marker         \
${CK_EXTRA_CXX_FLAGS}"                                                                            \
"-DCMAKE_BUILD_TYPE=Release"                                                                      \
"-DCPACK_PACKAGING_INSTALL_PREFIX=${CK_CMAKE_PREFIX_PATH}"                                        \
"-DCPACK_SET_DESTDIR=OFF"                                                                         \
"-DBUILD_DEV=OFF"                                                                                 \
"-DGPU_TARGETS=${CK_GPU_TARGETS}"                                                                 \
"-DCMAKE_VERBOSE_MAKEFILE:BOOL=${CK_VERBOSE_MAKEFILE}"                                            \
"-DUSE_BITINT_EXTENSION_INT4=OFF"                                                                 \
"${CK_EXTRA_CMAKE_ARGS[@]}"                                                                       \
"${MY_PROJECT_SOURCE}"
