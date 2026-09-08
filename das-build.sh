#!/bin/bash
# Copyright (c) 2026 Hygon Info Technologies Ltd.
# SPDX-License-Identifier: MIT


SRC_DIR=$(realpath $(dirname $0))
BUILD_DIR=$SRC_DIR/build
BUILD_TYPE=${BUILD_TYPE:=Release}
BUILD_CPUS=${BUILD_CPUS:=$(nproc)}
PACK_TYPE=${PACK_TYPE^^}

CK_EXTRA_CMAKE_ARGS=""

if [ -n "${CK_CMAKE_PREFIX_PATH:-}" ] && [ -n "${CK_CMAKE_CXX_COMPILER:-}" ]; then
    :
elif [ -e /opt/dtk/bin/aicc ]; then
    CK_CMAKE_PREFIX_PATH=/opt/dtk
    CK_CMAKE_CXX_COMPILER=/opt/dtk/bin/aicc
    CK_AICC_RESOURCE_DIR=$(${CK_CMAKE_CXX_COMPILER} -print-resource-dir 2>/dev/null || true)
    if [ -n "${CK_AICC_RESOURCE_DIR}" ] && [ -d "${CK_AICC_RESOURCE_DIR}/include" ]; then
        CK_EXTRA_CMAKE_ARGS="${CK_EXTRA_CMAKE_ARGS} -D HIP_CLANG_INCLUDE_PATH=${CK_AICC_RESOURCE_DIR}/include"
    fi
elif [ -e /opt/dtk/llvm/bin/hipcc ]; then
    CK_CMAKE_PREFIX_PATH=/opt/dtk
    CK_CMAKE_CXX_COMPILER=/opt/dtk/llvm/bin/hipcc
else
    CK_CMAKE_PREFIX_PATH=/opt/rocm
    CK_CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc
fi

case "${CK_CMAKE_PREFIX_PATH}" in
    /opt/dtk*)
        ARTIFACTS_DIR=${ARTIFACTS_DIR:=$SRC_DIR/../my_dtk}
        ;;
    *)
        ARTIFACTS_DIR=${ARTIFACTS_DIR:=$SRC_DIR/../my_rocm}
        ;;
esac
CPACK_INSTALL_PREFIX=${CPACK_INSTALL_PREFIX:=${CK_CMAKE_PREFIX_PATH}}

# FIXME: a workaround for adding 512/768 vgpr support option,
# since cmake will generate wrong option with target_compile_options()/add_compile_options()
# remove it after figure out how to do it in CMakeLists.txt

# resolve rocm-6.3.3 not set HIP_LIB_PATH issue with hipcc
export HIP_LIB_PATH=${CK_CMAKE_PREFIX_PATH}/lib
export HIPCC_COMPILE_FLAGS_APPEND="-mllvm -stream-unfolded-args-in-metadata"
export LD_LIBRARY_PATH=$ARTIFACTS_DIR/lib:${CK_CMAKE_PREFIX_PATH}/lib:${CK_CMAKE_PREFIX_PATH}/hip/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$ARTIFACTS_DIR/lib:${CK_CMAKE_PREFIX_PATH}/lib:${CK_CMAKE_PREFIX_PATH}/hip/lib:$LIBRARY_PATH

[ -d $BUILD_DIR ] || mkdir -p $BUILD_DIR
cd $BUILD_DIR
cmake $SRC_DIR \
      -DCMAKE_PREFIX_PATH="${ARTIFACTS_DIR};${CK_CMAKE_PREFIX_PATH};${CK_CMAKE_PREFIX_PATH}/llvm;${CK_CMAKE_PREFIX_PATH}/hip" \
      -DCMAKE_INSTALL_PREFIX=${ARTIFACTS_DIR} \
      -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -DCPACK_PACKAGING_INSTALL_PREFIX=${CPACK_INSTALL_PREFIX} \
      -DCPACK_SET_DESTDIR=OFF \
      -DCMAKE_CXX_COMPILER=${CK_CMAKE_CXX_COMPILER} \
      -DBUILD_EXAMPLE=OFF \
      -DBUILD_TEST=OFF \
      -DCPACK_GENERATOR=${PACK_TYPE} \
      -DUSE_BITINT_EXTENSION=ON \
      ${CK_EXTRA_CMAKE_ARGS} \
      || exit 1

if [ "${CK_DAS_CONFIGURE_ONLY:-OFF}" = "ON" ]; then
    exit 0
fi

make -j ${BUILD_CPUS} || exit 1
make install || exit 1

if [ -n "$CPACK_INSTALL_PREFIX" ] && [ -n "$PACK_TYPE" ]; then
    make package -j ${BUILD_CPUS} || exit 1
fi
