# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.14)

# Check support for CUDA/HIP in Cmake
set(version 1.0.0)
project(composable_kernel VERSION ${version})

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

enable_testing()

if (DTYPES)
    add_definitions(-DDTYPES)
    if (DTYPES MATCHES "int8")
        add_definitions(-DCK_ENABLE_INT8)
        set(CK_ENABLE_INT8 "ON")
    endif()
    if (DTYPES MATCHES "fp8")
        add_definitions(-DCK_ENABLE_FP8)
        set(CK_ENABLE_FP8 "ON")
    endif()
    if (DTYPES MATCHES "bf8")
        add_definitions(-DCK_ENABLE_BF8)
        set(CK_ENABLE_BF8 "ON")
    endif()
    if (DTYPES MATCHES "fp16")
        add_definitions(-DCK_ENABLE_FP16)
        set(CK_ENABLE_FP16 "ON")
    endif()
    if (DTYPES MATCHES "fp32")
        add_definitions(-DCK_ENABLE_FP32)
        set(CK_ENABLE_FP32 "ON")
    endif()
    if (DTYPES MATCHES "fp64")
        add_definitions(-DCK_ENABLE_FP64)
        set(CK_ENABLE_FP64 "ON")
    endif()
    if (DTYPES MATCHES "bf16")
        add_definitions(-DCK_ENABLE_BF16)
        set(CK_ENABLE_BF16 "ON")
    endif()
    message(STATUS "DTYPES macro set to ${DTYPES}")
else()
    add_definitions(-DCK_ENABLE_INT8 -DCK_ENABLE_FP16 -DCK_ENABLE_FP32 -DCK_ENABLE_FP64 -DCK_ENABLE_BF16 -DCK_ENABLE_FP8 -DCK_ENABLE_BF8)
    set(CK_ENABLE_INT8 "ON")
    set(CK_ENABLE_FP16 "ON")
    set(CK_ENABLE_FP32 "ON")
    set(CK_ENABLE_FP64 "ON")
    set(CK_ENABLE_BF16 "ON")
    set(CK_ENABLE_FP8 "ON")
    set(CK_ENABLE_BF8 "ON")
endif()

set(ROCM_SYMLINK_LIBS OFF)
set(CK_TOOLCHAIN_PREFIXES
    ${CMAKE_PREFIX_PATH}
    /opt/dtk
    /opt/dtk/llvm
    /opt/dtk/hip
    /opt/rocm
    /opt/rocm/llvm
    /opt/rocm/hip
)
list(REMOVE_DUPLICATES CK_TOOLCHAIN_PREFIXES)
find_package(ROCM REQUIRED PATHS ${CK_TOOLCHAIN_PREFIXES})

include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMSetupVersion)
include(ROCMInstallSymlinks)
include(ROCMCreatePackage)
include(CheckCXXCompilerFlag)

rocm_setup_version(VERSION 0.2.3)
include(TargetFlags)
list(APPEND CMAKE_PREFIX_PATH
    ${CMAKE_INSTALL_PREFIX}
    ${CMAKE_INSTALL_PREFIX}/llvm
    ${CMAKE_INSTALL_PREFIX}/hip
    ${CK_TOOLCHAIN_PREFIXES}
)
list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)

# CK version file records the release version and git commit hash identifying
# the HCU CK source tree. Degrade gracefully for release tarballs without .git.
find_package(Git)
set(COMMIT_ID "unknown")
set(CK_PARENT_COMMIT_ID "unknown")
if(GIT_FOUND)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" log -1 --format=%H -- .
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE GIT_COMMIT_ID
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE GIT_LOG_RESULT
        ERROR_QUIET
    )
    if(GIT_LOG_RESULT EQUAL 0 AND GIT_COMMIT_ID)
        set(COMMIT_ID "${GIT_COMMIT_ID}")
    endif()

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE GIT_REPO_HEAD
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE GIT_REV_PARSE_RESULT
        ERROR_QUIET
    )
    if(GIT_REV_PARSE_RESULT EQUAL 0 AND GIT_REPO_HEAD)
        set(CK_PARENT_COMMIT_ID "${GIT_REPO_HEAD}")
    endif()
endif()
configure_file(include/ck/version.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/ck/version.h)

set(USE_BITINT_EXTENSION ON)
option(USE_OPT_NAVI3X, "Whether to enable LDS cumode and Wavefront32 mode for NAVI3X silicons." OFF)

if(USE_BITINT_EXTENSION)
    add_compile_definitions(CK_EXPERIMENTAL_BIT_INT_EXTENSION)
    add_compile_options(-Wno-bit-int-extension)
    message("CK compiled with USE_BITINT_EXTENSION_INT4 set to ${USE_BITINT_EXTENSION_INT4}")
endif()

if(USE_OPT_NAVI3X)
    add_compile_options(-mcumode)
    add_compile_options(-mno-wavefrontsize64)
    message("CK compiled with USE_OPT_NAVI3X set to ${USE_OPT_NAVI3X}")
endif()

set(BUILD_ON_PERF_MODEL OFF)

if(BUILD_ON_PERF_MODEL)
    add_compile_definitions(CK_BUILD_ON_PERF_MODEL)
endif()

## Threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
link_libraries(Threads::Threads)

## C++
enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
message("CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")

## OpenMP
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	# workaround issue hipcc in rocm3.5 cannot find openmp
	set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
	set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
	set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
	set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
	set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
	set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
else()
	find_package(OpenMP REQUIRED)
endif()

message("OpenMP_CXX_LIB_NAMES: ${OpenMP_CXX_LIB_NAMES}")
message("OpenMP_gomp_LIBRARY: ${OpenMP_gomp_LIBRARY}")
message("OpenMP_pthread_LIBRARY: ${OpenMP_pthread_LIBRARY}")
message("OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")

link_libraries(${OpenMP_gomp_LIBRARY})
link_libraries(${OpenMP_pthread_LIBRARY})

## HIP
find_package(HIP REQUIRED)
# Override HIP version in config.h, if necessary.
# The variables set by find_package() can't be overwritten,
# therefore let's use intermediate variables.
set(CK_HIP_VERSION_MAJOR "${HIP_VERSION_MAJOR}")
set(CK_HIP_VERSION_MINOR "${HIP_VERSION_MINOR}")
set(CK_HIP_VERSION_PATCH "${HIP_VERSION_PATCH}")
if( DEFINED CK_OVERRIDE_HIP_VERSION_MAJOR )
    set(CK_HIP_VERSION_MAJOR "${CK_OVERRIDE_HIP_VERSION_MAJOR}")
    message(STATUS "CK_HIP_VERSION_MAJOR overriden with ${CK_OVERRIDE_HIP_VERSION_MAJOR}")
endif()
if( DEFINED CK_OVERRIDE_HIP_VERSION_MINOR )
    set(CK_HIP_VERSION_MINOR "${CK_OVERRIDE_HIP_VERSION_MINOR}")
    message(STATUS "CK_HIP_VERSION_MINOR overriden with ${CK_OVERRIDE_HIP_VERSION_MINOR}")
endif()
if( DEFINED CK_OVERRIDE_HIP_VERSION_PATCH )
    set(CK_HIP_VERSION_PATCH "${CK_OVERRIDE_HIP_VERSION_PATCH}")
    message(STATUS "CK_HIP_VERSION_PATCH overriden with ${CK_OVERRIDE_HIP_VERSION_PATCH}")
endif()
message(STATUS "Build with HIP ${HIP_VERSION}")
link_libraries(hip::device)
add_compile_definitions(__HIP_PLATFORM_HCC__=1)

if(NOT DEFINED GPU_TARGETS OR "${GPU_TARGETS}" STREQUAL "")
    add_compile_definitions(CK_ENABLE_HCU_GFX936_INSTANCES=1)
    add_compile_definitions(CK_ENABLE_HCU_GFX938_INSTANCES=1)
    add_compile_definitions(CK_ENABLE_HCU_GFX946_INSTANCES=1)
else()
    if("gfx936" IN_LIST GPU_TARGETS OR "gfx938" IN_LIST GPU_TARGETS)
        add_compile_definitions(CK_ENABLE_HCU_GFX936_INSTANCES=1)
    endif()
    if("gfx938" IN_LIST GPU_TARGETS)
        add_compile_definitions(CK_ENABLE_HCU_GFX938_INSTANCES=1)
    endif()
    if("gfx946" IN_LIST GPU_TARGETS)
        add_compile_definitions(CK_ENABLE_HCU_GFX946_INSTANCES=1)
    endif()
endif()

# CK config file to record supported datatypes and optional kernel families.
configure_file(include/ck/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/ck/config.h)

## tidy
include(EnableCompilerWarnings)
set(CK_TIDY_ERRORS ERRORS * -readability-inconsistent-declaration-parameter-name)
if(CMAKE_CXX_COMPILER MATCHES ".*hcc" OR CMAKE_CXX_COMPILER MATCHES ".*clang\\+\\+")
    set(CK_TIDY_CHECKS -modernize-use-override -readability-non-const-parameter)
# Enable tidy on hip
elseif(CK_BACKEND STREQUAL "HIP" OR CK_BACKEND STREQUAL "HIPNOGPU")
    set(CK_TIDY_ERRORS ALL)
endif()


include(ClangTidy)
enable_clang_tidy(
    CHECKS
        *
        -abseil-*
        -android-cloexec-fopen
        # Yea we shouldn't be using rand()
        -cert-msc30-c
        -bugprone-exception-escape
        -bugprone-macro-parentheses
        -cert-env33-c
        -cert-msc32-c
        -cert-msc50-cpp
        -cert-msc51-cpp
        -cert-dcl37-c
        -cert-dcl51-cpp
        -clang-analyzer-alpha.core.CastToStruct
        -clang-analyzer-optin.performance.Padding
        -clang-diagnostic-deprecated-declarations
        -clang-diagnostic-extern-c-compat
        -clang-diagnostic-unused-command-line-argument
        -cppcoreguidelines-avoid-c-arrays
        -cppcoreguidelines-avoid-magic-numbers
        -cppcoreguidelines-explicit-virtual-functions
        -cppcoreguidelines-init-variables
        -cppcoreguidelines-macro-usage
        -cppcoreguidelines-non-private-member-variables-in-classes
        -cppcoreguidelines-pro-bounds-array-to-pointer-decay
        -cppcoreguidelines-pro-bounds-constant-array-index
        -cppcoreguidelines-pro-bounds-pointer-arithmetic
        -cppcoreguidelines-pro-type-member-init
        -cppcoreguidelines-pro-type-reinterpret-cast
        -cppcoreguidelines-pro-type-union-access
        -cppcoreguidelines-pro-type-vararg
        -cppcoreguidelines-special-member-functions
        -fuchsia-*
        -google-explicit-constructor
        -google-readability-braces-around-statements
        -google-readability-todo
        -google-runtime-int
        -google-runtime-references
        -hicpp-vararg
        -hicpp-braces-around-statements
        -hicpp-explicit-conversions
        -hicpp-named-parameter
        -hicpp-no-array-decay
        # We really shouldn't use bitwise operators with signed integers, but
        # opencl leaves us no choice
        -hicpp-avoid-c-arrays
        -hicpp-signed-bitwise
        -hicpp-special-member-functions
        -hicpp-uppercase-literal-suffix
        -hicpp-use-auto
        -hicpp-use-equals-default
        -hicpp-use-override
        -llvm-header-guard
        -llvm-include-order
        #-llvmlibc-*
        -llvmlibc-restrict-system-libc-headers
        -llvmlibc-callee-namespace
        -llvmlibc-implementation-in-namespace
        -llvm-else-after-return
        -llvm-qualified-auto
        -misc-misplaced-const
        -misc-non-private-member-variables-in-classes
        -misc-no-recursion
        -modernize-avoid-bind
        -modernize-avoid-c-arrays
        -modernize-pass-by-value
        -modernize-use-auto
        -modernize-use-default-member-init
        -modernize-use-equals-default
        -modernize-use-trailing-return-type
        -modernize-use-transparent-functors
        -performance-unnecessary-value-param
        -readability-braces-around-statements
        -readability-else-after-return
        # we are not ready to use it, but very useful
        -readability-function-cognitive-complexity
        -readability-isolate-declaration
        -readability-magic-numbers
        -readability-named-parameter
        -readability-uppercase-literal-suffix
        -readability-convert-member-functions-to-static
        -readability-qualified-auto
        -readability-redundant-string-init
        # too many narrowing conversions in our code
        -bugprone-narrowing-conversions
        -cppcoreguidelines-narrowing-conversions
        -altera-struct-pack-align
        -cppcoreguidelines-prefer-member-initializer
        ${CK_TIDY_CHECKS}
        ${CK_TIDY_ERRORS}
    HEADER_FILTER
        "\.hpp$"
    EXTRA_ARGS
        -DCK_USE_CLANG_TIDY
)

include(CppCheck)
enable_cppcheck(
    CHECKS
        warning
        style
        performance
        portability
    SUPPRESS
        ConfigurationNotChecked
        constStatement
        duplicateCondition
        noExplicitConstructor
        passedByValue
        preprocessorErrorDirective
        shadowVariable
        unusedFunction
        unusedPrivateFunction
        unusedStructMember
        unmatchedSuppression
    FORCE
    SOURCES
        library/src
    INCLUDE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/library/include
    DEFINE
        CPPCHECK=1
        __linux__=1
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/library/include
    ${PROJECT_BINARY_DIR}/include
    ${HIP_INCLUDE_DIRS}
)


SET(BUILD_DEV ON CACHE BOOL "BUILD_DEV")
if(BUILD_DEV)
    add_compile_options(-Werror)
    add_compile_options(-Weverything)
endif()
message("CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})

file(GLOB_RECURSE INSTANCE_FILES "${PROJECT_SOURCE_DIR}/library/src/tensor_operation_instance_hcu/gpu/device_*_instance.cpp")
file(GLOB dir_list RELATIVE ${PROJECT_SOURCE_DIR}/library/src/tensor_operation_instance_hcu/gpu ${PROJECT_SOURCE_DIR}/library/src/tensor_operation_instance_hcu/gpu/*)
set(CK_DEVICE_INSTANCES)
FOREACH(subdir_path ${dir_list})
    IF(IS_DIRECTORY "${PROJECT_SOURCE_DIR}/library/src/tensor_operation_instance_hcu/gpu/${subdir_path}")
        IF(EXISTS "${PROJECT_SOURCE_DIR}/library/src/tensor_operation_instance_hcu/gpu/${subdir_path}/CMakeLists.txt")
            list(APPEND CK_DEVICE_INSTANCES device_${subdir_path}_instance)
        ENDIF()
    ENDIF()
ENDFOREACH()
add_custom_target(instances DEPENDS utility;${CK_DEVICE_INSTANCES}  SOURCES ${INSTANCE_FILES})

rocm_package_setup_component(profiler
        LIBRARY_NAME composablekernel
        PACKAGE_NAME ckProfiler
)

SET(BUILD_EXAMPLE OFF CACHE BOOL "BUILD_EXAMPLE")
SET(BUILD_TEST OFF CACHE BOOL "BUILD_TEST")

add_subdirectory(library)
if (BUILD_EXAMPLE)
    rocm_package_setup_component(examples
            LIBRARY_NAME composablekernel
            PACKAGE_NAME examples
    )
    add_subdirectory(example_hcu)
endif()
if (BUILD_TEST)
    rocm_package_setup_component(tests
            LIBRARY_NAME composablekernel
            PACKAGE_NAME tests # Prevent -static suffix on package name
    )
    add_subdirectory(test)
endif()
add_subdirectory(profiler)

#Create an interface target for the include only files and call it "composablekernels"
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/composable_kernelConfigVersion.cmake"
    VERSION "${version}"
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/composable_kernelConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
        NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

rocm_install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/composable_kernelConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/composable_kernelConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
)

rocm_install(FILES
    ${PROJECT_BINARY_DIR}/include/ck/version.h
    ${PROJECT_BINARY_DIR}/include/ck/config.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ck/
)

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")
set(CK_DEFAULT_PACKAGE_INSTALL_PREFIX "")
foreach(_ck_prefix ${CMAKE_PREFIX_PATH})
    if(_ck_prefix STREQUAL "/opt/dtk" OR _ck_prefix MATCHES "^/opt/dtk/")
        set(CK_DEFAULT_PACKAGE_INSTALL_PREFIX "/opt/dtk")
        break()
    elseif(_ck_prefix STREQUAL "/opt/rocm" OR _ck_prefix MATCHES "^/opt/rocm/")
        set(CK_DEFAULT_PACKAGE_INSTALL_PREFIX "/opt/rocm")
    endif()
endforeach()

if((NOT DEFINED CPACK_PACKAGING_INSTALL_PREFIX OR "${CPACK_PACKAGING_INSTALL_PREFIX}" STREQUAL "") AND CK_DEFAULT_PACKAGE_INSTALL_PREFIX)
    set(CPACK_PACKAGING_INSTALL_PREFIX "${CK_DEFAULT_PACKAGE_INSTALL_PREFIX}" CACHE PATH "Package install prefix" FORCE)
endif()
if(DEFINED CPACK_PACKAGING_INSTALL_PREFIX AND NOT "${CPACK_PACKAGING_INSTALL_PREFIX}" STREQUAL "")
    set(CPACK_SET_DESTDIR OFF CACHE BOOL "Use CPACK_PACKAGING_INSTALL_PREFIX for package layout" FORCE)
endif()

set(CPACK_DEBIAN_PACKAGE_VERSION "0.2.3~rc1")

rocm_create_package(
    NAME composablekernel
    DESCRIPTION "High Performance Composable Kernel for HCU"
    MAINTAINER "MIOpen Kernels Dev Team"
    LDCONFIG
    HEADER_ONLY
)
