#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

cmake_minimum_required(VERSION 3.25)

project(nccl_tests LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(NCCL_OS_LINUX ON)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(NCCL_OS_WINDOWS ON)
else()
  message(FATAL_ERROR "Unsupported OS: ${CMAKE_SYSTEM_NAME}")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(MPI QUIET)

if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.0)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CUDA_STANDARD 17)
else()
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CUDA_STANDARD 14)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.0)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;90;100;110;120")
  elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
    set(CMAKE_CUDA_ARCHITECTURES "60;70;80;90;100;120")
  elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 11.8)
    set(CMAKE_CUDA_ARCHITECTURES "60;70;80;90")
  else()
    set(CMAKE_CUDA_ARCHITECTURES "60;70;80")
  endif()
endif()

set(NCCL_HOME "" CACHE PATH "Path to NCCL install or build directory")

find_path(NCCL_INCLUDE_DIR
  NAMES nccl.h
  HINTS
    ${NCCL_HOME}/include
    ${NCCL_HOME}/build/include
    $ENV{NCCL_HOME}/include
    $ENV{NCCL_HOME}/build/include
    /usr/include
    /usr/local/include
)

find_library(NCCL_LIBRARY
  NAMES nccl libnccl nccl_static
  HINTS
    ${NCCL_HOME}/lib
    ${NCCL_HOME}/lib/x64
    ${NCCL_HOME}/build/lib
    ${NCCL_HOME}/build/lib/Release
    ${NCCL_HOME}/build/lib/Debug
    ${NCCL_HOME}/build/src
    ${NCCL_HOME}/build/src/Release
    ${NCCL_HOME}/build/src/Debug
    $ENV{NCCL_HOME}/lib
    $ENV{NCCL_HOME}/lib/x64
    $ENV{NCCL_HOME}/build/lib
    $ENV{NCCL_HOME}/build/lib/Release
    $ENV{NCCL_HOME}/build/lib/Debug
    $ENV{NCCL_HOME}/build/src
    $ENV{NCCL_HOME}/build/src/Release
    $ENV{NCCL_HOME}/build/src/Debug
    /usr/lib
    /usr/local/lib
    /usr/lib/x86_64-linux-gnu
)

if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
  message(FATAL_ERROR "Could not find NCCL. Set -DNCCL_HOME=/path/to/nccl or provide NCCL_INCLUDE_DIR and NCCL_LIBRARY.")
endif()

add_library(nccl UNKNOWN IMPORTED)
set_target_properties(nccl PROPERTIES
  IMPORTED_LOCATION "${NCCL_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
)

add_library(nccl_tests_options INTERFACE)
target_compile_definitions(nccl_tests_options INTERFACE
  $<$<BOOL:${NCCL_OS_LINUX}>:NCCL_OS_LINUX>
  $<$<BOOL:${NCCL_OS_LINUX}>:NCCL_GIN_PROXY_ENABLE=1>
  $<$<BOOL:${NCCL_OS_WINDOWS}>:NCCL_OS_WINDOWS>
  $<$<BOOL:${NCCL_OS_WINDOWS}>:WIN32_LEAN_AND_MEAN>
  $<$<BOOL:${NCCL_OS_WINDOWS}>:NOMINMAX>
  $<$<BOOL:${NCCL_OS_WINDOWS}>:NCCL_GIN_PROXY_ENABLE=0>
)

if(MSVC)
  target_compile_options(nccl_tests_options INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:/W3>
    $<$<COMPILE_LANGUAGE:CXX>:/wd4267>
    $<$<COMPILE_LANGUAGE:CXX>:/wd4244>
    $<$<COMPILE_LANGUAGE:CXX>:/wd4996>
    $<$<COMPILE_LANGUAGE:CXX>:/wd4146>
    $<$<COMPILE_LANGUAGE:CXX>:/wd4197>
    $<$<COMPILE_LANGUAGE:CXX>:/wd5105>
    $<$<COMPILE_LANGUAGE:CXX>:/wd4805>
    $<$<COMPILE_LANGUAGE:CXX>:/wd4018>
    $<$<COMPILE_LANGUAGE:CXX>:/FS>
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/FS>
  )
else()
  target_compile_options(nccl_tests_options INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-function>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-sign-compare>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
endif()

add_subdirectory(os)
add_subdirectory(verifiable)
add_subdirectory(src)
