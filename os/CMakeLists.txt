#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

# OS abstraction layer for NCCL tests.

if(NCCL_OS_LINUX)
  set(OS_SOURCES linux.cc)
elseif(NCCL_OS_WINDOWS)
  set(OS_SOURCES windows.cc)
else()
  message(FATAL_ERROR "Unsupported OS for nccl-tests: ${CMAKE_SYSTEM_NAME}")
endif()

add_library(nccl_test_os STATIC ${OS_SOURCES})

target_include_directories(nccl_test_os
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(nccl_test_os PRIVATE nccl_tests_options)
