#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

set(VERIFIABLE_LIB_SOURCES
  verifiable.cu
)

add_library(verifiable STATIC ${VERIFIABLE_LIB_SOURCES})

target_link_libraries(verifiable
  PUBLIC
    nccl_tests_options
    nccl
    CUDA::cudart
)

target_include_directories(verifiable
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
