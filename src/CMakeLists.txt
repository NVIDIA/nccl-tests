#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

set(PERF_COMMON_SOURCES
  common.cu
  util.cu
  timer.cc
)

set(COLLS
  all_reduce
  all_gather
  broadcast
  reduce_scatter
  reduce
  alltoall
  alltoallv
  gather
  scatter
  sendrecv
  hypercube
)

foreach(COLL IN LISTS COLLS)
  add_executable(${COLL}_perf ${COLL}.cu ${PERF_COMMON_SOURCES})
  set_target_properties(${COLL}_perf PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  target_include_directories(${COLL}_perf
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}/os
      ${CMAKE_SOURCE_DIR}/verifiable
  )

  target_link_libraries(${COLL}_perf
    PRIVATE
      nccl_tests_options
      nccl
      CUDA::cudart
      CUDA::cuda_driver
      verifiable
      nccl_test_os
  )

  if(MPI_FOUND)
    target_compile_definitions(${COLL}_perf PRIVATE MPI_SUPPORT)
    target_include_directories(${COLL}_perf PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(${COLL}_perf PRIVATE MPI::MPI_CXX)
  endif()
endforeach()
