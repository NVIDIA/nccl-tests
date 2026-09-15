if(NCCL_OS_WINDOWS)
    return()
endif()

if(NOT MPI_FOUND)
    return()
endif()

# The GIN device API requirements these benchmarks fill in are only available
# from NCCL 2.30.7 onwards.
set(DEVICE_API_GIN_MIN_NCCL_VERSION 23007)
nccl_older_than(DEVICE_API_GIN_NCCL_TOO_OLD ${DEVICE_API_GIN_MIN_NCCL_VERSION})

if(DEVICE_API_GIN_NCCL_TOO_OLD)
    message(STATUS "Skipping GIN device API performance tests: NCCL version code "
                   "${NCCL_HEADER_VERSION} is older than ${DEVICE_API_GIN_MIN_NCCL_VERSION}")
    return()
endif()

function(gin_perf_target_includes TARGET_NAME)
    target_include_directories(${TARGET_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
    )
endfunction()

function(gin_perf_target_compile_options TARGET_NAME CATEGORY_DIR)
    if(CATEGORY_DIR STREQUAL "throughput")
        target_compile_options(${TARGET_NAME}
            PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-maxrregcount=60>"
        )
    endif()
endfunction()

function(add_gin_perf_category CATEGORY_DIR)
    set(CATEGORY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${CATEGORY_DIR})
    if(NOT EXISTS ${CATEGORY_PATH})
        return()
    endif()

    file(GLOB RUNNER_SOURCES CONFIGURE_DEPENDS ${CATEGORY_PATH}/*_main.cu)
    file(GLOB TEST_SOURCES CONFIGURE_DEPENDS ${CATEGORY_PATH}/*.cu)
    if(RUNNER_SOURCES)
        list(REMOVE_ITEM TEST_SOURCES ${RUNNER_SOURCES})
    endif()
    if(NOT TEST_SOURCES OR NOT RUNNER_SOURCES)
        return()
    endif()

    set(TESTS_LIB ginPerf${CATEGORY_DIR})
    add_library(${TESTS_LIB} STATIC ${TEST_SOURCES})
    gin_perf_target_includes(${TESTS_LIB})
    gin_perf_target_compile_options(${TESTS_LIB} ${CATEGORY_DIR})
    target_link_libraries(${TESTS_LIB}
        PRIVATE
        nccl_tests_options
        nccl
        MPI::MPI_CXX
        CUDA::cudart
    )

    foreach(RUNNER_FILE ${RUNNER_SOURCES})
        get_filename_component(RUNNER_NAME ${RUNNER_FILE} NAME_WE)
        string(REGEX REPLACE "_main$" "" TEST_NAME ${RUNNER_NAME})
        set(TARGET_NAME ${TEST_NAME}_perf)

        add_executable(${TARGET_NAME}
            ${RUNNER_FILE}
            ${CMAKE_CURRENT_SOURCE_DIR}/utils/common.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/utils/args.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/utils/gin_context.cc
        )

        gin_perf_target_compile_options(${TARGET_NAME} ${CATEGORY_DIR})

        set_target_properties(${TARGET_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/device_api/gin/${CATEGORY_DIR}
        )

        gin_perf_target_includes(${TARGET_NAME})

        target_link_libraries(${TARGET_NAME}
            PRIVATE
            ${TESTS_LIB}
            nccl_tests_options
            nccl
            MPI::MPI_CXX
            CUDA::cudart
            m
            stdc++
        )
    endforeach()
endfunction()

add_gin_perf_category(latency)
add_gin_perf_category(throughput)
