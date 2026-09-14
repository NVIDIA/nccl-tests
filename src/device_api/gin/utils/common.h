#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <unistd.h>

#include "cuda_runtime.h"
#include "nccl.h"
#include "nccl_device.h"
#include "mpi.h"
#include "args.h"

/* Pre-MPI: use plain exit so the caller sees a clean error before MPI is up. */
#define MPICHECK(cmd) do { \
  int _e = (cmd); \
  if (_e != MPI_SUCCESS) { \
    fprintf(stderr, "Failed: MPI error %s:%d '%d'\n", \
            __FILE__, __LINE__, _e); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

#define CUDACHECK(cmd) do { \
  cudaError_t _e = (cmd); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n", \
            __FILE__, __LINE__, cudaGetErrorString(_e)); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

#define CUCHECK(cmd) do { \
  CUresult _e = (cmd); \
  if (_e != CUDA_SUCCESS) { \
    const char* _s = nullptr; \
    cuGetErrorString(_e, &_s); \
    fprintf(stderr, "Failed: CUDA driver error %s:%d '%s'\n", __FILE__, __LINE__, _s); \
    return ncclUnhandledCudaError; \
  } \
} while (0)

#define NCCLCHECK(cmd) do { \
  ncclResult_t _r = (cmd); \
  if (_r != ncclSuccess) { \
    fprintf(stderr, "Failed: NCCL error %s:%d '%s'\n", \
            __FILE__, __LINE__, ncclGetErrorString(_r)); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

/* Post-MPI: use MPI_Abort so all ranks are torn down consistently. */
#define MPICHECK_FATAL(cmd) do { \
  int _e = (cmd); \
  if (_e != MPI_SUCCESS) { \
    fprintf(stderr, "Fatal: MPI error %s:%d '%d'\n", \
            __FILE__, __LINE__, _e); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while (0)

#define CUDACHECK_FATAL(cmd) do { \
  cudaError_t _e = (cmd); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "Fatal: CUDA error %s:%d '%s'\n", \
            __FILE__, __LINE__, cudaGetErrorString(_e)); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while (0)

#define NCCLCHECK_FATAL(cmd) do { \
  ncclResult_t _r = (cmd); \
  if (_r != ncclSuccess) { \
    fprintf(stderr, "Fatal: NCCL error %s:%d '%s'\n", \
            __FILE__, __LINE__, ncclGetErrorString(_r)); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while (0)

/*
 * Benchmark callback invoked by ncclTestGinPerfRun for every (size, iters) point.
 *
 *   dcomm – device comm handle
 *   devBufHandle – registered buffer handle
 *   stream – CUDA stream to launch into
 *   args – parsed CLI args
 *   numElems – elements (ints) to transfer this size point
 *   iters – number of iterations to perform this call
 */
typedef void (*ginRunFn_t)(
    ncclDevComm dcomm,
    ncclDevResourceHandle devBufHandle,
    ncclWindow_t hostBufWindow,
    cudaStream_t stream,
    const ginArgs_t* args,
    size_t numElems,
    int iters
);

/*
 * Called on rank 0 only, once per size point, with the elapsed time of the
 * measured (non-warmup) phase. Print whatever metric this category cares
 * about (bandwidth divides bytes by time, latency divides time by iters, ...).
 */
typedef void (*ginReportFn_t)(size_t size, int iters, double milliseconds);

enum ginBenchmarkType_t {
  GIN_BENCHMARK_TYPE_PING,
  GIN_BENCHMARK_TYPE_PING_PONG,
  GIN_BENCHMARK_TYPE_THROUGHPUT,
};


enum ginThroughputPayload_t {
  GIN_THROUGHPUT_PAYLOAD_SIZE,
  GIN_THROUGHPUT_PAYLOAD_NONE,
  GIN_THROUGHPUT_PAYLOAD_FIXED,
};

/*
 * Bundles everything that differs between benchmark categories/APIs so a
 * single ncclTestGinPerfRun() can serve all of them:
 *
 *   run     – the kernel-launch callback (see ginRunFn_t above).
 *   name    – human-readable test name, reported above the results table.
 *   type    – benchmark category; ping-pong launches on both ranks, ping and
 *             throughput launch on rank 0 only.
 *   payloadMode – throughput only: how payload bytes per message are counted.
 *   payloadFixedBytes – when payloadMode is FIXED, bytes per message (e.g. sizeof(T)).
 */
typedef struct {
  ginRunFn_t run;
  const char* name;
  ginBenchmarkType_t type;
  ginThroughputPayload_t payloadMode;
  size_t payloadFixedBytes;
} ginBenchmark_t;

/*
 * Full perf harness: setup, warmup+measured size sweep, teardown. Shared by
 * every benchmark category (bandwidth, message rate, ping latency, ping-pong
 * latency) and every GIN API (Put, Get, Signal, PutSignal, ...) via the
 * ginBenchmark_t callbacks/flags above. ncclTestGinParseArgs must have been
 * called before this function.
 */
void ncclTestGinPerfRun(int argc, char** argv, const ginArgs_t* args, const ginBenchmark_t* bench);
