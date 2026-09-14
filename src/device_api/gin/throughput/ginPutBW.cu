#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "common.h"
#include "args.h"
#include "gin_context.h"

template <bool skipCreditCheck, bool aggregateRequests, ncclGinResourceSharingMode rsm>
__global__ void ginPutBwKernel(ncclDevComm comm, ncclWindow_t dWindow, ncclWindow_t sWindow, size_t numElems,
                               int iters, int queueDepth, size_t maxElems, bool isDestBufDev, bool isSourceBufDev) {
#if __CUDA_ARCH__ >= 700
  const int tag = blockIdx.x;
  ncclTeam team = ncclTeamWorld(comm);
  const int peer = team.rank ^ 1;
  ncclGin gin(comm, tag, rsm);

  const size_t slots = maxElems / numElems;
  const size_t offset = (size_t)(threadIdx.x % slots) * numElems;

  ncclSymPtr<int> dbuf = ncclSymPtr<int>(dWindow, offset * sizeof(int));
  ncclSymPtr<int> sbuf = ncclSymPtr<int>(sWindow, offset * sizeof(int));
  const GinPutFn ginPutFn = ginPutFns[isSourceBufDev][isDestBufDev];

  const int lastActive = (int)(blockDim.x - 1);

  constexpr uint32_t optFlags =
    skipCreditCheck ? ncclGinOptFlagsMaySkipCreditCheck : ncclGinOptFlagsDefault;

  const int wqesPerOp = 1;
  const int wqesPerIter = blockDim.x * wqesPerOp;
  const int flushEvery = (queueDepth / 2) / wqesPerIter < 1 ? 1 : (queueDepth / 2) / wqesPerIter;

  for (int i = 0; i < iters; i++) {
    if constexpr (aggregateRequests) {
      if (threadIdx.x != lastActive) {
        ginPutFn(gin, team, peer, dbuf, sbuf, numElems, ncclGin_None{}, ncclGin_None{}, ncclCoopThread{},
                 ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags | ncclGinOptFlagsAggregateRequests);
      }
      __syncthreads();
      if (threadIdx.x == lastActive) {
        ginPutFn(gin, team, peer, dbuf, sbuf, numElems, ncclGin_None{},
                 ncclGin_None{}, ncclCoopThread{}, ncclGin_None{}, cuda::thread_scope_thread,
                 cuda::thread_scope_thread, optFlags);
      }
    } else {
      ginPutFn(gin, team, peer, dbuf, sbuf, numElems, ncclGin_None{},
               ncclGin_None{}, ncclCoopThread{}, ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread,
               optFlags);
    }
    if constexpr (skipCreditCheck) {
      if (i % flushEvery == 0) gin.flush(ncclCoopCta{});
    } else {
      __syncthreads();
    }
  }
  gin.flush(ncclCoopCta{});
#endif
}

template <ncclGinResourceSharingMode rsm>
static void ginPutBWLaunchRsm(const ginContext_t* ctx, cudaStream_t stream,
                               const ginArgs_t* args, size_t numElems, int iters) {
  ncclWindow_t dWindow = args->remoteMemoryType == ncclGinMemoryDevice ? ctx->devBufWindow : ctx->hostBufWindow;
  ncclWindow_t sWindow = args->localMemoryType == ncclGinMemoryDevice ? ctx->devBufWindow : ctx->hostBufWindow;
  const int queueDepth = args->queueDepth;
  const size_t maxElems = args->maxBytes / sizeof(int);
  bool isDestBufDev = args->remoteMemoryType == ncclGinMemoryDevice;
  bool isSourceBufDev = args->localMemoryType == ncclGinMemoryDevice;
#define LAUNCH_BW_PUT(SKIP, AG) \
  ginPutBwKernel<SKIP, AG, rsm><<<args->numCtas, args->numThreads, 0, stream>>>( \
    ctx->dcomm, dWindow, sWindow, numElems, iters, queueDepth, maxElems, isDestBufDev, isSourceBufDev)

  if (args->ginSkipCreditCheck) {
    if (args->ginAggregateRequests) LAUNCH_BW_PUT(true, true);
    else                              LAUNCH_BW_PUT(true, false);
  } else {
    if (args->ginAggregateRequests) LAUNCH_BW_PUT(false, true);
    else                              LAUNCH_BW_PUT(false, false);
  }

#undef LAUNCH_BW_PUT
  CUDACHECK_FATAL(cudaGetLastError());
}

void ncclTestGinPutBWLaunch(const ginContext_t* ctx, cudaStream_t stream, const ginArgs_t* args, size_t numElems, int iters) {
  if (args->ginRsm == ncclGinRsmCta) {
    ginPutBWLaunchRsm<NCCL_GIN_RESOURCE_SHARING_CTA>(ctx, stream, args, numElems, iters);
  } else {
    ginPutBWLaunchRsm<NCCL_GIN_RESOURCE_SHARING_GPU>(ctx, stream, args, numElems, iters);
  }
}
