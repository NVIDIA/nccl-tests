#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "common.h"
#include "args.h"
#include "gin_context.h"

template <bool skipCreditCheck, bool aggregateRequests, ncclGinResourceSharingMode rsm>
__global__ void ginGetBwKernel(ncclDevComm comm, ncclWindow_t rWindow, ncclWindow_t lWindow, size_t numElems,
                               int iters, int queueDepth, size_t maxElems, bool isRemoteBufDev, bool isLocalBufDev) {
#if __CUDA_ARCH__ >= 700
  const int tag = blockIdx.x;
  ncclTeam team = ncclTeamWorld(comm);
  const int peer = team.rank ^ 1;
  ncclGin gin(comm, tag, rsm);

  const size_t slots = maxElems / numElems;
  const size_t offset = (size_t)(threadIdx.x % slots) * numElems;
  const size_t myBytes = numElems * sizeof(int);

  ncclSymPtr<int> rbuf = ncclSymPtr<int>(rWindow, offset * sizeof(int));
  ncclSymPtr<int> lbuf = ncclSymPtr<int>(lWindow, offset * sizeof(int));
  const GinGetFn ginGetFn = ginGetFns[isLocalBufDev][isRemoteBufDev];

  constexpr uint32_t optFlags =
    skipCreditCheck ? ncclGinOptFlagsMaySkipCreditCheck : ncclGinOptFlagsDefault;
  const int lastActive = (int)(blockDim.x - 1);
  const int wqesPerOp = 1;
  const int wqesPerIter = blockDim.x * wqesPerOp;
  const int flushEvery = (queueDepth / 2) / wqesPerIter < 1 ? 1 : (queueDepth / 2) / wqesPerIter;

  for (int i = 0; i < iters; i++) {
    if constexpr (aggregateRequests) {
      if (threadIdx.x != lastActive) {
        ginGetFn(gin, team, peer, rbuf.window, rbuf.offset, lbuf.window, lbuf.offset, myBytes, ncclCoopThread{},
                 ncclGin_None{}, optFlags | ncclGinOptFlagsAggregateRequests);
      }
      __syncthreads();
      if (threadIdx.x == lastActive) {
        ginGetFn(gin, team, peer, rbuf.window, rbuf.offset, lbuf.window, lbuf.offset, myBytes, ncclCoopThread{},
                 ncclGin_None{}, optFlags);
      }
    } else {
      ginGetFn(gin, team, peer, rbuf.window, rbuf.offset, lbuf.window, lbuf.offset, myBytes, ncclCoopThread{},
               ncclGin_None{}, optFlags);
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
static void ginGetBWLaunchRsm(const ginContext_t* ctx, cudaStream_t stream,
                               const ginArgs_t* args, size_t numElems, int iters) {
  ncclWindow_t rWindow = args->remoteMemoryType == ncclGinMemoryDevice ? ctx->devBufWindow : ctx->hostBufWindow;
  ncclWindow_t lWindow = args->localMemoryType == ncclGinMemoryDevice ? ctx->devBufWindow : ctx->hostBufWindow;
  const int queueDepth = args->queueDepth;
  const size_t maxElems = args->maxBytes / sizeof(int);
  const bool isRemoteBufDev = args->remoteMemoryType == ncclGinMemoryDevice;
  const bool isLocalBufDev = args->localMemoryType == ncclGinMemoryDevice;
#define LAUNCH_BW_GET(SKIP, AG) \
  ginGetBwKernel<SKIP, AG, rsm><<<args->numCtas, args->numThreads, 0, stream>>>( \
    ctx->dcomm, rWindow, lWindow, numElems, iters, queueDepth, maxElems, isRemoteBufDev, isLocalBufDev)

  if (args->ginSkipCreditCheck) {
    if (args->ginAggregateRequests) LAUNCH_BW_GET(true, true);
    else                              LAUNCH_BW_GET(true, false);
  } else {
    if (args->ginAggregateRequests) LAUNCH_BW_GET(false, true);
    else                              LAUNCH_BW_GET(false, false);
  }

#undef LAUNCH_BW_GET
  CUDACHECK_FATAL(cudaGetLastError());
}

void ncclTestGinGetBWLaunch(const ginContext_t* ctx, cudaStream_t stream, const ginArgs_t* args, size_t numElems, int iters) {
  if (args->ginRsm == ncclGinRsmCta) {
    ginGetBWLaunchRsm<NCCL_GIN_RESOURCE_SHARING_CTA>(ctx, stream, args, numElems, iters);
  } else {
    ginGetBWLaunchRsm<NCCL_GIN_RESOURCE_SHARING_GPU>(ctx, stream, args, numElems, iters);
  }
}
