#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "common.h"
#include "args.h"
#include "gin_context.h"

template <bool skipCreditCheck, bool aggregateRequests, ncclGinResourceSharingMode rsm>
__global__ void ginPutBwKernel(ncclDevComm comm, ncclDevResourceHandle hBuf, int iters, size_t numElems,
                                 int queueDepth, size_t maxElems) {
#if __CUDA_ARCH__ >= 700
  const int tag = blockIdx.x;
  ncclTeam team = ncclTeamWorld(comm);
  const int peer = team.rank ^ 1;
  ncclGin gin(comm, tag, rsm);

  const size_t slots = maxElems / numElems;
  ncclSymPtr<int> buf = (ncclSymPtr<int>)ncclGetResourceBuffer(comm, hBuf);
  buf += (size_t)(threadIdx.x % slots) * numElems;
  ncclSymPtr<int> sbuf = buf;
  ncclSymPtr<int> dbuf = buf;

  const int lastActive = (int)(blockDim.x - 1);

  constexpr uint32_t optFlags =
    skipCreditCheck ? ncclGinOptFlagsMaySkipCreditCheck : ncclGinOptFlagsDefault;

  const int wqesPerOp = 1;
  const int wqesPerIter = blockDim.x * wqesPerOp;
  const int flushEvery = (queueDepth / 2) / wqesPerIter < 1 ? 1 : (queueDepth / 2) / wqesPerIter;

  for (int i = 0; i < iters; i++) {
    if(aggregateRequests) {
      if(threadIdx.x != lastActive) {
        gin.put(team, peer, dbuf, sbuf, numElems, ncclGin_None{}, ncclGin_None{}, ncclCoopThread{},
                ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags | ncclGinOptFlagsAggregateRequests);
      }
      __syncthreads();
      if(threadIdx.x == lastActive) {
        gin.put(team, peer, dbuf, sbuf, numElems, ncclGin_None{}, ncclGin_None{}, ncclCoopThread{},
                ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
      }
    } else {
      gin.put(team, peer, dbuf, sbuf, numElems, ncclGin_None{}, ncclGin_None{}, ncclCoopThread{},
              ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
    }
    if (skipCreditCheck) {
      if (i % flushEvery == 0) gin.flush(ncclCoopCta{});
    } else {
      __syncthreads();
    }
  }
  gin.flush(ncclCoopCta{});
#endif
}

template <ncclGinResourceSharingMode rsm>
static void ginPutBWLaunchRsm(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                const ginArgs_t* args, size_t numElems, int iters) {
  const int queueDepth = args->queueDepth;
  const size_t maxElems = args->maxBytes / sizeof(int);
#define LAUNCH_BW_PUT(SKIP, AG) \
  ginPutBwKernel<SKIP, AG, rsm><<<args->numCtas, args->numThreads, 0, stream>>>(dcomm, hBuf, iters, numElems, queueDepth, maxElems)

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

void ncclTestGinPutBWLaunch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                            const ginArgs_t* args, size_t numElems, int iters) {
  if (args->ginRsm == GIN_RSM_CTA) {
    ginPutBWLaunchRsm<NCCL_GIN_RESOURCE_SHARING_CTA>(dcomm, hBuf, stream, args, numElems, iters);
  } else {
    ginPutBWLaunchRsm<NCCL_GIN_RESOURCE_SHARING_GPU>(dcomm, hBuf, stream, args, numElems, iters);
  }
}
