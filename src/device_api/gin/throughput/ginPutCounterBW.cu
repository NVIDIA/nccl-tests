#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "common.h"
#include "args.h"
#include "gin_context.h"

template <bool skipCreditCheck, bool aggregateRequests, ncclGinResourceSharingMode rsm>
__global__ void ginPutCounterBwKernel(ncclDevComm comm, ncclDevResourceHandle devBufHandle, int iters, size_t numElems,
                                        int queueDepth, size_t maxElems) {
#if __CUDA_ARCH__ >= 700
  const int tag = blockIdx.x;
  ncclTeam team = ncclTeamWorld(comm);
  const int peer = team.rank ^ 1;
  ncclGin gin(comm, tag, rsm);

  const size_t slots = maxElems / numElems;
  ncclSymPtr<int> buf = (ncclSymPtr<int>)ncclGetResourceBuffer(comm, devBufHandle);
  buf += (size_t)(threadIdx.x % slots) * numElems;
  ncclSymPtr<int> sbuf = buf;
  ncclSymPtr<int> dbuf = buf;

  const int activeThreads = (int)blockDim.x;
  const int lastActive = activeThreads - 1;

  constexpr uint32_t optFlags =
    skipCreditCheck ? ncclGinOptFlagsMaySkipCreditCheck : ncclGinOptFlagsDefault;

  const int wqesPerOp = 2;
  const int wqesPerIter = blockDim.x * wqesPerOp;
  const int flushEvery = (queueDepth / 2) / wqesPerIter < 1 ? 1 : (queueDepth / 2) / wqesPerIter;

  constexpr ncclGinCounter_t counterId = 0;

  for (int i = 0; i < iters; i++) {
    if (aggregateRequests) {
      if (threadIdx.x != lastActive) {
        gin.put(team, peer, dbuf, sbuf, numElems, ncclGin_None{}, ncclGin_WeakCounterInc{counterId}, ncclCoopThread{},
                ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread,
                optFlags | ncclGinOptFlagsAggregateRequests);
      }
      __syncthreads();
      if (threadIdx.x == lastActive) {
        gin.put(team, peer, dbuf, sbuf, numElems, ncclGin_None{}, ncclGin_WeakCounterInc{counterId}, ncclCoopThread{},
                ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
      }
    } else {
      gin.put(team, peer, dbuf, sbuf, numElems, ncclGin_None{}, ncclGin_WeakCounterInc{counterId}, ncclCoopThread{},
              ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
    }
    if (skipCreditCheck) {
      if (i % flushEvery == 0) gin.flush(ncclCoopCta{});
    } else {
      __syncthreads();
    }
  }
  gin.flush(ncclCoopCta{});
  gin.resetCounter(counterId);
#endif
}

template <ncclGinResourceSharingMode rsm>
static void ginPutCounterBWLaunchRsm(ncclDevComm dcomm, ncclDevResourceHandle devBufHandle, cudaStream_t stream,
                                       const ginArgs_t* args, size_t numElems, int iters) {
  const int queueDepth = args->queueDepth;
  const size_t maxElems = args->maxBytes / sizeof(int);
#define LAUNCH_BW_PUT_COUNTER(SKIP, AG) \
  ginPutCounterBwKernel<SKIP, AG, rsm><<<args->numCtas, args->numThreads, 0, stream>>>(dcomm, devBufHandle, iters, numElems, queueDepth, maxElems)

  if (args->ginSkipCreditCheck) {
    if (args->ginAggregateRequests) LAUNCH_BW_PUT_COUNTER(true, true);
    else                              LAUNCH_BW_PUT_COUNTER(true, false);
  } else {
    if (args->ginAggregateRequests) LAUNCH_BW_PUT_COUNTER(false, true);
    else                              LAUNCH_BW_PUT_COUNTER(false, false);
  }

#undef LAUNCH_BW_PUT_COUNTER
  CUDACHECK_FATAL(cudaGetLastError());
}

void ncclTestGinPutCounterBWLaunch(ncclDevComm dcomm, ncclDevResourceHandle devBufHandle, cudaStream_t stream,
                                   const ginArgs_t* args, size_t numElems, int iters) {
  if (args->ginRsm == ncclGinRsmCta) {
    ginPutCounterBWLaunchRsm<NCCL_GIN_RESOURCE_SHARING_CTA>(dcomm, devBufHandle, stream, args, numElems, iters);
  } else {
    ginPutCounterBWLaunchRsm<NCCL_GIN_RESOURCE_SHARING_GPU>(dcomm, devBufHandle, stream, args, numElems, iters);
  }
}
