#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "common.h"
#include "args.h"
#include "gin_context.h"

template <bool skipCreditCheck, bool aggregateRequests, bool strongSignal,
          ncclGinResourceSharingMode rsm>
__global__ void ginPutSignalBwKernel(ncclDevComm comm, ncclWindow_t devWindow, int iters, size_t numElems,
                                       int queueDepth, size_t maxElems) {
#if __CUDA_ARCH__ >= 700
  const int tag = blockIdx.x;
  ncclTeam team = ncclTeamWorld(comm);
  const int peer = team.rank ^ 1;
  ncclGin gin(comm, tag, rsm);

  const size_t slots = maxElems / numElems;
  const size_t offset = (size_t)(threadIdx.x % slots) * numElems;
  ncclSymPtr<int> dbuf = ncclSymPtr<int>(devWindow, offset * sizeof(int));
  ncclSymPtr<int> sbuf = ncclSymPtr<int>(devWindow, offset * sizeof(int));

  const int lastActive = (int)(blockDim.x - 1);

  constexpr uint32_t optFlags =
    skipCreditCheck ? ncclGinOptFlagsMaySkipCreditCheck : ncclGinOptFlagsDefault;

  const int wqesPerOp = 2;
  const int wqesPerIter = blockDim.x * wqesPerOp;
  const int flushEvery = (queueDepth / 2) / wqesPerIter < 1 ? 1 : (queueDepth / 2) / wqesPerIter;

  constexpr ncclGinSignal_t signalId = 0;

  for (int i = 0; i < iters; i++) {
    if constexpr (aggregateRequests) {
      if (threadIdx.x != lastActive) {
        if constexpr (strongSignal) {
          gin.put(team, peer, dbuf, sbuf, numElems, ncclGin_StrongSignalInc{signalId}, ncclGin_None{}, ncclCoopThread{},
                  ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread,
                  optFlags | ncclGinOptFlagsAggregateRequests);
        } else {
          gin.put(team, peer, dbuf, sbuf, numElems, ncclGin_WeakSignalInc{signalId}, ncclGin_None{}, ncclCoopThread{},
                  ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread,
                  optFlags | ncclGinOptFlagsAggregateRequests);
        }
      }
      __syncthreads();
      if (threadIdx.x == lastActive) {
        if constexpr (strongSignal) {
          gin.put(team, peer, dbuf, sbuf, numElems, ncclGin_StrongSignalInc{signalId}, ncclGin_None{}, ncclCoopThread{},
                  ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
        } else {
          gin.put(team, peer, dbuf, sbuf, numElems, ncclGin_WeakSignalInc{signalId}, ncclGin_None{}, ncclCoopThread{},
                  ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
        }
      }
    } else {
      if constexpr (strongSignal) {
        gin.put(team, peer, dbuf, sbuf, numElems, ncclGin_StrongSignalInc{signalId}, ncclGin_None{}, ncclCoopThread{},
                ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
      } else {
        gin.put(team, peer, dbuf, sbuf, numElems, ncclGin_WeakSignalInc{signalId}, ncclGin_None{}, ncclCoopThread{},
                ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
      }
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
static void ginPutSignalBWLaunchRsm(const ginContext_t* ctx, cudaStream_t stream,
                                     const ginArgs_t* args, size_t numElems, int iters) {
  const int queueDepth = args->queueDepth;
  const size_t maxElems = args->maxBytes / sizeof(int);
#define LAUNCH_BW_PUT_SIGNAL(SKIP, AG, STRONG) \
  ginPutSignalBwKernel<SKIP, AG, STRONG, rsm><<<args->numCtas, args->numThreads, 0, stream>>>(ctx->dcomm, ctx->devBufWindow, iters, numElems, queueDepth, maxElems)

  if (args->ginSkipCreditCheck) {
    if (args->ginAggregateRequests) {
      if (args->ginStrongSignal) LAUNCH_BW_PUT_SIGNAL(true, true, true);
      else                         LAUNCH_BW_PUT_SIGNAL(true, true, false);
    } else {
      if (args->ginStrongSignal) LAUNCH_BW_PUT_SIGNAL(true, false, true);
      else                         LAUNCH_BW_PUT_SIGNAL(true, false, false);
    }
  } else {
    if (args->ginAggregateRequests) {
      if (args->ginStrongSignal) LAUNCH_BW_PUT_SIGNAL(false, true, true);
      else                         LAUNCH_BW_PUT_SIGNAL(false, true, false);
    } else {
      if (args->ginStrongSignal) LAUNCH_BW_PUT_SIGNAL(false, false, true);
      else                         LAUNCH_BW_PUT_SIGNAL(false, false, false);
    }
  }

#undef LAUNCH_BW_PUT_SIGNAL
  CUDACHECK_FATAL(cudaGetLastError());
}

void ncclTestGinPutSignalBWLaunch(const ginContext_t* ctx, cudaStream_t stream, const ginArgs_t* args, size_t numElems, int iters) {
  if (args->ginRsm == ncclGinRsmCta) {
    ginPutSignalBWLaunchRsm<NCCL_GIN_RESOURCE_SHARING_CTA>(ctx, stream, args, numElems, iters);
  } else {
    ginPutSignalBWLaunchRsm<NCCL_GIN_RESOURCE_SHARING_GPU>(ctx, stream, args, numElems, iters);
  }
}
