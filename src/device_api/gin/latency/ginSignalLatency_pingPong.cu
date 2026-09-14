#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "common.h"
#include "args.h"
#include "gin_context.h"

template <bool skipCreditCheck, bool strongSignal, ncclGinResourceSharingMode rsm>
__global__ void ginSignalPingPongKernel(ncclDevComm comm, ncclWindow_t devWindow, int iters, int queueDepth) {
#if __CUDA_ARCH__ >= 700
  ncclTeam team = ncclTeamWorld(comm);
  ncclGin gin(comm, 0, rsm);

  constexpr uint32_t optFlags =
    skipCreditCheck ? ncclGinOptFlagsMaySkipCreditCheck : ncclGinOptFlagsDefault;

  const int flushEvery = queueDepth / 2 < 1 ? 1 : queueDepth / 2;

  constexpr ncclGinSignal_t signalId = 0;
  uint64_t expected = 0;

  for (int i = 0; i < iters; i++) {
    expected++;
    if (comm.rank == 0) {
      gin.waitSignal(ncclCoopThread{}, signalId, expected);
      if constexpr (strongSignal) {
        gin.signal(team, 1, ncclGin_StrongSignalInc{signalId}, ncclCoopThread{}, ncclGin_None{},
                   cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
      } else {
        gin.signal(team, 1, ncclGin_WeakSignalInc{signalId}, ncclCoopThread{}, ncclGin_None{},
                   cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
      }
      if constexpr (skipCreditCheck) {
        if (i % flushEvery == 0) gin.flush(ncclCoopThread{});
      }
    } else {
      if constexpr (strongSignal) {
        gin.signal(team, 0, ncclGin_StrongSignalInc{signalId}, ncclCoopThread{}, ncclGin_None{},
                   cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
      } else {
        gin.signal(team, 0, ncclGin_WeakSignalInc{signalId}, ncclCoopThread{}, ncclGin_None{},
                   cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
      }
      if constexpr (skipCreditCheck) {
        if (i % flushEvery == 0) gin.flush(ncclCoopThread{});
      }
      gin.waitSignal(ncclCoopThread{}, signalId, expected);
    }
  }
  gin.resetSignal(signalId);
  gin.flush(ncclCoopThread{});
#endif
}

template <ncclGinResourceSharingMode rsm>
static void ginSignalLatencyPingPongLaunchRsm(const ginContext_t* ctx, cudaStream_t stream,
                                               const ginArgs_t* args, int iters) {
  const int queueDepth = args->queueDepth;
#define LAUNCH_PING_PONG_SIGNAL(SKIP, STRONG) \
  ginSignalPingPongKernel<SKIP, STRONG, rsm><<<1, 1, 0, stream>>>(ctx->dcomm, ctx->devBufWindow, iters, queueDepth)

  if (args->ginSkipCreditCheck) {
    if (args->ginStrongSignal) LAUNCH_PING_PONG_SIGNAL(true, true);
    else                         LAUNCH_PING_PONG_SIGNAL(true, false);
  } else {
    if (args->ginStrongSignal) LAUNCH_PING_PONG_SIGNAL(false, true);
    else                         LAUNCH_PING_PONG_SIGNAL(false, false);
  }

#undef LAUNCH_PING_PONG_SIGNAL
  CUDACHECK_FATAL(cudaGetLastError());
}

void ncclTestGinSignalLatencyPingPongLaunch(const ginContext_t* ctx, cudaStream_t stream, const ginArgs_t* args, size_t numElems, int iters) {
  if (args->ginRsm == ncclGinRsmThread) {
    ginSignalLatencyPingPongLaunchRsm<NCCL_GIN_RESOURCE_SHARING_THREAD>(ctx, stream, args, iters);
  } else if (args->ginRsm == ncclGinRsmCta) {
    ginSignalLatencyPingPongLaunchRsm<NCCL_GIN_RESOURCE_SHARING_CTA>(ctx, stream, args, iters);
  } else {
    ginSignalLatencyPingPongLaunchRsm<NCCL_GIN_RESOURCE_SHARING_GPU>(ctx, stream, args, iters);
  }
}
