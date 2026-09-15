#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "common.h"
#include "args.h"
#include "gin_context.h"

template <bool skipCreditCheck, bool strongSignal, ncclGinResourceSharingMode rsm>
__global__ void ginPutSignalPingPongKernel(ncclDevComm comm, ncclDevResourceHandle hBuf, int iters, size_t numElems,
                                     int queueDepth) {
#if __CUDA_ARCH__ >= 700
  ncclTeam team = ncclTeamWorld(comm);
  ncclGin gin(comm, 0, rsm);
  ncclSymPtr<int> sbuf = (ncclSymPtr<int>)ncclGetResourceBuffer(comm, hBuf);
  ncclSymPtr<int> dbuf = sbuf;

  constexpr uint32_t optFlags =
    skipCreditCheck ? ncclGinOptFlagsMaySkipCreditCheck : ncclGinOptFlagsDefault;

  const int flushEvery = queueDepth / 2 < 1 ? 1 : queueDepth / 2;

  constexpr ncclGinSignal_t signalId = 0;
  uint64_t expected = 0;

  for (int i = 0; i < iters; i++) {
    expected++;
    if (comm.rank == 0) {
      gin.waitSignal(ncclCoopThread{}, signalId, expected);
      if (strongSignal) {
        gin.put(team, 1, dbuf, sbuf, numElems, ncclGin_StrongSignalInc{signalId}, ncclGin_None{}, ncclCoopThread{},
                ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
      } else {
        gin.put(team, 1, dbuf, sbuf, numElems, ncclGin_WeakSignalInc{signalId}, ncclGin_None{}, ncclCoopThread{},
                ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
      }
      if (skipCreditCheck && (i % flushEvery == 0)) gin.flush(ncclCoopThread{});
    } else {
      if (strongSignal) {
        gin.put(team, 0, dbuf, sbuf, numElems, ncclGin_StrongSignalInc{signalId}, ncclGin_None{}, ncclCoopThread{},
                ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
      } else {
        gin.put(team, 0, dbuf, sbuf, numElems, ncclGin_WeakSignalInc{signalId}, ncclGin_None{}, ncclCoopThread{},
                ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
      }
      if (skipCreditCheck && (i % flushEvery == 0)) gin.flush(ncclCoopThread{});
      gin.waitSignal(ncclCoopThread{}, signalId, expected);
    }
  }
  gin.resetSignal(signalId);
  gin.flush(ncclCoopThread{});
#endif
}

template <ncclGinResourceSharingMode rsm>
static void ginPutSignalLatencyPingPongLaunchRsm(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                                    const ginArgs_t* args, size_t numElems, int iters) {
  const int queueDepth = args->queueDepth;
#define LAUNCH_PING_PONG_PUT_SIGNAL(SKIP, STRONG) \
  ginPutSignalPingPongKernel<SKIP, STRONG, rsm><<<1, 1, 0, stream>>>(dcomm, hBuf, iters, numElems, queueDepth)

  if (args->ginSkipCreditCheck) {
    if (args->ginStrongSignal) LAUNCH_PING_PONG_PUT_SIGNAL(true, true);
    else                         LAUNCH_PING_PONG_PUT_SIGNAL(true, false);
  } else {
    if (args->ginStrongSignal) LAUNCH_PING_PONG_PUT_SIGNAL(false, true);
    else                         LAUNCH_PING_PONG_PUT_SIGNAL(false, false);
  }

#undef LAUNCH_PING_PONG_PUT_SIGNAL
  CUDACHECK_FATAL(cudaGetLastError());
}

void ncclTestGinPutSignalLatencyPingPongLaunch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                               const ginArgs_t* args, size_t numElems, int iters) {
  if (args->ginRsm == GIN_RSM_THREAD) {
    ginPutSignalLatencyPingPongLaunchRsm<NCCL_GIN_RESOURCE_SHARING_THREAD>(dcomm, hBuf, stream, args, numElems, iters);
  } else if (args->ginRsm == GIN_RSM_CTA) {
    ginPutSignalLatencyPingPongLaunchRsm<NCCL_GIN_RESOURCE_SHARING_CTA>(dcomm, hBuf, stream, args, numElems, iters);
  } else {
    ginPutSignalLatencyPingPongLaunchRsm<NCCL_GIN_RESOURCE_SHARING_GPU>(dcomm, hBuf, stream, args, numElems, iters);
  }
}
