#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "common.h"
#include "args.h"
#include "gin_context.h"

template <bool skipCreditCheck, bool strongSignal, ncclGinResourceSharingMode rsm>
__global__ void ginPutSignalPingKernel(ncclDevComm comm, ncclDevResourceHandle hBuf, int iters, size_t numElems) {
#if __CUDA_ARCH__ >= 700
  ncclTeam team = ncclTeamWorld(comm);
  ncclGin gin(comm, 0, rsm);
  ncclSymPtr<int> sbuf = (ncclSymPtr<int>)ncclGetResourceBuffer(comm, hBuf);
  ncclSymPtr<int> dbuf = sbuf;

  constexpr uint32_t optFlags =
    skipCreditCheck ? ncclGinOptFlagsMaySkipCreditCheck : ncclGinOptFlagsDefault;

  constexpr ncclGinSignal_t signalId = 0;

  for (int i = 0; i < iters; i++) {
    if (strongSignal) {
      gin.put(team, 1, dbuf, sbuf, numElems, ncclGin_StrongSignalInc{signalId}, ncclGin_None{}, ncclCoopThread{},
              ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
    } else {
      gin.put(team, 1, dbuf, sbuf, numElems, ncclGin_WeakSignalInc{signalId}, ncclGin_None{}, ncclCoopThread{},
              ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
    }
    gin.flush(ncclCoopThread{});
  }
#endif
}

template <ncclGinResourceSharingMode rsm>
static void ginPutSignalLatencyPingLaunchRsm(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                                const ginArgs_t* args, size_t numElems, int iters) {
#define LAUNCH_PING_PUT_SIGNAL(SKIP, STRONG) \
  ginPutSignalPingKernel<SKIP, STRONG, rsm><<<1, 1, 0, stream>>>(dcomm, hBuf, iters, numElems)

  if (args->ginSkipCreditCheck) {
    if (args->ginStrongSignal) LAUNCH_PING_PUT_SIGNAL(true, true);
    else                         LAUNCH_PING_PUT_SIGNAL(true, false);
  } else {
    if (args->ginStrongSignal) LAUNCH_PING_PUT_SIGNAL(false, true);
    else                         LAUNCH_PING_PUT_SIGNAL(false, false);
  }

#undef LAUNCH_PING_PUT_SIGNAL
  CUDACHECK_FATAL(cudaGetLastError());
}

void ncclTestGinPutSignalLatencyPingLaunch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                           const ginArgs_t* args, size_t numElems, int iters) {
  if (args->ginRsm == ncclGinRsmThread) {
    ginPutSignalLatencyPingLaunchRsm<NCCL_GIN_RESOURCE_SHARING_THREAD>(dcomm, hBuf, stream, args, numElems, iters);
  } else if (args->ginRsm == ncclGinRsmCta) {
    ginPutSignalLatencyPingLaunchRsm<NCCL_GIN_RESOURCE_SHARING_CTA>(dcomm, hBuf, stream, args, numElems, iters);
  } else {
    ginPutSignalLatencyPingLaunchRsm<NCCL_GIN_RESOURCE_SHARING_GPU>(dcomm, hBuf, stream, args, numElems, iters);
  }
}
