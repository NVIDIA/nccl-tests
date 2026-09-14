#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "common.h"
#include "args.h"
#include "gin_context.h"

template <bool skipCreditCheck, ncclGinResourceSharingMode rsm>
__global__ void ginPutPingKernel(ncclDevComm comm, ncclDevResourceHandle devBufHandle, int iters, size_t numElems) {
#if __CUDA_ARCH__ >= 700
  ncclTeam team = ncclTeamWorld(comm);
  ncclGin gin(comm, 0, rsm);
  ncclSymPtr<int> sbuf = (ncclSymPtr<int>)ncclGetResourceBuffer(comm, devBufHandle);
  ncclSymPtr<int> dbuf = sbuf;

  constexpr uint32_t optFlags =
    skipCreditCheck ? ncclGinOptFlagsMaySkipCreditCheck : ncclGinOptFlagsDefault;

  for (int i = 0; i < iters; i++) {
    gin.put(team, 1, dbuf, sbuf, numElems, ncclGin_None{}, ncclGin_None{}, ncclCoopThread{},
            ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
    gin.flush(ncclCoopThread{});
  }
#endif
}

template <ncclGinResourceSharingMode rsm>
static void ginPutLatencyPingLaunchRsm(ncclDevComm dcomm, ncclDevResourceHandle devBufHandle, cudaStream_t stream,
                                          const ginArgs_t* args, size_t numElems, int iters) {
#define LAUNCH_PING_PUT(SKIP) \
  ginPutPingKernel<SKIP, rsm><<<1, 1, 0, stream>>>(dcomm, devBufHandle, iters, numElems)

  if (args->ginSkipCreditCheck) LAUNCH_PING_PUT(true);
  else LAUNCH_PING_PUT(false);

#undef LAUNCH_PING_PUT
  CUDACHECK_FATAL(cudaGetLastError());
}

void ncclTestGinPutLatencyPingLaunch(ncclDevComm dcomm, ncclDevResourceHandle devBufHandle, cudaStream_t stream,
                                     const ginArgs_t* args, size_t numElems, int iters) {
  if (args->ginRsm == ncclGinRsmThread) {
    ginPutLatencyPingLaunchRsm<NCCL_GIN_RESOURCE_SHARING_THREAD>(dcomm, devBufHandle, stream, args, numElems, iters);
  } else if (args->ginRsm == ncclGinRsmCta) {
    ginPutLatencyPingLaunchRsm<NCCL_GIN_RESOURCE_SHARING_CTA>(dcomm, devBufHandle, stream, args, numElems, iters);
  } else {
    ginPutLatencyPingLaunchRsm<NCCL_GIN_RESOURCE_SHARING_GPU>(dcomm, devBufHandle, stream, args, numElems, iters);
  }
}
