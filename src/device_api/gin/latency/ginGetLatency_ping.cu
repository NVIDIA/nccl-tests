#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "common.h"
#include "args.h"
#include "gin_context.h"

template <bool skipCreditCheck, ncclGinResourceSharingMode rsm>
__global__ void ginGetPingKernel(ncclDevComm comm, ncclDevResourceHandle hBuf, int iters, size_t bytes) {
#if __CUDA_ARCH__ >= 700
  ncclTeam team = ncclTeamWorld(comm);
  ncclGin gin(comm, 0, rsm);
  ncclSymPtr<int> sbuf = (ncclSymPtr<int>)ncclGetResourceBuffer(comm, hBuf);

  constexpr uint32_t optFlags =
    skipCreditCheck ? ncclGinOptFlagsMaySkipCreditCheck : ncclGinOptFlagsDefault;

  for (int i = 0; i < iters; i++) {
    gin.get(team, 1, sbuf.window, sbuf.offset, sbuf.window, sbuf.offset, bytes, ncclCoopThread{},
            ncclGin_None{}, optFlags);
    gin.flush(ncclCoopThread{});
  }
#endif
}

template <ncclGinResourceSharingMode rsm>
static void ginGetLatencyPingLaunchRsm(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                          const ginArgs_t* args, size_t numElems, int iters) {
  size_t bytes = numElems * sizeof(int);
#define LAUNCH_PING_GET(SKIP) \
  ginGetPingKernel<SKIP, rsm><<<1, 1, 0, stream>>>(dcomm, hBuf, iters, bytes)

  if (args->ginSkipCreditCheck) LAUNCH_PING_GET(true);
  else LAUNCH_PING_GET(false);

#undef LAUNCH_PING_GET
  CUDACHECK_FATAL(cudaGetLastError());
}

void ncclTestGinGetLatencyPingLaunch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                     const ginArgs_t* args, size_t numElems, int iters) {
  if (args->ginRsm == ncclGinRsmThread) {
    ginGetLatencyPingLaunchRsm<NCCL_GIN_RESOURCE_SHARING_THREAD>(dcomm, devBufHandle, hostBufWindow, stream, args, numElems, iters);
  } else if (args->ginRsm == ncclGinRsmCta) {
    ginGetLatencyPingLaunchRsm<NCCL_GIN_RESOURCE_SHARING_CTA>(dcomm, hBuf, stream, args, numElems, iters);
  } else {
    ginGetLatencyPingLaunchRsm<NCCL_GIN_RESOURCE_SHARING_GPU>(dcomm, hBuf, stream, args, numElems, iters);
  }
}
