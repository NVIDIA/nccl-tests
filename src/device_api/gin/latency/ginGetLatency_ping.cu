#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "common.h"
#include "args.h"
#include "gin_context.h"

template <bool skipCreditCheck, ncclGinResourceSharingMode rsm>
__global__ void ginGetPingKernel(ncclDevComm comm, ncclWindow_t rWindow, ncclWindow_t lWindow, int iters, size_t bytes, bool isRemoteBufDev, bool isLocalBufDev) {
#if __CUDA_ARCH__ >= 700
  ncclTeam team = ncclTeamWorld(comm);
  ncclGin gin(comm, 0, rsm);

  ncclSymPtr<int> rbuf = ncclSymPtr<int>(rWindow, 0);
  ncclSymPtr<int> lbuf = ncclSymPtr<int>(lWindow, 0);
  const GinGetFn ginGetFn = ginGetFns[isLocalBufDev][isRemoteBufDev];

  constexpr uint32_t optFlags =
    skipCreditCheck ? ncclGinOptFlagsMaySkipCreditCheck : ncclGinOptFlagsDefault;

  for (int i = 0; i < iters; i++) {
    ginGetFn(gin, team, 1, rbuf.window, rbuf.offset, lbuf.window, lbuf.offset, bytes, ncclCoopThread{},
             ncclGin_None{}, optFlags);
    gin.flush(ncclCoopThread{});
  }
#endif
}

template <ncclGinResourceSharingMode rsm>
static void ginGetLatencyPingLaunchRsm(const ginContext_t* ctx, cudaStream_t stream,
                                        const ginArgs_t* args, size_t numElems, int iters) {
  ncclWindow_t rWindow = args->remoteMemoryType == ncclGinMemoryDevice ? ctx->devBufWindow : ctx->hostBufWindow;
  ncclWindow_t lWindow = args->localMemoryType == ncclGinMemoryDevice ? ctx->devBufWindow : ctx->hostBufWindow;
  size_t bytes = numElems * sizeof(int);
  bool isRemoteBufDev = args->remoteMemoryType == ncclGinMemoryDevice;
  bool isLocalBufDev = args->localMemoryType == ncclGinMemoryDevice;
#define LAUNCH_PING_GET(SKIP) \
  ginGetPingKernel<SKIP, rsm><<<1, 1, 0, stream>>>(ctx->dcomm, rWindow, lWindow, iters, bytes, isRemoteBufDev, isLocalBufDev)
  if (args->ginSkipCreditCheck) LAUNCH_PING_GET(true);
  else LAUNCH_PING_GET(false);

#undef LAUNCH_PING_GET
  CUDACHECK_FATAL(cudaGetLastError());
}

void ncclTestGinGetLatencyPingLaunch(const ginContext_t* ctx, cudaStream_t stream, const ginArgs_t* args, size_t numElems, int iters) {
  if (args->ginRsm == ncclGinRsmThread) {
    ginGetLatencyPingLaunchRsm<NCCL_GIN_RESOURCE_SHARING_THREAD>(ctx, stream, args, numElems, iters);
  } else if (args->ginRsm == ncclGinRsmCta) {
    ginGetLatencyPingLaunchRsm<NCCL_GIN_RESOURCE_SHARING_CTA>(ctx, stream, args, numElems, iters);
  } else {
    ginGetLatencyPingLaunchRsm<NCCL_GIN_RESOURCE_SHARING_GPU>(ctx, stream, args, numElems, iters);
  }
}
