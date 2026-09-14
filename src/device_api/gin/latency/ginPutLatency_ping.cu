#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "common.h"
#include "args.h"
#include "gin_context.h"

template <bool skipCreditCheck, ncclGinResourceSharingMode rsm>
__global__ void ginPutPingKernel(ncclDevComm comm, ncclWindow_t dWindow, ncclWindow_t sWindow, size_t numElems, int iters, bool isDestBufDev, bool isSourceBufDev) {
#if __CUDA_ARCH__ >= 700
  ncclTeam team = ncclTeamWorld(comm);
  ncclGin gin(comm, 0, rsm);

  ncclSymPtr<int> dbuf = ncclSymPtr<int>(dWindow, 0);
  ncclSymPtr<int> sbuf = ncclSymPtr<int>(sWindow, 0);
  const GinPutFn ginPutFn = ginPutFns[isSourceBufDev][isDestBufDev];

  constexpr uint32_t optFlags =
    skipCreditCheck ? ncclGinOptFlagsMaySkipCreditCheck : ncclGinOptFlagsDefault;

  for (int i = 0; i < iters; i++) {
    ginPutFn(gin, team, 1, dbuf, sbuf, numElems, ncclGin_None{}, ncclGin_None{}, ncclCoopThread{},
             ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
    gin.flush(ncclCoopThread{});
  }
#endif
}

template <ncclGinResourceSharingMode rsm>
static void ginPutLatencyPingLaunchRsm(const ginContext_t* ctx, cudaStream_t stream,
                                        const ginArgs_t* args, size_t numElems, int iters) {
  ncclWindow_t dWindow = args->remoteMemoryType == ncclGinMemoryDevice ? ctx->devBufWindow : ctx->hostBufWindow;
  ncclWindow_t sWindow = args->localMemoryType == ncclGinMemoryDevice ? ctx->devBufWindow : ctx->hostBufWindow;
  bool isDestBufDev = args->remoteMemoryType == ncclGinMemoryDevice;
  bool isSourceBufDev = args->localMemoryType == ncclGinMemoryDevice;
#define LAUNCH_PING_PUT(SKIP) \
  ginPutPingKernel<SKIP, rsm><<<1, 1, 0, stream>>>(ctx->dcomm, dWindow, sWindow, numElems, iters, isDestBufDev, isSourceBufDev)

  if (args->ginSkipCreditCheck) LAUNCH_PING_PUT(true);
  else LAUNCH_PING_PUT(false);

#undef LAUNCH_PING_PUT
  CUDACHECK_FATAL(cudaGetLastError());
}

void ncclTestGinPutLatencyPingLaunch(const ginContext_t* ctx, cudaStream_t stream, const ginArgs_t* args, size_t numElems, int iters) {
  if (args->ginRsm == ncclGinRsmThread) {
    ginPutLatencyPingLaunchRsm<NCCL_GIN_RESOURCE_SHARING_THREAD>(ctx, stream, args, numElems, iters);
  } else if (args->ginRsm == ncclGinRsmCta) {
    ginPutLatencyPingLaunchRsm<NCCL_GIN_RESOURCE_SHARING_CTA>(ctx, stream, args, numElems, iters);
  } else {
    ginPutLatencyPingLaunchRsm<NCCL_GIN_RESOURCE_SHARING_GPU>(ctx, stream, args, numElems, iters);
  }
}
