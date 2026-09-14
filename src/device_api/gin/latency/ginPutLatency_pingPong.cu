#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "common.h"
#include "args.h"
#include "gin_context.h"

template <bool skipCreditCheck, ncclGinResourceSharingMode rsm>
__global__ void ginPutPingPongKernel(ncclDevComm comm, ncclDevResourceHandle devBufHandle, int iters, size_t numElems,
                                int queueDepth) {
#if __CUDA_ARCH__ >= 700
  ncclTeam team = ncclTeamWorld(comm);
  ncclGin gin(comm, 0, rsm);
  ncclSymPtr<int> sbuf = (ncclSymPtr<int>)ncclGetResourceBuffer(comm, devBufHandle);
  ncclSymPtr<int> dbuf = sbuf;

  constexpr uint32_t optFlags =
    skipCreditCheck ? ncclGinOptFlagsMaySkipCreditCheck : ncclGinOptFlagsDefault;

  const int flushEvery = queueDepth / 2 < 1 ? 1 : queueDepth / 2;

  int* lastElem = (int*)ncclGetResourceBufferLocalPointer(comm, devBufHandle) + (numElems - 1);

  int serverSign = 1;
  int clientSign = -1;
  for (int i = 0; i < iters; i++) {
    if (comm.rank == 0) {
      while (*(volatile int*)lastElem != serverSign) continue;
      cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
      *lastElem = clientSign;
      gin.put(team, 1, dbuf, sbuf, numElems, ncclGin_None{}, ncclGin_None{}, ncclCoopThread{},
              ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
      if (skipCreditCheck && (i % flushEvery == 0)) gin.flush(ncclCoopThread{});
    } else {
      *lastElem = serverSign;
      gin.put(team, 0, dbuf, sbuf, numElems, ncclGin_None{}, ncclGin_None{}, ncclCoopThread{},
              ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
      if (skipCreditCheck && (i % flushEvery == 0)) gin.flush(ncclCoopThread{});
      while (*(volatile int*)lastElem != clientSign) continue;
      cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
    }
    clientSign--;
    serverSign++;
  }
  gin.flush(ncclCoopThread{});
#endif
}

template <ncclGinResourceSharingMode rsm>
static void ginPutLatencyPingPongLaunchRsm(ncclDevComm dcomm, ncclDevResourceHandle devBufHandle, cudaStream_t stream,
                                              const ginArgs_t* args, size_t numElems, int iters) {
  const int queueDepth = args->queueDepth;
#define LAUNCH_PING_PONG_PUT(SKIP) \
  ginPutPingPongKernel<SKIP, rsm><<<1, 1, 0, stream>>>(dcomm, devBufHandle, iters, numElems, queueDepth)

  if (args->ginSkipCreditCheck) LAUNCH_PING_PONG_PUT(true);
  else LAUNCH_PING_PONG_PUT(false);

#undef LAUNCH_PING_PONG_PUT
  CUDACHECK_FATAL(cudaGetLastError());
}

void ncclTestGinPutLatencyPingPongLaunch(ncclDevComm dcomm, ncclDevResourceHandle devBufHandle, cudaStream_t stream,
                                         const ginArgs_t* args, size_t numElems, int iters) {
  if (args->ginRsm == ncclGinRsmThread) {
    ginPutLatencyPingPongLaunchRsm<NCCL_GIN_RESOURCE_SHARING_THREAD>(dcomm, devBufHandle, stream, args, numElems, iters);
  } else if (args->ginRsm == ncclGinRsmCta) {
    ginPutLatencyPingPongLaunchRsm<NCCL_GIN_RESOURCE_SHARING_CTA>(dcomm, devBufHandle, stream, args, numElems, iters);
  } else {
    ginPutLatencyPingPongLaunchRsm<NCCL_GIN_RESOURCE_SHARING_GPU>(dcomm, devBufHandle, stream, args, numElems, iters);
  }
}
