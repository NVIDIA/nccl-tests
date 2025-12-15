/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
#include "nccl_device.h"
#include "vector_types.h"
#endif

#pragma weak ncclAlltoAll

void AlltoAllGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  *paramcount = (count/nranks) & -(16/eltSize);
  *sendcount = nranks*(*paramcount);
  *recvcount = *sendcount;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
}

testResult_t AlltoAllInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, 33*rep + rank, 1, 0));
    for (int j=0; j<nranks; j++) {
      size_t partcount = sendcount/nranks;
      TESTCHECK(InitData((char*)args->expected[i] + j*partcount*wordSize(type), partcount, rank*partcount, type, ncclSum, 33*rep + j, 1, 0));
    }
    CUDACHECK(cudaDeviceSynchronize());
  }
  // We don't support in-place alltoall
  args->reportErrors = in_place ? 0 : 1;
  return testSuccess;
}

void AlltoAllGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks-1))/((double)(nranks));
  *busBw = baseBw * factor;
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
// set devComm reqs for alltoall device kernels
testResult_t AlltoAllGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs, ncclCommProperties_t* commProperties) {
  if (!reqs || !commProperties) return testInternalError;

  switch(deviceImpl) {
    case 1: // NvlAlltoAllKernel
    case 2: // NvlAlltoAllKernelOptimized
      reqs->lsaBarrierCount = deviceCtaCount;
      return testSuccess;
    case 3: // GinAlltoAllKernel
    case 4: // HybridAlltoAllKernel (LSA+GIN)
      if (commProperties->ginType == NCCL_GIN_TYPE_NONE) {
        fprintf(stderr, "This test requires GIN support, but GIN support is not enabled for this communicator.\n");
        return testInternalError;
      }
      reqs->barrierCount = deviceCtaCount;
      reqs->ginSignalCount = deviceCtaCount;
      return testSuccess;
    default:
      return testNotImplemented;
  }
}
#elif NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
// set devComm reqs for alltoall device kernels
bool AlltoAllGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs) {
  if (!reqs) return false;
  memset(reqs, 0, sizeof(*reqs));

  switch(deviceImpl) {
    case 1: // NvlAlltoAllKernel
    case 2: // NvlAlltoAllKernelOptimized
      reqs->lsaBarrierCount = deviceCtaCount;
      return true;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
    case 3: // GinAlltoAllKernel
    case 4: // HybridAlltoAllKernel (LSA+GIN)
      reqs->barrierCount = deviceCtaCount;
      reqs->ginSignalCount = deviceCtaCount;
      return true;
#endif
    default:
      return false;
  }
}
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
// shared scalar AlltoAll implementation used by both kernels
template <typename T>
__device__ void AlltoAllScalarImpl(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int rank, int nRanks, int tid, int nthreads) {
  T* sendPtr = (T*)ncclGetLsaPointer(sendwin, sendoffset, rank);

  for (size_t offset = tid; offset < count; offset += nthreads) {
    for (int peer = 0; peer < nRanks; peer++) {
      T value = sendPtr[peer * count + offset];
      T* recvPtr = (T*)ncclGetLsaPointer(recvwin, recvoffset, peer);
      recvPtr[rank * count + offset] = value;
    }
  }
}

// Device implementation #1 - simple NVL kernel
template <typename T>
__global__ void NvlAlltoAllKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  int rank = devComm.rank, nRanks = devComm.nRanks;
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int nthreads = blockDim.x * gridDim.x;

  AlltoAllScalarImpl<T>(sendwin, sendoffset, recvwin, recvoffset, count, rank, nRanks, tid, nthreads);

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

// Device implementation #2 - optimized NVL kernel using vectorization and unrolling
template <typename T>
__global__ void NvlAlltoAllKernelOptimized(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  using TN = typename VectorTypeMapping<T>::Type;
  constexpr int VECTOR_FACTOR = sizeof(TN) / sizeof(T);
  constexpr int UNROLL_FACTOR = 128/sizeof(TN);
  constexpr int PEER_UNROLL = 2;

  int rank = devComm.rank, nRanks = devComm.nRanks;
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int nthreads = blockDim.x * gridDim.x;

  T* sendPtr = (T*)ncclGetLsaPointer(sendwin, sendoffset, rank);

  // alignment check: can we use vectorized operations?
  bool canVectorize = (sizeof(TN) > sizeof(T)) &&  // Only if vectorization helps
                      (reinterpret_cast<uintptr_t>(sendPtr) % sizeof(TN) == 0) &&  // Base aligned
                      ((count * sizeof(T)) % sizeof(TN) == 0);  // Stride compatible

  if (canVectorize) {
    size_t vector_count = count / VECTOR_FACTOR;
    int elements_per_iteration = nthreads * UNROLL_FACTOR;

    // process aligned vectorized elements without bounds checks
    size_t aligned_vector_count = (vector_count / elements_per_iteration) * elements_per_iteration;
    for (size_t base_offset = tid; base_offset < aligned_vector_count; base_offset += elements_per_iteration) {
      // unroll a limited number of peers at a time
      for (int peerBase = 0; peerBase < nRanks; peerBase += PEER_UNROLL) {
        int peersInGroup = min(PEER_UNROLL, nRanks - peerBase);

        #pragma unroll
        for (int p = 0; p < peersInGroup; p++) {
          int peer = peerBase + p;
          TN* sendVecPtr = (TN*)(sendPtr + peer * count);
          TN* recvVecPtr = (TN*)((T*)ncclGetLsaPointer(recvwin, recvoffset, peer) + rank * count);
          TN values[UNROLL_FACTOR];

          // split load/store into separate loops for better overlap and ILP
          #pragma unroll
          for (int i = 0; i < UNROLL_FACTOR; i++) {
            size_t offset = base_offset + i * nthreads;
            values[i] = sendVecPtr[offset];
          }
          #pragma unroll
          for (int i = 0; i < UNROLL_FACTOR; i++) {
            size_t offset = base_offset + i * nthreads;
            recvVecPtr[offset] = values[i];
          }
        }
      }
    }

    // handle remaining vectorized elements that didn't fit in aligned chunks
    for (size_t base_offset = aligned_vector_count + tid; base_offset < vector_count; base_offset += nthreads) {
      for (int peer = 0; peer < nRanks; peer++) {
        TN* sendVecPtr = (TN*)(sendPtr + peer * count);
        TN* recvVecPtr = (TN*)((T*)ncclGetLsaPointer(recvwin, recvoffset, peer) + rank * count);
        recvVecPtr[base_offset] = sendVecPtr[base_offset];
      }
    }

    // handle any remaining elements not divisible by vectorization factor
    size_t scalar_start = vector_count * VECTOR_FACTOR;
    for (size_t offset = scalar_start + tid; offset < count; offset += nthreads) {
      for (int peer = 0; peer < nRanks; peer++) {
        T value = sendPtr[peer * count + offset];
        T* recvPtr = (T*)ncclGetLsaPointer(recvwin, recvoffset, peer);
        recvPtr[rank * count + offset] = value;
      }
    }
  } else {
    // simple scalar fallback for unaligned data (identical to simple kernel)
    AlltoAllScalarImpl<T>(sendwin, sendoffset, recvwin, recvoffset, count, rank, nRanks, tid, nthreads);
  }

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
template <typename T>
__global__ void GinAlltoAllKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  int ginContext = 0;
  unsigned int signalIndex = 0;
  ncclGin gin { devComm, ginContext };
  uint64_t signalValue = gin.readSignal(signalIndex);

  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  /* send to all peers via GIN */
  const size_t size = count * sizeof(T);
  for (int r=tid; r<devComm.nRanks; r+=nthreads) {
    gin.put(ncclTeamWorld(devComm), r,
        recvwin, recvoffset + devComm.rank * size,
        sendwin, sendoffset + r * size,
        size, ncclGin_SignalInc{signalIndex});
  }

  gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + devComm.nRanks);
  gin.flush(ncclCoopCta());

  bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
}

template <typename T>
__global__ void HybridAlltoAllKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  int ginContext = 0;
  unsigned int signalIndex = 0;
  ncclGin gin { devComm, ginContext };
  uint64_t signalValue = gin.readSignal(signalIndex);

  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  ncclTeam world = ncclTeamWorld(devComm);
  ncclTeam lsa = ncclTeamLsa(devComm);
  const int startLsa = world.rank - lsa.rank;
  const int lsaSize  = lsa.nRanks;

  /* handle remote peers (i.e., non-LSA) using GIN */
  const size_t size = count * sizeof(T);
  for (int r = tid; r < startLsa; r += nthreads) {
    gin.put(world, r,
        recvwin, recvoffset + world.rank * size,
        sendwin, sendoffset + r * size,
        size, ncclGin_SignalInc{signalIndex});
  }
  for (int r = startLsa + lsaSize + tid; r < world.nRanks; r += nthreads) {
    gin.put(world, r,
        recvwin, recvoffset + world.rank * size,
        sendwin, sendoffset + r * size,
        size, ncclGin_SignalInc{signalIndex});
  }

  /* handle local peers with LSA */
  T* sendLocal = (T*)ncclGetLocalPointer(sendwin, sendoffset);
  for (size_t offset = tid; offset < count; offset += nthreads) {
    for (int lp = 0; lp < lsa.nRanks; lp++) {
      int wr = startLsa + lp;
      T* recvPtr = (T*)ncclGetLsaPointer(recvwin, recvoffset, lp);
      recvPtr[world.rank * count + offset] = sendLocal[wr * count + offset];
    }
  }

  int numRemotePeers = world.nRanks - lsa.nRanks;
  gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + numRemotePeers);
  gin.flush(ncclCoopCta());

  bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
}
#endif
#endif

testResult_t AlltoAllRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl) {
  if (deviceImpl == 0) {
    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
    if (test_ncclVersion >= NCCL_VERSION(2,28,0)) {
      NCCLCHECK(ncclAlltoAll(sptr, rptr, count, type, comm, stream));
      return testSuccess;
    }
    // fall-through to send/recv implementation if ncclAlltoAll is not available
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,7,0)
    int nRanks;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    size_t rankOffset = count * wordSize(type);
    NCCLCHECK(ncclGroupStart());
    for (int r=0; r<nRanks; r++) {
      NCCLCHECK(ncclSend(sptr+r*rankOffset, count, type, r, comm, stream));
      NCCLCHECK(ncclRecv(rptr+r*rankOffset, count, type, r, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
#else
    printf("NCCL 2.7 or later is needed for alltoall. This test was compiled with %d.%d.\n", NCCL_MAJOR, NCCL_MINOR);
    return testNcclError;
#endif
  } else {
    switch(deviceImpl) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
      case 1:
        TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(NvlAlltoAllKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
        return testSuccess;
      case 2:
        TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(NvlAlltoAllKernelOptimized, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
        return testSuccess;
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
      case 3:
        TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(GinAlltoAllKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
        return testSuccess;
      case 4:
        TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(HybridAlltoAllKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
        return testSuccess;
#endif
      default:
        return testNotImplemented;
    }
  }
  return testSuccess;
}

struct testColl alltoAllTest = {
  "AlltoAll",
  AlltoAllGetCollByteCount,
  AlltoAllInitData,
  AlltoAllGetBw,
  AlltoAllRunColl
};

void AlltoAllGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AlltoAllGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t AlltoAllRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &alltoAllTest;
  ncclDataType_t *run_types;
  const char **run_typenames;
  int type_count;

  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = test_typenum;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  for (int i=0; i<type_count; i++) {
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "none", -1));
  }
  return testSuccess;
}

struct testEngine alltoAllEngine = {
  .getBuffSize = AlltoAllGetBuffSize,
  .runTest = AlltoAllRunTest,
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  .getDevCommRequirements = AlltoAllGetDevCommRequirements
#endif
};

#pragma weak ncclTestEngine=alltoAllEngine
