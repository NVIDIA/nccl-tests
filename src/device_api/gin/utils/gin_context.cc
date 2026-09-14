#include "gin_context.h"
#include "common.h"

#include <cuda.h>

static ncclResult_t ncclTestGinHostAlloc(void** ptr, size_t size) {
#if CUDART_VERSION >= 12020
  size_t granularity = 0;
  CUdevice currentDev;
  CUmemAllocationProp prop = {};
  CUmemAccessDesc accessDesc = {};
  CUmemGenericAllocationHandle handle;
  int cudaDev, cpuNumaNodeId = -1;

  CUDACHECK(cudaGetDevice(&cudaDev));
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));
  CUCHECK(cuDeviceGetAttribute(&cpuNumaNodeId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, currentDev));

  if (cpuNumaNodeId < 0) cpuNumaNodeId = 0;
  prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  prop.location.id = cpuNumaNodeId;
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

  size = ((size + granularity - 1) / granularity) * granularity;
  CUCHECK(cuMemCreate(&handle, size, &prop, 0));
  CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, granularity, 0, 0));
  CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));

  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));

  accessDesc.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  accessDesc.location.id = cpuNumaNodeId;
  CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));

  return ncclSuccess;
#else /* CUDART_VERSION >= 12020 */
  (void)ptr;
  (void)size;
  return ncclSystemError;
#endif /* CUDART_VERSION >= 12020 */
}

static ncclResult_t ncclTestGinHostFree(void* ptr) {
#if CUDART_VERSION >= 12020
  if (ptr == nullptr) return ncclSuccess;

  CUmemGenericAllocationHandle handle;
  size_t size = 0;

  CUCHECK(cuMemRetainAllocationHandle(&handle, ptr));
  CUCHECK(cuMemRelease(handle));
  CUCHECK(cuMemGetAddressRange(nullptr, &size, (CUdeviceptr)ptr));
  CUCHECK(cuMemUnmap((CUdeviceptr)ptr, size));
  CUCHECK(cuMemRelease(handle));
  CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
  return ncclSuccess;
#else /* CUDART_VERSION >= 12020 */
  (void)ptr;
  return ncclSystemError;
#endif /* CUDART_VERSION >= 12020 */
}

/*
 * The build only produces these benchmarks against headers new enough for the
 * GIN requirements below, so this guards the remaining case the build system
 * cannot see: a library older than those headers loaded at run time.
 */
static void checkNcclVersion(void) {
  int runtimeVersion = 0;
  NCCLCHECK_FATAL(ncclGetVersion(&runtimeVersion));
  if (runtimeVersion < NCCL_VERSION(2, 30, 7)) {
    fprintf(stderr,
            "Incompatible NCCL versions. nccl-tests was compiled with NCCL %d, but is "
            "running with NCCL %d. The GIN device API is not compatible with versions "
            "before 2.30.7.\n",
            NCCL_VERSION_CODE, runtimeVersion);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

size_t ncclTestGinMaxBufferBytes(void) {
  return 20ULL * (1ULL << 31);
}

void ncclTestGinDevCommCreate(ncclComm_t comm, const ginArgs_t* args, ginContext_t* ctx) {
  checkNcclVersion();

  if (args->localMemoryType == ncclGinMemoryHost || args->remoteMemoryType == ncclGinMemoryHost) {
    NCCLCHECK_FATAL(ncclTestGinHostAlloc(&ctx->hostBuf, args->maxBytes));
    NCCLCHECK_FATAL(ncclCommWindowRegister(comm, ctx->hostBuf, args->maxBytes, &ctx->hostBufWindow, NCCL_WIN_GIN_ONLY));
  }

  ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginContextCount = args->numCtas;
  reqs.ginQueueDepth = args->queueDepth;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  reqs.ginSignalCount = 1;
  reqs.ginCounterCount = 1;

  reqs.ginStrongSignalsRequired = args->ginStrongSignal;
  reqs.ginVaSignalsRequired = false;

  ncclDevResourceRequirements bufReq = {};
  bufReq.bufferSize = args->maxBytes;
  bufReq.outBufferHandle = &ctx->devBufHandle;
  bufReq.next = reqs.resourceRequirementsList;
  reqs.resourceRequirementsList = &bufReq;

  NCCLCHECK_FATAL(ncclDevCommCreate(comm, &reqs, &ctx->dcomm));
}

void ncclTestGinDevCommDestroy(ncclComm_t comm, ginContext_t* ctx) {
  NCCLCHECK_FATAL(ncclDevCommDestroy(comm, &ctx->dcomm));
  if (ctx->hostBufWindow != nullptr) NCCLCHECK_FATAL(ncclCommWindowDeregister(comm, ctx->hostBufWindow));
  if (ctx->hostBuf != nullptr) NCCLCHECK_FATAL(ncclTestGinHostFree(ctx->hostBuf));
}
