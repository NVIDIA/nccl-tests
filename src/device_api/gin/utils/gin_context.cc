#include "gin_context.h"
#include "common.h"

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
  bufReq.outBufferHandle = &ctx->hBuf;
  bufReq.next = reqs.resourceRequirementsList;
  reqs.resourceRequirementsList = &bufReq;

  NCCLCHECK_FATAL(ncclDevCommCreate(comm, &reqs, &ctx->dcomm));
}

void ncclTestGinDevCommDestroy(ncclComm_t comm, ginContext_t* ctx) {
  NCCLCHECK_FATAL(ncclDevCommDestroy(comm, &ctx->dcomm));
}
