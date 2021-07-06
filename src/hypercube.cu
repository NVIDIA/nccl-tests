/*************************************************************************
 * Copyright (c) 2016-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"

#define ALIGN 4

void print_header() {
  PRINT("# %10s  %12s  %8s            out-of-place                       in-place          \n", "", "", "");
  PRINT("# %10s  %12s  %8s  %7s  %6s  %6s  %5s  %7s  %6s  %6s  %5s\n", "size", "count", "type",
        "time", "algbw", "busbw", "error", "time", "algbw", "busbw", "error");
  PRINT("# %10s  %12s  %8s  %7s  %6s  %6s  %5s  %7s  %6s  %6s  %5s\n", "(B)", "(elements)", "",
        "(us)", "(GB/s)", "(GB/s)", "", "(us)", "(GB/s)", "(GB/s)", "");
}

void print_line_header (size_t size, size_t count, const char *typeName, const char *opName, int root) {
  PRINT("%12li  %12li  %8s", size, count, typeName);
}

void HyperCubeGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, int nranks) {
  size_t base = (count/(ALIGN*nranks))*ALIGN;
  *sendcount = base;
  *recvcount = base*nranks;
  *sendInplaceOffset = base;
  *recvInplaceOffset = 0;
  *paramcount = base;
}

testResult_t HyperCubeInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    int gpuid = args->localRank*args->nThreads*args->nGpus + args->thread*args->nGpus + i;
    CUDACHECK(cudaSetDevice(gpuid));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? ((char*)args->recvbuffs[i])+rank*args->sendBytes : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, type, rep, rank));
    for (int j=0; j<nranks; j++) {
      TESTCHECK(InitData(((char*)args->expected[i])+args->sendBytes*j, sendcount, type, rep, j));
    }
    CUDACHECK(cudaDeviceSynchronize());
  }
  return testSuccess;
}

void HyperCubeGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize * (nranks - 1)) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = 1;
  *busBw = baseBw * factor;
}

testResult_t HyperCubeRunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  char* sbuff = (char*)sendbuff;
  char* rbuff = (char*)recvbuff;
  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  int rank;
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  size_t rankSize = count * wordSize(type);

  if (rbuff+rank*rankSize != sbuff) CUDACHECK(cudaMemcpyAsync(rbuff+rank*rankSize, sbuff, rankSize, cudaMemcpyDeviceToDevice, stream));

  // Hypercube AllGather
  for (int mask=1; mask<nRanks; mask<<=1) {
    NCCLCHECK(ncclGroupStart());
    int s = rank & ~(mask-1);
    int r = s ^ mask;
    NCCLCHECK(ncclSend(rbuff+s*rankSize, count*mask, type, rank^mask, comm, stream));
    NCCLCHECK(ncclRecv(rbuff+r*rankSize, count*mask, type, rank^mask, comm, stream));
    NCCLCHECK(ncclGroupEnd());
  }
  return testSuccess;
}

struct testColl hyperCubeTest = {
  "HyperCube",
  HyperCubeGetCollByteCount,
  HyperCubeInitData,
  HyperCubeGetBw,
  HyperCubeRunColl
};

void HyperCubeGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  HyperCubeGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t HyperCubeRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &hyperCubeTest;
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
    TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "", -1));
  }
  return testSuccess;
}

struct testEngine hyperCubeEngine = {
  HyperCubeGetBuffSize,
  HyperCubeRunTest
};

#pragma weak ncclTestEngine=hyperCubeEngine
