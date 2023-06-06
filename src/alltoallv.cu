/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"


void AlltoAllvGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, int nranks) {
  *sendcount = (count/nranks)*nranks; //each rank in a2av should be able to send up to count to all of the others combined. 
  *recvcount = (count/nranks)*nranks; //each rank in a2av should be able to receive up to count from all of its peers.
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = count/nranks; //each rank in a2av gets one even chunk to send out.
}

testResult_t AlltoAllvInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes)); //zeroes out the receive buffer of each GPU with total size (recvcount*wordSize(type))
    CUDACHECK(cudaMemcpy(args->expected[i], args->recvbuffs[i], args->expectedBytes, cudaMemcpyDefault)); //copies the zeroed out receive buffer to the expected buffer
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i); //current rank
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, 33*rep + rank, 1, 0)); //initializes the sendbuffer data for this rank 
    for (int j=0; j<nranks; j++) {
      //j == peer rank 
      size_t partcount = sendcount/nranks; //create chunk definition to use in offsetting the data initialization
      size_t partcount_mod = (partcount - j - rank - 1) % partcount; //imbalance the count of data to initialize same way we do in the test
      TESTCHECK(InitData((char*)args->expected[i] + j*partcount*wordSize(type), partcount_mod, rank*partcount, type, ncclSum, 33*rep + j, 1, 0));
    }
    CUDACHECK(cudaDeviceSynchronize());
  }
  // We don't support in-place alltoallv
  args->reportErrors = in_place ? 0 : 1;
  return testSuccess;
}

void AlltoAllvGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks-1))/((double)(nranks));
  *busBw = baseBw * factor;
}

testResult_t AlltoAllvRunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  int nRanks, myRank;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  NCCLCHECK(ncclCommUserRank(comm, &myRank));
  size_t rankOffset = count * wordSize(type);

#if NCCL_MAJOR < 2 || NCCL_MINOR < 7
  printf("NCCL 2.7 or later is needed for alltoallv. This test was compiled with %d.%d.\n", NCCL_MAJOR, NCCL_MINOR);
  return testNcclError;
#else
  NCCLCHECK(ncclGroupStart());


  for (int r=0; r<nRanks; r++) {
    int count_mod = (count-myRank-r-1) % count; //modify the count variable to to be strictly less than count, but depend on both the peer rank and the sending rank
    NCCLCHECK(ncclSend(((char*)sendbuff)+r*rankOffset, count_mod, type, r, comm, stream));
    NCCLCHECK(ncclRecv(((char*)recvbuff)+r*rankOffset, count_mod, type, r, comm, stream));
  }


  NCCLCHECK(ncclGroupEnd());
  return testSuccess;
#endif
}

struct testColl AlltoAllvTest = {
  "AlltoAllV",
  AlltoAllvGetCollByteCount,
  AlltoAllvInitData,
  AlltoAllvGetBw,
  AlltoAllvRunColl
};

void AlltoAllvGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AlltoAllvGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t AlltoAllvRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &AlltoAllvTest;
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

struct testEngine AlltoAllvEngine = {
  AlltoAllvGetBuffSize,
  AlltoAllvRunTest
};

#pragma weak ncclTestEngine=AlltoAllvEngine
