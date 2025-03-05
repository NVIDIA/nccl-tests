/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"

void BisectionGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, int nranks) {
  *sendcount = count;
  *recvcount = count;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = count;
}

int getPeer(int rank, int n_ranks){
    return (rank + n_ranks/2) % n_ranks;
}


testResult_t BisectionInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  if (nranks % 2 != 0){
    print("Bisection test should run on an even number of ranks.\n");
    return testNcclError;
  }

  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, rank*sendcount, type, ncclSum, rep, 1, 0));
    int peer = getPeer(rank, nranks);
    TESTCHECK(InitData(args->expected[i], recvcount, peer*recvcount, type, ncclSum, rep, 1, 0));
    CUDACHECK(cudaDeviceSynchronize());
  }
  // We don't support in-place sendrecv
  args->reportErrors = in_place ? 0 : 1;
  return testSuccess;
}

void BisectionGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  *busBw = *algBw = (double)(count * typesize) / 1.0E9 / sec;
}

testResult_t BisectionRunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  int n_ranks, comm_rank, peer;

  NCCLCHECK(ncclCommUserRank(comm, &comm_rank));
  NCCLCHECK(ncclCommCount(comm, &n_ranks));

  peer = getPeer(comm_rank, n_ranks);

  NCCLCHECK(ncclGroupStart());
  NCCLCHECK(ncclSend(sendbuff, count, type, peer, comm, stream));
  NCCLCHECK(ncclRecv(recvbuff, count, type, peer, comm, stream));
  NCCLCHECK(ncclGroupEnd());

  return testSuccess;
}

struct testColl bisectionTest = {
  "Bisection",
  BisectionGetCollByteCount,
  BisectionInitData,
  BisectionGetBw,
  BisectionRunColl
};

void BisectionGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  BisectionGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t BisectionRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &bisectionTest;
  ncclDataType_t *run_types;
  ncclRedOp_t *run_ops;
  const char **run_typenames, **run_opnames;
  int type_count, op_count;

  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = test_typenum;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  if ((int)op != -1) {
    op_count = 1;
    run_ops = &op;
    run_opnames = &opName;
  } else {
    op_count = test_opnum;
    run_ops = test_ops;
    run_opnames = test_opnames;
  }

  for (int i=0; i<type_count; i++) {
    for (int j=0; j<op_count; j++) {
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], run_ops[j], run_opnames[j], -1));
    }
  }
  return testSuccess;
}

struct testEngine bisectionEngine = {
  BisectionGetBuffSize,
  BisectionRunTest
};

#pragma weak ncclTestEngine=bisectionEngine
