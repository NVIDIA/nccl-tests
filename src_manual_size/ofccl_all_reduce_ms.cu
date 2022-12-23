/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common_ms.h"
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>

void print_header() {
  PRINT("# %10s  %12s  %8s  %6s            out-of-place                       in-place          \n", "", "", "", "\n");
  PRINT("# %10s  %12s  %8s  %6s  %7s  %6s  %6s  %5s  %7s  %6s  %6s  %5s\n", "size", "count", "type", "redop",
        "time", "algbw", "busbw", "error", "time", "algbw", "busbw", "error\n");
  PRINT("# %10s  %12s  %8s  %6s  %7s  %6s  %6s  %5s  %7s  %6s  %6s  %5s\n", "(B)", "(elements)", "", "",
        "(us)", "(GB/s)", "(GB/s)", "", "(us)", "(GB/s)", "(GB/s)", "\n");
}

void print_line_header (size_t size, size_t count, const char *typeName, const char *opName, int root) {
  PRINT("%12li  %12li  %8s  %6s", size, count, typeName, opName);
}

void AllReduceGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, int nranks) {
  int cudaDev;
  cudaGetDevice(&cudaDev);
  OFTEST_LOG(TEST, "Hi <%lu> Rank<%d>, sendcount = %p, recvcount = %p, paramcount = %p, sendInplaceOffset = %p, recvInplaceOffset = %p, count = %lu, nranks = %d", pthread_self(), cudaDev, sendcount, recvcount, paramcount, sendInplaceOffset, recvInplaceOffset, count, nranks);

  *sendcount = count;
  *recvcount = count;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = *sendcount;
}

void AllReduceGetCollByteCountList(size_t *sendCntList, size_t *recvCntList, const size_t *countList, int listLen) { // listLen就等于multi_iter
  // OFTEST_LOG1(TEST, "hi");
  for (int i = 0; i < listLen; i++) {
    *(sendCntList + i) = *(countList + i);
    *(recvCntList + i) = *(countList + i);
  }
}

testResult_t AllReduceInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));

  for (int i=0; i<args->nGpus; i++) {
    int gpuid = args->localRank*args->nThreads*args->nGpus + args->thread*args->nGpus + i;
    CUDACHECK(cudaSetDevice(gpuid));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, type, rep, rank));
    TESTCHECK(InitDataReduce(args->expected[i], recvcount, 0, type, op, rep, nranks));
    CUDACHECK(cudaDeviceSynchronize());
  }
  // OFTEST_LOG(TEST, "<%lu> Rank<%d>, done AllReduceInitData", pthread_self(), cudaDev);
  return testSuccess;
}

void AllReduceGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(2*(nranks - 1)))/((double)nranks);
  *busBw = baseBw * factor;
}

int myCallback(int collIdFromCqe, void *args) {
  // 不打log把这里删了，不然影响性能。
  // if (collId != collIdFromCqe) {
  //   // more robust error handle.
  //   OFTEST_LOG(TEST_ERROR, "<%lu> Rank<%d>, collIdFromCqe(%d) is not expected(%d)", pthread_self(), cudaDev, collIdFromCqe, collId);
  //   return -1;
  // }
  pthread_mutex_lock(&(((CallBackArgs *)args)->mutex));
  ((CallBackArgs *)args)->gotCqe = 1;
  pthread_mutex_unlock(&(((CallBackArgs *)args)->mutex));

  // int cudaDev;
  // CUDACHECK(cudaGetDevice(&cudaDev)); // 这个函数之后在poller线程里调用的，所以这个获得的dev应该是不对的。

  // int collId = ((CallBackArgs *)args)->collId;
  // int cudaDev = ((CallBackArgs *)args)->cudaDev;
  // OFTEST_LOG(TEST, "<%lu> Rank<%d>, callback get cqe for coll_id = %d", pthread_self(), cudaDev, collId);

  return 0;
}

testResult_t AllReduceRunColl(void* sendbuff, void* recvbuff, int collId, CallBackArgs *args, ofcclRankCtx_t rankCtx) {
  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));

  args->collId = collId;
  args->gotCqe = 0;
  args->cudaDev = cudaDev;
  pthread_mutex_init(&args->mutex, NULL);

  NCCLCHECK(ofcclRunAllReduce(sendbuff, recvbuff, collId, myCallback, args, rankCtx));
  // OFTEST_LOG(TEST, "<%lu> Rank<%d>, invoke ofcclRunAllReduce for coll_id = %d", pthread_self(), cudaDev, collId);
  // OFTEST_LOG(TEST, "<%lu> Rank<%d>, invoke ofcclRunAllReduce sendbuff @ %p, recvbuff @ %p", pthread_self(), cudaDev, sendbuff, recvbuff);
  
  return testSuccess;
}

testResult_t AllReducePrepare(size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, int collId, ofcclRankCtx_t rankCtx) {

  NCCLCHECK(ofcclPrepareAllReduce(count, datatype, op, comm, collId, rankCtx));
  // OFTEST_LOG(TEST, "tid<%lu> invoke ofcclPrepareAllReduce with count=%lu, collId=%d", pthread_self(), count, collId);
  return testSuccess;
}

struct testColl allReduceTest = {
  "AllReduce",
  AllReduceGetCollByteCount,
  AllReduceInitData,
  AllReduceGetBw,
  AllReduceRunColl,
  AllReducePrepare
};

void AllReduceGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AllReduceGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t AllReduceRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &allReduceTest;
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
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], run_ops[j], run_opnames[j], -1, true));
    }
  }
  return testSuccess;
}

struct testEngine allReduceEngine = {
  AllReduceGetBuffSize,
  AllReduceRunTest,
  AllReduceGetCollByteCountList
};

#pragma weak ncclTestEngine=allReduceEngine