#include "cuda_runtime.h"
#include "common_simple.h"
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>

void print_header() {
  PRINT("# %10s  %12s  %8s  %6s            out-of-place                       in-place          \n", "", "", "", "");
  PRINT("# %10s  %12s  %8s  %6s  %7s  %6s  %6s  %5s  %7s  %6s  %6s  %5s\n", "size", "count", "type", "redop",
        "time", "algbw", "busbw", "error", "time", "algbw", "busbw", "error");
  PRINT("# %10s  %12s  %8s  %6s  %7s  %6s  %6s  %5s  %7s  %6s  %6s  %5s\n", "(B)", "(elements)", "", "",
        "(us)", "(GB/s)", "(GB/s)", "", "(us)", "(GB/s)", "(GB/s)", "");
}

void print_line_header (size_t size, size_t count, const char *typeName, const char *opName, int root) {
  PRINT("%12li  %12li  %8s  %6s", size, count, typeName, opName);
}

void ReduceScatterGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, int nranks) {
  *sendcount = (count/nranks)*nranks;
  *recvcount = count/nranks;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = count/nranks;
  *paramcount = *recvcount;
}

testResult_t ReduceScatterInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    int gpuid = args->localRank*args->nThreads*args->nGpus + args->thread*args->nGpus + i;
    CUDACHECK(cudaSetDevice(gpuid));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, type, rep, rank));
    CUDACHECK(cudaMemcpy(args->expected[i], args->recvbuffs[i], args->expectedBytes, cudaMemcpyDefault));
    TESTCHECK(InitDataReduce(args->expected[i], recvcount, rank*recvcount, type, op, rep, nranks));
    CUDACHECK(cudaDeviceSynchronize());
  }
  return testSuccess;
}

void ReduceScatterGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize * nranks) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks - 1))/((double)nranks);
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

  // int cudaDev;
  // CUDACHECK(cudaGetDevice(&cudaDev));
  // int collId = ((CallBackArgs *)args)->collId;
  // OFTEST_LOG(TEST, "<%lu> Rank<%d>, callback get cqe for coll_id = %d", pthread_self(), cudaDev, collId);
  // OFTEST_LOG(TEST, "<%lu> Rank<%d>, callback get %dth cqe for coll_id = %d", pthread_self(), cudaDev, ((CallBackArgs *)args)->cqeCnt++, collId);

  pthread_mutex_unlock(&(((CallBackArgs *)args)->mutex));
  return 0;
}

testResult_t ReduceScatterRunColl(void* sendbuff, void* recvbuff, int collId, CallBackArgs *args, ofcclRankCtx_t rankCtx) {
  args->collId = collId;
  args->gotCqe = 0;
  pthread_mutex_init(&args->mutex, NULL);
  NCCLCHECK(ofcclRunReduceScatter(sendbuff, recvbuff, collId, myCallback, args, rankCtx));

  // int cudaDev;
  // CUDACHECK(cudaGetDevice(&cudaDev));
  // OFTEST_LOG(TEST, "<%lu> Rank<%d>, invoke ofcclRunReduceScatter for coll_id = %d with args @ %p", pthread_self(), cudaDev, collId, args);
  // OFTEST_LOG(TEST, "<%lu> Rank<%d>, invoke ofcclRunReduceScatter sendbuff @ %p, recvbuff @ %p", pthread_self(), cudaDev, sendbuff, recvbuff);
  
  return testSuccess;
}

testResult_t ReduceScatterPrepare(size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, int collId, ofcclRankCtx_t rankCtx) {

  NCCLCHECK(ofcclPrepareReduceScatter(count, datatype, op, comm, collId, rankCtx));
  // OFTEST_LOG(TEST, "tid<%lu> invoke ofcclPrepareReduceScatter with count=%lu, collId=%d", pthread_self(), count, collId);
  return testSuccess;
}

struct testColl reduceScatterTest = {
  "ReduceScatter",
  ReduceScatterGetCollByteCount,
  ReduceScatterInitData,
  ReduceScatterGetBw,
  ReduceScatterRunColl,
  ReduceScatterPrepare
};

void ReduceScatterGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  ReduceScatterGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t ReduceScatterRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &reduceScatterTest;
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

struct testEngine reduceScatterEngine = {
  ReduceScatterGetBuffSize,
  ReduceScatterRunTest
};

#pragma weak ncclTestEngine=reduceScatterEngine



