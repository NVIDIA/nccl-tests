#include "cuda_runtime.h"
#include "common_simple.h"
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>

void print_header() {
  PRINT("# %10s  %12s  %8s  %6s            out-of-place                       in-place          \n", "", "", "", "");
  PRINT("# %10s  %12s  %8s  %6s  %7s  %6s  %6s  %5s  %7s  %6s  %6s  %5s\n", "size", "count", "type", "root",
        "time", "algbw", "busbw", "error", "time", "algbw", "busbw", "error");
  PRINT("# %10s  %12s  %8s  %6s  %7s  %6s  %6s  %5s  %7s  %6s  %6s  %5s\n", "(B)", "(elements)", "", "",
        "(us)", "(GB/s)", "(GB/s)", "", "(us)", "(GB/s)", "(GB/s)", "");
}

void print_line_header (size_t size, size_t count, const char *typeName, const char *opName, int root) {
  PRINT("%12li  %12li  %8s  %6i", size, count, typeName, root);
}

void BroadcastGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, int nranks) {
  *sendcount = count;
  *recvcount = count;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = *sendcount;
}

testResult_t BroadcastInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);

  for (int i=0; i<args->nGpus; i++) {
    int gpuid = args->localRank*args->nThreads*args->nGpus + args->thread*args->nGpus + i;
    CUDACHECK(cudaSetDevice(gpuid));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    if (rank == root) TESTCHECK(InitData(data, sendcount, type, rep, rank));
    TESTCHECK(InitData(args->expected[i], recvcount, type, rep, root));
    CUDACHECK(cudaDeviceSynchronize());
  }
  return testSuccess;
}

void BroadcastGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = 1;
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
  // OFTEST_LOG(TEST, "<%lu> Rank<%d>, callback get %dth cqe for coll_id = %d", pthread_self(), cudaDev, ((CallBackArgs *)args)->cqeCnt++, collId);

  pthread_mutex_unlock(&(((CallBackArgs *)args)->mutex));
  return 0;
}

testResult_t BroadcastRunColl(void* sendbuff, void* recvbuff, int collId, CallBackArgs *args, ofcclRankCtx_t rankCtx) {
  args->collId = collId;
  args->gotCqe = 0;
  pthread_mutex_init(&args->mutex, NULL);
  NCCLCHECK(ofcclRunBroadcast(sendbuff, recvbuff, collId, myCallback, args, rankCtx));

  // int cudaDev;
  // CUDACHECK(cudaGetDevice(&cudaDev));
  // OFTEST_LOG(TEST, "<%lu> Rank<%d>, invoke ofcclRunBroadcast for coll_id = %d with args @ %p", pthread_self(), cudaDev, collId, args);
  // OFTEST_LOG(TEST, "<%lu> Rank<%d>, invoke ofcclRunBroadcast sendbuff @ %p, recvbuff @ %p", pthread_self(), cudaDev, sendbuff, recvbuff);
  
  return testSuccess;
}

testResult_t BroadcastPrepare(size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm* comm, int collId, ofcclRankCtx_t rankCtx) {

  NCCLCHECK(ofcclPrepareBroadcast(count, datatype, root, comm, collId, rankCtx));
  OFTEST_LOG(TEST, "tid<%lu> invoke ofcclPrepareBroadcast with count=%lu, collId=%d", pthread_self(), count, collId);
  return testSuccess;
}

struct testColl broadcastTest = {
  "Broadcast",
  BroadcastGetCollByteCount,
  BroadcastInitData,
  BroadcastGetBw,
  BroadcastRunColl,
  BroadcastPrepare
};

void BroadcastGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  BroadcastGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t BroadcastRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &broadcastTest;
  ncclDataType_t *run_types;
  const char **run_typenames;
  int type_count;
  int begin_root, end_root;

  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = test_typenum;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  if (root != -1) {
    begin_root = end_root = root;
  } else {
    begin_root = 0;
    end_root = args->nProcs*args->nThreads*args->nGpus-1;
  }

  for (int i=0; i<type_count; i++) {
    for (int j=begin_root; j<=end_root; j++) {
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "", j));
    }
  }
  return testSuccess;
}

struct testEngine broadcastEngine = {
  BroadcastGetBuffSize,
  BroadcastRunTest
};

#pragma weak ncclTestEngine=broadcastEngine


