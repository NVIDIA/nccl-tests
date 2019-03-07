/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef __COMMON_H__
#define __COMMON_H__

#include "nccl.h"
#include <stdio.h>
#include <algorithm>
#include <curand.h>
#ifdef MPI_SUPPORT
#include "mpi.h"
#endif
#include <pthread.h>
#include "nccl1_compat.h"

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf("%s: Test CUDA failure %s:%d '%s'\n",    \
         hostname,                                  \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    return testCudaError;                           \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf("%s: Test NCCL failure %s:%d '%s'\n",    \
         hostname,                                  \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    return testNcclError;                           \
  }                                                 \
} while(0)

typedef enum {
  testSuccess = 0,
  testInternalError = 1,
  testCudaError = 2,
  testNcclError = 3,
  testCuRandError = 4
} testResult_t;

// Relay errors up and trace
#define TESTCHECK(cmd) do {                         \
  testResult_t r = cmd;                             \
  if (r!= testSuccess) {                            \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf(" .. %s: Test failure %s:%d\n",          \
         hostname,                                  \
        __FILE__,__LINE__);                         \
    return r;                                       \
  }                                                 \
} while(0)

struct testColl {
  const char name[20];
  void (*getCollByteCount)(
      size_t *sendcount, size_t *recvcount, size_t *paramcount,
      size_t *sendInplaceOffset, size_t *recvInplaceOffset,
      size_t count, int nranks);
  testResult_t (*initData)(struct threadArgs* args, ncclDataType_t type,
      ncclRedOp_t op, int root, int rep, int in_place);
  void (*getBw)(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks);
  testResult_t (*runColl)(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type,
      ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
};
extern struct testColl allReduceTest;
extern struct testColl allGatherTest;
extern struct testColl reduceScatterTest;
extern struct testColl broadcastTest;
extern struct testColl reduceTest;

struct testEngine {
  void (*getBuffSize)(size_t *sendcount, size_t *recvcount, size_t count, int nranks);
  testResult_t (*runTest)(struct threadArgs* args, int root, ncclDataType_t type,
      const char* typeName, ncclRedOp_t op, const char* opName);
};

extern struct testEngine ncclTestEngine;

struct threadArgs {
  size_t nbytes;
  size_t minbytes;
  size_t maxbytes;
  size_t stepbytes;
  size_t stepfactor;

  int nProcs;
  int proc;
  int nThreads;
  int thread;
  int nGpus;
  int localRank;
  void** sendbuffs;
  size_t sendBytes;
  size_t sendInplaceOffset;
  void** recvbuffs;
  size_t recvInplaceOffset;
  ncclUniqueId ncclId;
  ncclComm_t* comms;
  cudaStream_t* streams;

  void** expected;
  size_t expectedBytes;
  volatile int* sync;
  int sync_idx;
  volatile int* barrier;
  int barrier_idx;
  int syncRank;
  int syncNranks;
  double* deltaThreads;
  double* deltaHost;
  double* delta;
  int* errors;
  double* bw;
  int* bw_count;

  struct testColl* collTest;
};

typedef testResult_t (*threadFunc_t)(struct threadArgs* args);
struct testThread {
  pthread_t thread;
  threadFunc_t func;
  struct threadArgs args;
  testResult_t ret;
};

#include <chrono>

// Provided by common.cu
extern void Barrier(struct threadArgs* args);
extern testResult_t TimeTest(struct threadArgs* args, ncclDataType_t type, const char* typeName, ncclRedOp_t op,  const char* opName, int root);
extern testResult_t InitDataReduce(void* data, const size_t count, const size_t offset, ncclDataType_t type, ncclRedOp_t op, const int rep, const int nranks);
extern testResult_t InitData(void* data, const size_t count, ncclDataType_t type, const int rep, const int rank);
extern void AllocateBuffs(void **sendbuff, void **recvbuff, void **expected, void **expectedHost, size_t nbytes, int nranks);

// Provided by each coll
extern void print_line_header (size_t size, size_t count, const char *typeName, const char *opName, int root);
extern void print_header();

#include <unistd.h>

static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}

#include <stdint.h>

static uint64_t getHostHash(const char* string) {
  // Based on DJB2, result = result * 33 + char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) + string[c];
  }
  return result;
}

static size_t wordSize(ncclDataType_t type) {
  switch(type) {
    case ncclChar:
#if NCCL_MAJOR >= 2
    //case ncclInt8:
    case ncclUint8:
#endif
      return 1;
    case ncclHalf:
    //case ncclFloat16:
      return 2;
    case ncclInt:
    case ncclFloat:
#if NCCL_MAJOR >= 2
    //case ncclInt32:
    case ncclUint32:
    //case ncclFloat32:
#endif
      return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclDouble:
    //case ncclFloat64: 
      return 8;
    default: return 0;
  }
}

extern ncclDataType_t test_types[ncclNumTypes];
extern const char *test_typenames[ncclNumTypes];
extern ncclRedOp_t test_ops[ncclNumOps];
extern const char *test_opnames[ncclNumOps];

static int ncclstringtotype(char *str) {
    for (int t=0; t<ncclNumTypes; t++) {
      if (strcmp(str, test_typenames[t]) == 0) {
        return t;
      }
    }
    if (strcmp(str, "all") == 0) {
      return -1;
    }
    printf("invalid type %s, defaulting to %s .. \n", str, test_typenames[ncclFloat]);
    return ncclFloat;
}

static int ncclstringtoop (char *str) {
    for (int o=0; o<ncclNumOps; o++) {
      if (strcmp(str, test_opnames[o]) == 0) {
        return o;
      }
    }
    if (strcmp(str, "all") == 0) {
      return -1;
    }
    printf("invalid op %s, defaulting to %s .. \n", str, test_opnames[ncclSum]);
    return ncclSum;
}

extern thread_local int is_main_thread;
#define PRINT if (is_main_thread) printf

#endif
