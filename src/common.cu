/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.h"
#include <pthread.h>
#include <cstdio>
#include <getopt.h>
#include <libgen.h>
#include "cuda.h"

#if NCCL_MAJOR >= 2
ncclDataType_t test_types[ncclNumTypes] = {ncclInt8, ncclUint8, ncclInt32, ncclUint32, ncclInt64, ncclUint64, ncclHalf, ncclFloat, ncclDouble};
const char *test_typenames[ncclNumTypes] = {"int8", "uint8", "int32", "uint32", "int64", "uint64", "half", "float", "double"};
#else
ncclDataType_t test_types[ncclNumTypes] = {ncclChar, ncclInt, ncclHalf, ncclFloat, ncclDouble, ncclInt64, ncclUint64};
const char *test_typenames[ncclNumTypes] = {"char", "int", "half", "float", "double", "int64", "uint64"};
#endif
ncclRedOp_t test_ops[ncclNumOps] = {ncclSum, ncclProd, ncclMax, ncclMin};
const char *test_opnames[ncclNumOps] = {"sum", "prod", "max", "min"};

thread_local int is_main_thread = 0;

// Command line parameter defaults
static int nThreads = 1;
static int nGpus = 1;
static size_t minBytes = 32*1024*1024;
static size_t maxBytes = 32*1024*1024;
static size_t stepBytes = 1*1024*1024;
static size_t stepFactor = 1;
static int datacheck = 1;
static int warmup_iters = 5;
static int iters = 20;
static int agg_iters = 1;
static int ncclop = ncclSum;
static int nccltype = ncclFloat;
static int ncclroot = 0;
static int parallel_init = 0;
static int blocking_coll = 0;

double parsesize(char *value) {
    long long int units;
    double size;

    if (strchr(value, 'G') != NULL) {
        units=1024*1024*1024;
    } else if (strchr(value, 'M') != NULL) {
        units=1024*1024;
    } else if (strchr(value, 'K') != NULL) {
        units=1024;
    } else {
        units=1;
    }

    size = atof(value)*units;
    return size;
}

double DeltaMaxValue(ncclDataType_t type) {
  switch(type) {
    case ncclHalf: return 1e-2;
    case ncclFloat: return 1e-5;
    case ncclDouble: return 1e-12;
    case ncclInt:
#if NCCL_MAJOR >= 2
    case ncclUint8:
    //case ncclInt32:
    case ncclUint32:
#endif
    case ncclInt64:
    case ncclUint64: return 1e-200;
  }
  return 1e-200;
}

template<typename T> __device__
double absDiff(T a, T b) {
  return fabs((double)(b - a));
}

template<> __device__
double absDiff<half>(half a, half b) {
  float x = __half2float(a);
  float y = __half2float(b);
  return fabs((double)(y-x));
}

template<typename T> __device__
float toFloat(T a) {
  return (float)a;
}
template<> __device__
float toFloat(half a) {
  return __half2float(a);
}

template<typename T, int BSIZE> __global__
void deltaKern(void* A_, void* B_, size_t count, double* max) {
  const T* A = (const T*)A_;
  const T* B = (const T*)B_;
  __shared__ double temp[BSIZE];
  int tid = threadIdx.x;
  double locmax = 0.0;
  for(int i=tid; i<count; i+=blockDim.x) {

    double delta = absDiff(A[i], B[i]);
    if( delta > locmax ) {
      locmax = delta;
#ifdef DEBUG_PRINT
      if (delta > .1) printf("Error at %d/%ld : %f != %f\n", i, count, toFloat(A[i]), toFloat(B[i]));
#endif
    }
  }

  temp[tid] = locmax;
  for(int stride = BSIZE/2; stride > 1; stride>>=1) {
    __syncthreads();
    if( tid < stride )
      temp[tid] = temp[tid] > temp[tid+stride] ? temp[tid] : temp[tid+stride];
  }
  __syncthreads();
  if( threadIdx.x == 0)
    *max = temp[0] > temp[1] ? temp[0] : temp[1];
}


testResult_t CheckDelta(void* expected, void* results, size_t count, ncclDataType_t type, double* devmax) {
  switch (type) {
    case ncclHalf:
      deltaKern<half, 512><<<1, 512>>>(results, expected, count, devmax); break;
    case ncclFloat:
      deltaKern<float, 512><<<1, 512>>>(results, expected, count, devmax); break;
    case ncclDouble:
      deltaKern<double, 512><<<1, 512>>>(results, expected, count, devmax); break;

    case ncclChar:
#if NCCL_MAJOR >= 2
    case ncclUint8:
#endif
      deltaKern<uint8_t, 512><<<1, 512>>>(results, expected, count, devmax); break;
    case ncclInt:
#if NCCL_MAJOR >= 2
    case ncclUint32:
#endif
      deltaKern<uint32_t, 512><<<1, 512>>>(results, expected, count, devmax); break;
    case ncclInt64:
    case ncclUint64:
      deltaKern<uint64_t, 512><<<1, 512>>>(results, expected, count, devmax); break;
  }
  CUDACHECK(cudaDeviceSynchronize());
  return testSuccess;
}

// For integer values, we use values between 0 and 255
template<typename T>
__device__ T testValue(const size_t offset, const int rep, const int rank) {
  uint8_t v = (rep+rank+offset) % 256;
  return (T)v;
}

// For floating point datatype, we use values between 0 and 1 otherwise the
// Product operation will produce NaNs.
template<>
__device__ double testValue<double>(const size_t offset, const int rep, const int rank) {
  return 1.0/(1.0+(double)testValue<int>(offset, rep, rank));
}
template<>
__device__ float testValue<float>(const size_t offset, const int rep, const int rank) {
  return 1.0/(1.0+(float)testValue<int>(offset, rep, rank));
}
template<>
__device__ half testValue<half>(const size_t offset, const int rep, const int rank) {
  return __float2half(testValue<float>(offset, rep, rank));
}

// Operations
template<typename T>
__device__ T ncclOpSum(T a, T b) { return a+b; }
template<typename T>
__device__ T ncclOpProd(T a, T b) { return a*b; }
template<typename T>
__device__ T ncclOpMax(T a, T b) { return a>b ? a : b; }
template<typename T>
__device__ T ncclOpMin(T a, T b) { return a<b ? a : b; }

// Definitions for half
template<>
__device__ half ncclOpSum(half a, half b) { return __float2half(__half2float(a)+__half2float(b)); }
template<>
__device__ half ncclOpProd(half a, half b) { return __float2half(__half2float(a)*__half2float(b)); }
template<>
__device__ half ncclOpMax(half a, half b) { return __half2float(a)>__half2float(b) ? a : b; }
template<>
__device__ half ncclOpMin(half a, half b) { return __half2float(a)<__half2float(b) ? a : b; }

template<typename T, T (*Op)(T, T)>
__global__ void InitDataReduceKernel(T* data, const size_t N, const size_t offset, const int rep, const int nranks) {
  for (size_t o=blockIdx.x*blockDim.x+threadIdx.x; o<N; o+=gridDim.x*blockDim.x) {
    T val = testValue<T>(o+offset, rep, 0);
    for (int i=1; i<nranks; i++) {
      val = Op(val, testValue<T>(o+offset, rep, i));
    }
    data[o] = val;
  }
}

#define KERN(type, op) (void*)InitDataReduceKernel<type, op<type>>
#define OPS(type) KERN(type, ncclOpSum), KERN(type, ncclOpProd), KERN(type, ncclOpMax), KERN(type, ncclOpMin)

static void* const redInitDataKerns[ncclNumOps*ncclNumTypes] = {
  OPS(int8_t), OPS(uint8_t), OPS(int32_t), OPS(uint32_t), OPS(int64_t), OPS(uint64_t), OPS(half), OPS(float), OPS(double)
};

testResult_t InitDataReduce(void* data, const size_t count, const size_t offset, ncclDataType_t type, ncclRedOp_t op, const int rep, const int nranks) {
  dim3 grid = { 32, 1, 1 };
  dim3 block = { 256, 1, 1 };
  void* args[5] = { (void*)&data, (void*)&count, (void*)&offset, (void*)&rep, (void*)&nranks };
  CUDACHECK(cudaLaunchKernel(redInitDataKerns[type*ncclNumOps+op], grid, block, args, 0, cudaStreamDefault));
  return testSuccess;
}

template<typename T>
__global__ void InitDataKernel(T* data, const size_t N, const int rep, const int rank) {
  for (size_t o=blockIdx.x*blockDim.x+threadIdx.x; o<N; o+=gridDim.x*blockDim.x)
    data[o] = testValue<T>(o, rep, rank);
}

static void* const initDataKerns[ncclNumTypes] = {
  (void*)InitDataKernel<  int8_t>,
  (void*)InitDataKernel< uint8_t>,
  (void*)InitDataKernel< int32_t>,
  (void*)InitDataKernel<uint32_t>,
  (void*)InitDataKernel< int64_t>,
  (void*)InitDataKernel<uint64_t>,
  (void*)InitDataKernel<    half>,
  (void*)InitDataKernel<   float>,
  (void*)InitDataKernel<  double>
};

template<typename T>
testResult_t InitDataType(void* dest, const size_t N, const int rep, const int rank) {
  T* ptr = (T*)dest;
  InitDataKernel<<<16, 512>>>(ptr, N, rep, rank);
  return testSuccess;
}

testResult_t InitData(void* data, const size_t count, ncclDataType_t type, const int rep, const int rank) {
  dim3 grid = { 32, 1, 1 };
  dim3 block = { 256, 1, 1 };
  void* args[4] = { (void*)&data, (void*)&count, (void*)&rep, (void*)&rank };
  CUDACHECK(cudaLaunchKernel(initDataKerns[type], grid, block, args, 0, cudaStreamDefault));
  return testSuccess;
}

void Barrier(struct threadArgs* args)
{
  while (args->barrier[args->barrier_idx] != args->thread) pthread_yield();

  args->barrier[args->barrier_idx] = args->thread + 1;

  if (args->thread+1 == args->nThreads) {
#ifdef MPI_SUPPORT
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    args->barrier[args->barrier_idx] = 0;
  } else {
    while (args->barrier[args->barrier_idx]) pthread_yield();
  }

  args->barrier_idx=!args->barrier_idx;
}

testResult_t CheckData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, double *delta) {
  size_t count = args->expectedBytes/wordSize(type);
  double maxDelta = 0.0;
  for (int i=0; i<args->nGpus; i++) {
    int device;
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    NCCLCHECK(ncclCommCuDevice(args->comms[i], &device));
    CUDACHECK(cudaSetDevice(device));
    void *data = in_place ? ((void *)((uintptr_t)args->recvbuffs[i] + args->recvInplaceOffset*rank)) : args->recvbuffs[i];
    TESTCHECK(CheckDelta(data , args->expected[i], count, type, args->delta));
    maxDelta = std::max(*(args->deltaHost), maxDelta);

#ifdef DEBUG_PRINT
    if (rank == 0) {
       int *expectedHost = (int *)malloc(args->expectedBytes);
       int *dataHost = (int *)malloc(args->expectedBytes);

       cudaMemcpy(expectedHost, args->expected[0], args->expectedBytes, cudaMemcpyDeviceToHost);
       printf("\n Expected: ");
       for(int j=0; j<args->expectedBytes/sizeof(int); j++) {
         printf("%d:%d ", j, expectedHost[j]);
       }
       printf("\n");

       cudaMemcpy(dataHost, data, args->expectedBytes, cudaMemcpyDeviceToHost);
       printf("\n Actual: ");
       for (int j=0; j<args->expectedBytes/sizeof(int); j++) {
         printf("%d:%d ", j, dataHost[j]);
       }
       printf("\n");
       free(temp);
    }
#endif
  }
  double nranks = args->nProcs*args->nThreads*args->nGpus;
  if (maxDelta > DeltaMaxValue(type)*(nranks - 1)) args->errors[0]++;
  *delta = maxDelta;
  return testSuccess;
}

testResult_t testStreamSynchronize(int ngpus, cudaStream_t* streams, ncclComm_t* comms) {
  cudaError_t cudaErr;
  int remaining = ngpus;
  int* done = (int*)malloc(sizeof(int)*ngpus);
  memset(done, 0, sizeof(int)*ngpus);
  while (remaining) {
   int idle = 1;
   for (int i=0; i<ngpus; i++) {
     if (done[i]) continue;

     cudaErr = cudaStreamQuery(streams[i]);
     if (cudaErr == cudaSuccess) {
       done[i] = 1;
       remaining--;
       idle = 0;
       continue;
     }

     if (cudaErr != cudaErrorNotReady) CUDACHECK(cudaErr);

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,4,0)
     if (comms) {
       ncclResult_t ncclAsyncErr;
       NCCLCHECK(ncclCommGetAsyncError(comms[i], &ncclAsyncErr));
       if (ncclAsyncErr != ncclSuccess) {
         // An asynchronous error happened. Stop the operation and destroy
         // the communicator
         for (int i=0; i<ngpus; i++)
           NCCLCHECK(ncclCommAbort(comms[i]));
         // Abort the perf test
         NCCLCHECK(ncclAsyncErr);
       }
     }
#endif
   }

   // We might want to let other threads (including NCCL threads) use the CPU.
   if (idle) pthread_yield();
  }
  return testSuccess;
}

testResult_t startColl(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, int iter) {
  size_t count = args->nbytes / wordSize(type);

  // Try to change offset for each iteration so that we avoid cache effects and catch race conditions in ptrExchange
  size_t totalnbytes = max(args->sendBytes, args->expectedBytes);
  size_t shift = (totalnbytes * iter) % args->maxbytes;
  if (shift + totalnbytes > args->maxbytes) shift = 0;

  if (args->nGpus > 1) NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < args->nGpus; i++) {
#ifndef NCCL_MAJOR
    int cudaDev;
    NCCLCHECK(ncclCommCuDevice(args->comms[i], &cudaDev));
    CUDACHECK(cudaSetDevice(cudaDev));
#endif
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    char* recvBuff = ((char*)args->recvbuffs[i]) + shift;
    char* sendBuff = ((char*)args->sendbuffs[i]) + shift;
    TESTCHECK(args->collTest->runColl(
          (void*)(in_place ? recvBuff + args->sendInplaceOffset*rank : sendBuff),
          (void*)(in_place ? recvBuff + args->recvInplaceOffset*rank : recvBuff),
        count, type, op, root, args->comms[i], args->streams[i]));
  }
  if (args->nGpus > 1) NCCLCHECK(ncclGroupEnd());

  if (blocking_coll) {
    // Complete op before returning
    TESTCHECK(testStreamSynchronize(args->nGpus, args->streams, args->comms));
  }
  if (blocking_coll) Barrier(args);
  return testSuccess;
}

testResult_t completeColl(struct threadArgs* args) {
  if (blocking_coll) return testSuccess;

  TESTCHECK(testStreamSynchronize(args->nGpus, args->streams, args->comms));
  return testSuccess;
}

testResult_t BenchTime(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place) {
  size_t count = args->nbytes / wordSize(type);

  // Sync
  TESTCHECK(startColl(args, type, op, root, in_place, 0));
  TESTCHECK(completeColl(args));

  Barrier(args);

  // Performance Benchmark
  auto start = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < iters; iter++) {
    if (agg_iters>1) NCCLCHECK(ncclGroupStart());
    for (int aiter = 0; aiter < agg_iters; aiter++) {
      TESTCHECK(startColl(args, type, op, root, in_place, iter*agg_iters+aiter));
    }
    if (agg_iters>1) NCCLCHECK(ncclGroupEnd());
  }
  TESTCHECK(completeColl(args));

  auto delta = std::chrono::high_resolution_clock::now() - start;
  double deltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count();
  deltaSec = deltaSec/(iters*agg_iters);

  double algBw, busBw;
  args->collTest->getBw(count, wordSize(type), deltaSec, &algBw, &busBw, args->nProcs*args->nThreads*args->nGpus);

  Barrier(args);

  double maxDelta = 0;
  static __thread int rep = 0;
  rep++;
  if (datacheck) {
      // Initialize sendbuffs, recvbuffs and expected
      TESTCHECK(args->collTest->initData(args, type, op, root, rep, in_place));

      //test validation in single itertion, should ideally be included into the multi-iteration run
      TESTCHECK(startColl(args, type, op, root, in_place, 0));
      TESTCHECK(completeColl(args));

      TESTCHECK(CheckData(args, type, op, root, in_place, &maxDelta));

      //aggregate delta from all threads and procs
      Barrier(args);
      if (args->thread == 0) {
        for (int i=1; i<args->nThreads; i++) {
          maxDelta += args->deltaThreads[i];
        }
#ifdef MPI_SUPPORT
        MPI_Allreduce(MPI_IN_PLACE, &maxDelta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
      }
      Barrier(args);
  }

  double timeUsec = deltaSec*1.0E6;
  char timeStr[10];
  if (timeUsec > 10000.0) {
    sprintf(timeStr, "%7.0f", timeUsec);
  } else if (timeUsec > 100.0) {
    sprintf(timeStr, "%7.1f", timeUsec);
  } else {
    sprintf(timeStr, "%7.2f", timeUsec);
  }
  if (datacheck) {
     PRINT("  %7s  %6.2f  %6.2f  %5.0le", timeStr, algBw, busBw, maxDelta);
  } else {
     PRINT("  %7s  %6.2f  %6.2f  %5s", timeStr, algBw, busBw, "N/A");
  }

  args->bw[0] += busBw;
  args->bw_count[0]++;
  return testSuccess;
}

void setupArgs(size_t size, ncclDataType_t type, struct threadArgs* args) {
  int nranks = args->nProcs*args->nGpus*args->nThreads;
  size_t count, sendCount, recvCount, paramCount, sendInplaceOffset, recvInplaceOffset;

  count = size / wordSize(type);
  args->collTest->getCollByteCount(&sendCount, &recvCount, &paramCount, &sendInplaceOffset, &recvInplaceOffset, (size_t)count, (size_t)nranks);

  args->nbytes = paramCount * wordSize(type);
  args->sendBytes = sendCount * wordSize(type);
  args->expectedBytes = recvCount * wordSize(type);
  args->sendInplaceOffset = sendInplaceOffset * wordSize(type);
  args->recvInplaceOffset = recvInplaceOffset * wordSize(type);
}

testResult_t TimeTest(struct threadArgs* args, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName, int root) {
  // Warm-up for large size
  setupArgs(args->maxbytes, type, args);
  for (int iter = 0; iter < warmup_iters; iter++) {
    TESTCHECK(startColl(args, type, op, root, 0, iter));
  }
  TESTCHECK(completeColl(args));

  // Warm-up for small size
  setupArgs(args->minbytes, type, args);
  for (int iter = 0; iter < warmup_iters; iter++) {
    TESTCHECK(startColl(args, type, op, root, 0, iter));
  }
  TESTCHECK(completeColl(args));

  // Benchmark
  for (size_t size = args->minbytes; size<=args->maxbytes; size = ((args->stepfactor > 1) ? size*args->stepfactor : size+args->stepbytes)) {
      setupArgs(size, type, args);
      print_line_header(max(args->sendBytes, args->expectedBytes), args->nbytes / wordSize(type), typeName, opName, root);
      TESTCHECK(BenchTime(args, type, op, root, 0));
      TESTCHECK(BenchTime(args, type, op, root, 1));
      PRINT("\n");
  }
  return testSuccess;
}

testResult_t threadRunTests(struct threadArgs* args) {
  // Set device to the first of our GPUs. If we don't do that, some operations
  // will be done on the current GPU (by default : 0) and if the GPUs are in
  // exclusive mode those operations will fail.
  int gpuid = args->localRank*args->nThreads*args->nGpus + args->thread*args->nGpus;
  CUDACHECK(cudaSetDevice(gpuid));
  TESTCHECK(ncclTestEngine.runTest(args, ncclroot, (ncclDataType_t)nccltype, test_typenames[nccltype], (ncclRedOp_t)ncclop, test_opnames[ncclop]));
  return testSuccess;
}

testResult_t threadInit(struct threadArgs* args) {
  char hostname[1024];
  getHostName(hostname, 1024);
  int nranks =  args->nProcs*args->nThreads*args->nGpus;

  //set main thread again
  is_main_thread = (args->proc == 0 && args->thread == 0) ? 1 : 0;

  NCCLCHECK(ncclGroupStart());
  for (int i=0; i<args->nGpus; i++) {
    int rank = args->proc*args->nThreads*args->nGpus + args->thread*args->nGpus + i;
    int gpuid = args->localRank*args->nThreads*args->nGpus + args->thread*args->nGpus + i;
    CUDACHECK(cudaSetDevice(gpuid));
    NCCLCHECK(ncclCommInitRank(args->comms+i, nranks, args->ncclId, rank));
  }
  NCCLCHECK(ncclGroupEnd());

  TESTCHECK(threadRunTests(args));

  for (int i=0; i<args->nGpus; i++) {
    NCCLCHECK(ncclCommDestroy(args->comms[i]));
  }
  return testSuccess;
}

void* threadLauncher(void* thread_) {
  struct testThread* thread = (struct testThread*)thread_;
  thread->ret = thread->func(&thread->args);
  return NULL;
}
testResult_t threadLaunch(struct testThread* thread) {
  pthread_create(&thread->thread, NULL, threadLauncher, thread);
  return testSuccess;
}

testResult_t AllocateBuffs(void **sendbuff, size_t sendBytes, void **recvbuff, size_t recvBytes, void **expected, size_t nbytes, int nranks) {
    CUDACHECK(cudaMalloc(sendbuff, nbytes));
    CUDACHECK(cudaMalloc(recvbuff, nbytes));
    CUDACHECK(cudaMalloc(expected, recvBytes));
    return testSuccess;
}

testResult_t run(); // Main function

int main(int argc, char* argv[]) {
  // Make sure everyline is flushed so that we see the progress of the test
  setlinebuf(stdout);

  // Parse args
  int longindex;
  static struct option longopts[] = {
    {"nthreads", required_argument, 0, 't'},
    {"ngpus", required_argument, 0, 'g'},
    {"minbytes", required_argument, 0, 'b'},
    {"maxbytes", required_argument, 0, 'e'},
    {"stepbytes", required_argument, 0, 'i'},
    {"stepfactor", required_argument, 0, 'f'},
    {"iters", required_argument, 0, 'n'},
    {"agg_iters", required_argument, 0, 'm'},
    {"warmup_iters", required_argument, 0, 'w'},
    {"parallel_init", required_argument, 0, 'p'},
    {"check", required_argument, 0, 'c'},
    {"op", required_argument, 0, 'o'},
    {"datatype", required_argument, 0, 'd'},
    {"root", required_argument, 0, 'r'},
    {"blocking", required_argument, 0, 'z'},
    {"help", no_argument, 0, 'h'}
  };

  while(1) {
    int c;
    c = getopt_long(argc, argv, "t:g:b:e:i:f:n:m:w:p:c:o:d:r:z:h", longopts, &longindex);

    if (c == -1)
      break;

    switch(c) {
      case 't':
        nThreads = strtol(optarg, NULL, 0);
        break;
      case 'g':
        nGpus = strtol(optarg, NULL, 0);
        break;
      case 'b':
        minBytes = (size_t)parsesize(optarg);
        break;
      case 'e':
        maxBytes = (size_t)parsesize(optarg);
        break;
      case 'i':
        stepBytes = strtol(optarg, NULL, 0);
        break;
      case 'f':
        stepFactor = strtol(optarg, NULL, 0);
        break;
      case 'n':
        iters = (int)strtol(optarg, NULL, 0);
        break;
      case 'm':
#if NCCL_MAJOR >= 2 && NCCL_MINOR >= 2
        agg_iters = (int)strtol(optarg, NULL, 0);
#else
        printf("Option -m not supported before NCCL 2.2. Ignoring\n");
#endif
        break;
      case 'w':
        warmup_iters = (int)strtol(optarg, NULL, 0);
        break;
      case 'c':
        datacheck = (int)strtol(optarg, NULL, 0);
        break;
      case 'p':
        parallel_init = (int)strtol(optarg, NULL, 0);
        break;
      case 'o':
        ncclop = ncclstringtoop(optarg);
        break;
      case 'd':
        nccltype = ncclstringtotype(optarg);
        break;
      case 'r':
        ncclroot = strtol(optarg, NULL, 0);
        break;
      case 'z':
        blocking_coll = strtol(optarg, NULL, 0);
        break;
      case 'h':
	printf("USAGE: %s \n\t"
            "[-t,--nthreads <num threads>] \n\t"
            "[-g,--ngpus <gpus per thread>] \n\t"
            "[-b,--minbytes <min size in bytes>] \n\t"
            "[-e,--maxbytes <max size in bytes>] \n\t"
            "[-i,--stepbytes <increment size>] \n\t"
            "[-f,--stepfactor <increment factor>] \n\t"
            "[-n,--iters <iteration count>] \n\t"
            "[-m,--agg_iters <aggregated iteration count>] \n\t"
            "[-w,--warmup_iters <warmup iteration count>] \n\t"
            "[-p,--parallel_init <0/1>] \n\t"
            "[-c,--check <0/1>] \n\t"
            "[-o,--op <sum/prod/min/max/all>] \n\t"
            "[-d,--datatype <nccltype/all>] \n\t"
            "[-r,--root <root>] \n\t"
            "[-z,--blocking <0/1>] \n\t"
            "[-h,--help]\n",
	    basename(argv[0]));
	return 0;
      default:
        printf("invalid option \n");
	printf("USAGE: %s \n\t"
            "[-t,--nthreads <num threads>] \n\t"
            "[-g,--ngpus <gpus per thread>] \n\t"
            "[-b,--minbytes <min size in bytes>] \n\t"
            "[-e,--maxbytes <max size in bytes>] \n\t"
            "[-i,--stepbytes <increment size>] \n\t"
            "[-f,--stepfactor <increment factor>] \n\t"
            "[-n,--iters <iteration count>] \n\t"
            "[-m,--agg_iters <aggregated iteration count>] \n\t"
            "[-w,--warmup_iters <warmup iteration count>] \n\t"
            "[-p,--parallel_init <0/1>] \n\t"
            "[-c,--check <0/1>] \n\t"
            "[-o,--op <sum/prod/min/max/all>] \n\t"
            "[-d,--datatype <nccltype/all>] \n\t"
            "[-r,--root <root>] \n\t"
            "[-z,--blocking <0/1>] \n\t"
            "[-h,--help]\n",
	    basename(argv[0]));
	return 0;
    }
  }
#ifdef MPI_SUPPORT
  MPI_Init(&argc, &argv);
#endif
  return run();
}

testResult_t run() {
  int nProcs = 1, proc = 0;
  int localRank = 0;
  char hostname[1024];
  getHostName(hostname, 1024);

#ifdef MPI_SUPPORT
  MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  uint64_t hostHashs[nProcs];
  hostHashs[proc] = getHostHash(hostname);
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);
  for (int p=0; p<nProcs; p++) {
    if (p == proc) break;
    if (hostHashs[p] == hostHashs[proc]) localRank++;
  }
#endif
  is_main_thread = (proc == 0) ? 1 : 0;

  PRINT("# nThread %d nGpus %d minBytes %ld maxBytes %ld step: %ld(%s) warmup iters: %d iters: %d validation: %d \n", nThreads, nGpus, minBytes, maxBytes,
      (stepFactor > 1)?stepFactor:stepBytes, (stepFactor > 1)?"factor":"bytes", warmup_iters, iters, datacheck);
  if (blocking_coll) PRINT("# Blocking Enabled: wait for completion and barrier after each collective \n");
  if (parallel_init) PRINT("# Parallel Init Enabled: threads call into NcclInitRank concurrently \n");
  PRINT("#\n");

  PRINT("# Using devices\n");
#define MAX_LINE 2048
  char line[MAX_LINE];
  int len = 0;
  for (int i=0; i<nThreads*nGpus; i++) {
    int cudaDev = localRank*nThreads*nGpus+i;
    int rank = proc*nThreads*nGpus+i;
    cudaDeviceProp prop;
    CUDACHECK(cudaGetDeviceProperties(&prop, cudaDev));
    len += snprintf(line+len, MAX_LINE-len, "#   Rank %2d Pid %6d on %10s device %2d [0x%02x] %s\n",
                    rank, getpid(), hostname, cudaDev, prop.pciBusID, prop.name);
  }

#if MPI_SUPPORT
  char *lines = (proc == 0) ? (char *)malloc(nProcs*MAX_LINE) : NULL;
  // Gather all output in rank order to root (0)
  MPI_Gather(line, MAX_LINE, MPI_BYTE, lines, MAX_LINE, MPI_BYTE, 0, MPI_COMM_WORLD);
  if (proc == 0) {
    for (int p = 0; p < nProcs; p++)
      PRINT("%s", lines+MAX_LINE*p);
    free(lines);
  }
#else
  PRINT("%s", line);
#endif

  ncclUniqueId ncclId;
  if (proc == 0) {
    NCCLCHECK(ncclGetUniqueId(&ncclId));
  }
#ifdef MPI_SUPPORT
  MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
  cudaStream_t streams[nGpus*nThreads];
  void* sendbuffs[nGpus*nThreads];
  void* recvbuffs[nGpus*nThreads];
  void* expected[nGpus*nThreads];
  size_t sendBytes, recvBytes;

  ncclTestEngine.getBuffSize(&sendBytes, &recvBytes, (size_t)maxBytes, (size_t)nProcs*nGpus*nThreads);

  for (int i=0; i<nGpus*nThreads; i++) {
    CUDACHECK(cudaSetDevice(localRank*nThreads*nGpus+i));
    AllocateBuffs(sendbuffs+i, sendBytes, recvbuffs+i, recvBytes, expected+i, (size_t)maxBytes, nProcs*nThreads*nGpus);
    CUDACHECK(cudaStreamCreateWithFlags(streams+i, cudaStreamNonBlocking));
  }

  //if parallel init is not selected, use main thread to initialize NCCL
  ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t)*nThreads*nGpus);
  if (!parallel_init) {
     if (nProcs == 1) {
       int gpuArray[nGpus*nThreads];
       for (int i=0; i<nGpus*nThreads; i++) gpuArray[i] = i;
       NCCLCHECK(ncclCommInitAll(comms, nGpus*nThreads, gpuArray));
     } else {
       NCCLCHECK(ncclGroupStart());
       for (int i=0; i<nGpus*nThreads; i++) {
         CUDACHECK(cudaSetDevice(localRank*nThreads*nGpus+i));
         NCCLCHECK(ncclCommInitRank(comms+i, nProcs*nThreads*nGpus, ncclId, proc*nThreads*nGpus+i));
       }
       NCCLCHECK(ncclGroupEnd());
     }
  }

  int errors[nThreads];
  double bw[nThreads];
  double* delta;
  CUDACHECK(cudaHostAlloc(&delta, sizeof(double)*nThreads, cudaHostAllocPortable | cudaHostAllocMapped));
  int bw_count[nThreads];
  for (int t=0; t<nThreads; t++) {
    bw[t] = 0.0;
    errors[t] = bw_count[t] = 0;
  }

  PRINT("#\n");
  print_header();

  int* sync = (int*)calloc(2, sizeof(int));
  int* barrier = (int*)calloc(2, sizeof(int));

  struct testThread threads[nThreads];
  memset(threads, 0, sizeof(struct testThread)*nThreads);

  for (int t=nThreads-1; t>=0; t--) {
    threads[t].args.minbytes=minBytes;
    threads[t].args.maxbytes=maxBytes;
    threads[t].args.stepbytes=stepBytes;
    threads[t].args.stepfactor=stepFactor;
    threads[t].args.localRank = localRank;

    threads[t].args.nProcs=nProcs;
    threads[t].args.proc=proc;
    threads[t].args.nThreads=nThreads;
    threads[t].args.thread=t;
    threads[t].args.nGpus=nGpus;
    threads[t].args.sendbuffs = sendbuffs+t*nGpus;
    threads[t].args.recvbuffs = recvbuffs+t*nGpus;
    threads[t].args.expected = expected+t*nGpus;
    threads[t].args.ncclId = ncclId;
    threads[t].args.comms=comms+t*nGpus;
    threads[t].args.streams=streams+t*nGpus;

    threads[t].args.barrier = (volatile int*)barrier;
    threads[t].args.barrier_idx = 0;
    threads[t].args.sync = (volatile int*)sync;
    threads[t].args.sync_idx = 0;
    threads[t].args.deltaThreads = delta;
    threads[t].args.deltaHost = (delta + t);
    threads[t].args.delta = delta;
    threads[t].args.errors=errors+t;
    threads[t].args.bw=bw+t;
    threads[t].args.bw_count=bw_count+t;

    threads[t].func = parallel_init ? threadInit : threadRunTests;
    if (t)
      TESTCHECK(threadLaunch(threads+t));
    else
      TESTCHECK(threads[t].func(&threads[t].args));
  }

  // Wait for other threads and accumulate stats and errors
  for (int t=nThreads-1; t>=0; t--) {
    if (t) pthread_join(threads[t].thread, NULL);
    TESTCHECK(threads[t].ret);
    if (t) {
      errors[0] += errors[t];
      bw[0] += bw[t];
      bw_count[0] += bw_count[t];
    }
  }

#ifdef MPI_SUPPORT
  MPI_Allreduce(MPI_IN_PLACE, &errors[0], 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

  if (!parallel_init) {
    for(int i=0; i<nGpus*nThreads; ++i)
      NCCLCHECK(ncclCommDestroy(comms[i]));
    free(comms);
  }

  // Free off CUDA allocated memory
  for (int i=0; i<nGpus*nThreads; i++) {
    CUDACHECK(cudaFree(sendbuffs[i]));
    CUDACHECK(cudaFree(recvbuffs[i]));
    CUDACHECK(cudaFree(expected[i]));
  }
  CUDACHECK(cudaFreeHost(delta));

  char* str = getenv("NCCL_TESTS_MIN_BW");
  double check_avg_bw = str ? atof(str) : -1;
  bw[0] /= bw_count[0];

  PRINT("# Out of bounds values : %d %s\n", errors[0], errors[0] ? "FAILED" : "OK");
  PRINT("# Avg bus bandwidth    : %g %s\n", bw[0], check_avg_bw == -1 ? "" : (bw[0] < check_avg_bw*(0.9) ? "FAILED" : "OK"));
  PRINT("#\n");
#ifdef MPI_SUPPORT
  MPI_Finalize();
#endif

  // 'cuda-memcheck --leak-check full' requires this
  cudaDeviceReset();

  if (errors[0] || bw[0] < check_avg_bw*(0.9))
    exit(EXIT_FAILURE);
  else
    exit(EXIT_SUCCESS);
}
