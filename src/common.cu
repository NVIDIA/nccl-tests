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

int test_ncclVersion = 0; // init'd with ncclGetVersion()

#if NCCL_MAJOR >= 2
  ncclDataType_t test_types[ncclNumTypes] = {
    ncclInt8, ncclUint8, ncclInt32, ncclUint32, ncclInt64, ncclUint64, ncclHalf, ncclFloat, ncclDouble
  #if defined(__CUDA_BF16_TYPES_EXIST__) && NCCL_VERSION_CODE >= NCCL_VERSION(2,10,0)
    , ncclBfloat16
  #endif
  };
  const char *test_typenames[ncclNumTypes] = {
    "int8", "uint8", "int32", "uint32", "int64", "uint64", "half", "float", "double"
  #if defined(__CUDA_BF16_TYPES_EXIST__) && NCCL_VERSION_CODE >= NCCL_VERSION(2,10,0)
    , "bfloat16"
  #endif
  };
  int test_typenum = -1;

  const char *test_opnames[] = {"sum", "prod", "max", "min", "avg", "mulsum"};
  ncclRedOp_t test_ops[] = {ncclSum, ncclProd, ncclMax, ncclMin
  #if NCCL_VERSION_CODE >= NCCL_VERSION(2,10,0)
    , ncclAvg
  #endif
  #if NCCL_VERSION_CODE >= NCCL_VERSION(2,11,0)
    , ncclNumOps // stand in for ncclRedOpCreatePreMulSum() created on-demand
  #endif
  };
  int test_opnum = -1;
#else
  ncclDataType_t test_types[ncclNumTypes] = {ncclChar, ncclInt, ncclHalf, ncclFloat, ncclDouble, ncclInt64, ncclUint64};
  const char *test_typenames[ncclNumTypes] = {"char", "int", "half", "float", "double", "int64", "uint64"};
  int test_typenum = 7;
  const char *test_opnames[] = {"sum", "prod", "max", "min"};
  ncclRedOp_t test_ops[] = {ncclSum, ncclProd, ncclMax, ncclMin};
  int test_opnum = 4;
#endif

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
static int cudaGraphLaunches = 0;
// Report average iteration time: (0=RANK0,1=AVG,2=MIN,3=MAX)
static int average = 1;

#define NUM_BLOCKS 32

static double parsesize(const char *value) {
    long long int units;
    double size;
    char size_lit;

    int count = sscanf(value, "%lf %1s", &size, &size_lit);

    switch (count) {
    case 2:
      switch (size_lit) {
      case 'G':
      case 'g':
        units = 1024*1024*1024;
        break;
      case 'M':
      case 'm':
        units = 1024*1024;
        break;
      case 'K':
      case 'k':
        units = 1024;
        break;
      default:
        return -1.0;
      };
      break;
    case 1:
      units = 1;
      break;
    default:
      return -1.0;
    }

    return size * units;
}

double DeltaMaxValue(ncclDataType_t type) {
  switch(type) {
    case ncclHalf: return 1e-2;
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16: return 1e-2;
#endif
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
#if defined(__CUDA_BF16_TYPES_EXIST__)
template<> __device__
float toFloat(__nv_bfloat16 a) {
  return __bfloat162float(a);
}
#endif

template<typename T, int BSIZE> __global__
void deltaKern(void* A_, void* B_, size_t count, double* max) {
  const T* A = (const T*)A_;
  const T* B = (const T*)B_;
  __shared__ double temp[BSIZE];
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  double locmax = 0.0;
  for(size_t i=tid; i<count; i+=blockDim.x*gridDim.x) {

    double delta = absDiff(A[i], B[i]);
    if( delta > locmax ) {
      locmax = delta;
#ifdef DEBUG_PRINT
      if (delta > .1) printf("Error at %ld/%ld(%p) : %f != %f\n", i, count, B+i, toFloat(A[i]), toFloat(B[i]));
#endif
    }
  }

  tid = threadIdx.x;
  temp[tid] = locmax;
  for(int stride = BSIZE/2; stride > 1; stride>>=1) {
    __syncthreads();
    if( tid < stride )
      temp[tid] = temp[tid] > temp[tid+stride] ? temp[tid] : temp[tid+stride];
  }
  __syncthreads();
  if( threadIdx.x == 0)
    max[blockIdx.x] = temp[0] > temp[1] ? temp[0] : temp[1];
}

testResult_t CheckDelta(void* results, void* expected, size_t count, ncclDataType_t type, double* devmax) {
  switch (type) {
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16:
      deltaKern<__nv_bfloat16, 512><<<NUM_BLOCKS, 512>>>(results, expected, count, devmax); break;
#endif
    case ncclHalf:
      deltaKern<half, 512><<<NUM_BLOCKS, 512>>>(results, expected, count, devmax); break;
    case ncclFloat:
      deltaKern<float, 512><<<NUM_BLOCKS, 512>>>(results, expected, count, devmax); break;
    case ncclDouble:
      deltaKern<double, 512><<<NUM_BLOCKS, 512>>>(results, expected, count, devmax); break;

    case ncclChar:
#if NCCL_MAJOR >= 2
    case ncclUint8:
#endif
      deltaKern<uint8_t, 512><<<NUM_BLOCKS, 512>>>(results, expected, count, devmax); break;
    case ncclInt:
#if NCCL_MAJOR >= 2
    case ncclUint32:
#endif
      deltaKern<uint32_t, 512><<<NUM_BLOCKS, 512>>>(results, expected, count, devmax); break;
    case ncclInt64:
    case ncclUint64:
      deltaKern<uint64_t, 512><<<NUM_BLOCKS, 512>>>(results, expected, count, devmax); break;
  }
  CUDACHECK(cudaDeviceSynchronize());
  for (int i=1; i<NUM_BLOCKS; i++) devmax[0] = std::max(devmax[0], devmax[i]);
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
#if defined(__CUDA_BF16_TYPES_EXIST__)
template<>
__device__ __nv_bfloat16 testValue<__nv_bfloat16>(const size_t offset, const int rep, const int rank) {
  return __float2bfloat16(testValue<float>(offset, rep, rank));
}
#endif

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

template<typename T>
__device__ T ncclPPOpIdent(T x, int arg) { return x; }
template<typename T>
__device__ T ncclPPOpMul(T x, int arg) { return x*T(arg); }
template<typename T>
__device__ T ncclPPOpDiv(T x, int arg) { return x/T(arg); }
template<>
__device__ half ncclPPOpMul(half x, int arg) {
  return __float2half(__half2float(x)*float(arg));
}
template<>
__device__ half ncclPPOpDiv(half x, int n) {
  return __float2half(__half2float(x)/n);
}
#if defined(__CUDA_BF16_TYPES_EXIST__)
template<>
__device__ __nv_bfloat16 ncclPPOpMul(__nv_bfloat16 x, int arg) {
  return __float2bfloat16(__bfloat162float(x)*float(arg));
}
template<>
__device__ __nv_bfloat16 ncclPPOpDiv(__nv_bfloat16 x, int n) {
  return __float2bfloat16(__bfloat162float(x)/n);
}
#endif

__host__ __device__ int preMulScalar(int rank) {
  return 1 + rank%2;
}

template<typename T, T (*Op)(T, T), T(*PreOp)(T,int), T(*PostOp)(T,int)>
__global__ void InitDataReduceKernel(T* data, const size_t N, const size_t offset, const int rep, const int nranks) {
  for (size_t o=blockIdx.x*blockDim.x+threadIdx.x; o<N; o+=gridDim.x*blockDim.x) {
    T val = testValue<T>(o+offset, rep, 0);
    val = PreOp(val, preMulScalar(0));
    for (int i=1; i<nranks; i++) {
      T val1 = testValue<T>(o+offset, rep, i);
      val1 = PreOp(val1, preMulScalar(i));
      val = Op(val, val1);
    }
    data[o] = PostOp(val, nranks);
  }
}

#define KERN(type, op, preop, postop) (void*)InitDataReduceKernel<type, op<type>, preop<type>, postop<type> >
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,11,0)
  #define OPS(type) \
    KERN(type, ncclOpSum, ncclPPOpIdent, ncclPPOpIdent), \
    KERN(type, ncclOpProd, ncclPPOpIdent, ncclPPOpIdent), \
    KERN(type, ncclOpMax, ncclPPOpIdent, ncclPPOpIdent), \
    KERN(type, ncclOpMin, ncclPPOpIdent, ncclPPOpIdent), \
    KERN(type, ncclOpSum/*Avg*/, ncclPPOpIdent, ncclPPOpDiv), \
    KERN(type, ncclOpSum/*PreMulSum*/, ncclPPOpMul, ncclPPOpIdent)
#elif NCCL_VERSION_CODE >= NCCL_VERSION(2,10,0)
  #define OPS(type) \
    KERN(type, ncclOpSum, ncclPPOpIdent, ncclPPOpIdent), \
    KERN(type, ncclOpProd, ncclPPOpIdent, ncclPPOpIdent), \
    KERN(type, ncclOpMax, ncclPPOpIdent, ncclPPOpIdent), \
    KERN(type, ncclOpMin, ncclPPOpIdent, ncclPPOpIdent), \
    KERN(type, ncclOpSum/*Avg*/, ncclPPOpIdent, ncclPPOpDiv)
#else
  #define OPS(type) \
    KERN(type, ncclOpSum, ncclPPOpIdent, ncclPPOpIdent), \
    KERN(type, ncclOpProd, ncclPPOpIdent, ncclPPOpIdent), \
    KERN(type, ncclOpMax, ncclPPOpIdent, ncclPPOpIdent), \
    KERN(type, ncclOpMin, ncclPPOpIdent, ncclPPOpIdent)
#endif

static void* const redInitDataKerns[test_opNumMax*ncclNumTypes] = {
  OPS(int8_t), OPS(uint8_t), OPS(int32_t), OPS(uint32_t), OPS(int64_t), OPS(uint64_t), OPS(half), OPS(float), OPS(double),
#if defined(__CUDA_BF16_TYPES_EXIST__) && NCCL_VERSION_CODE >= NCCL_VERSION(2,10,0)
  OPS(__nv_bfloat16)
#endif
};

testResult_t InitDataReduce(void* data, const size_t count, const size_t offset, ncclDataType_t type, ncclRedOp_t op, const int rep, const int nranks) {
  dim3 grid = { 32, 1, 1 };
  dim3 block = { 256, 1, 1 };
  void* args[5] = { (void*)&data, (void*)&count, (void*)&offset, (void*)&rep, (void*)&nranks };
  CUDACHECK(cudaLaunchKernel(redInitDataKerns[type*test_opNumMax+op], grid, block, args, 0, cudaStreamDefault));
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
  (void*)InitDataKernel<  double>,
#if defined(__CUDA_BF16_TYPES_EXIST__) && NCCL_VERSION_CODE >= NCCL_VERSION(2,10,0)
  (void*)InitDataKernel<__nv_bfloat16>
#endif
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

void Barrier(struct threadArgs* args) {
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

// Inter-thread/process barrier+allreduce
void Allreduce(struct threadArgs* args, double* value, int average) {
  while (args->barrier[args->barrier_idx] != args->thread) pthread_yield();
  double val = *value;
  if (args->thread > 0) {
    double val2 = args->reduce[args->barrier_idx];
    if (average == 1) val += val2;
    if (average == 2) val = std::min(val, val2);
    if (average == 3) val = std::max(val, val2);
  }
  if (average || args->thread == 0) args->reduce[args->barrier_idx] = val;
  args->barrier[args->barrier_idx] = args->thread + 1;
  if (args->thread+1 == args->nThreads) {
#ifdef MPI_SUPPORT
    if (average != 0) {
      MPI_Op op = average == 1 ? MPI_SUM : average == 2 ? MPI_MIN : MPI_MAX;
      MPI_Allreduce(MPI_IN_PLACE, (void*)&args->reduce[args->barrier_idx], 1, MPI_DOUBLE, op, MPI_COMM_WORLD);
    }
#endif
    if (average == 1) args->reduce[args->barrier_idx] /= args->nProcs*args->nThreads;
    args->reduce[1-args->barrier_idx] = 0;
    args->barrier[args->barrier_idx] = 0;
  } else {
    while (args->barrier[args->barrier_idx]) pthread_yield();
  }
  *value = args->reduce[args->barrier_idx];
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
    TESTCHECK(CheckDelta(data , args->expected[i], count, type, args->deltaHost));
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
       free(expectedHost);
       free(dataHost);
    }
#endif
  }
  double nranks = args->nProcs*args->nThreads*args->nGpus;
  if (args->reportErrors && maxDelta > DeltaMaxValue(type)*(nranks - 1)) args->errors[0]++;
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
     if (test_ncclVersion >= NCCL_VERSION(2,4,0) && comms) {
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
  free(done);
  return testSuccess;
}

testResult_t startColl(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t opIndex, int root, int in_place, int iter) {
  size_t count = args->nbytes / wordSize(type);

  // Try to change offset for each iteration so that we avoid cache effects and catch race conditions in ptrExchange
  size_t totalnbytes = max(args->sendBytes, args->expectedBytes);
  size_t steps = totalnbytes ? args->maxbytes / totalnbytes : 1;
  size_t shift = totalnbytes * (iter % steps);

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
    ncclRedOp_t op;

    if(opIndex < ncclNumOps) {
      op = opIndex;
    }
    #if NCCL_VERSION_CODE >= NCCL_VERSION(2,11,0)
    else {
      union {
        int8_t i8; uint8_t u8; int32_t i32; uint32_t u32; int64_t i64; uint64_t u64;
        half f16; float f32; double f64;
        #if defined(__CUDA_BF16_TYPES_EXIST__)
        __nv_bfloat16 bf16;
        #endif
      };
      int scalar = preMulScalar(rank);
      switch(type) {
      case ncclInt8: i8 = int8_t(scalar); break;
      case ncclUint8: u8 = uint8_t(scalar); break;
      case ncclInt32: i32 = int32_t(scalar); break;
      case ncclUint32: u32 = uint32_t(scalar); break;
      case ncclInt64: i64 = int32_t(scalar); break;
      case ncclUint64: u64 = uint32_t(scalar); break;
      case ncclFloat16: f16 = __float2half(float(scalar)); break;
      case ncclFloat32: f32 = float(scalar); break;
      case ncclFloat64: f64 = double(scalar); break;
      #if defined(__CUDA_BF16_TYPES_EXIST__)
      case ncclBfloat16: bf16 = __float2bfloat16(float(scalar)); break;
      #endif
      }
      NCCLCHECK(ncclRedOpCreatePreMulSum(&op, &u64, type, ncclScalarHostImmediate, args->comms[i]));
    }
    #endif

    TESTCHECK(args->collTest->runColl(
          (void*)(in_place ? recvBuff + args->sendInplaceOffset*rank : sendBuff),
          (void*)(in_place ? recvBuff + args->recvInplaceOffset*rank : recvBuff),
        count, type, op, root, args->comms[i], args->streams[i]));

    #if NCCL_VERSION_CODE >= NCCL_VERSION(2,11,0)
    if(opIndex >= ncclNumOps) {
      NCCLCHECK(ncclRedOpDestroy(op, args->comms[i]));
    }
    #endif
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
  if (datacheck) {
    // Initialize sendbuffs, recvbuffs and expected
    TESTCHECK(args->collTest->initData(args, type, op, root, 99, in_place));
  }

  // Sync
  TESTCHECK(startColl(args, type, op, root, in_place, 0));
  TESTCHECK(completeColl(args));

  Barrier(args);

#if CUDART_VERSION >= 11030
  cudaGraph_t graphs[args->nGpus];
  cudaGraphExec_t graphExec[args->nGpus];
  if (cudaGraphLaunches >= 1) {
    // Begin cuda graph capture
    for (int i=0; i<args->nGpus; i++) {
      // Thread local mode is needed for:
      // - Multi-thread mode
      // - P2P pre-connect
      CUDACHECK(cudaStreamBeginCapture(args->streams[i], cudaStreamCaptureModeThreadLocal));
    }
  }
#endif

  // Performance Benchmark
  auto start = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < iters; iter++) {
    if (agg_iters>1) NCCLCHECK(ncclGroupStart());
    for (int aiter = 0; aiter < agg_iters; aiter++) {
      TESTCHECK(startColl(args, type, op, root, in_place, iter*agg_iters+aiter));
    }
    if (agg_iters>1) NCCLCHECK(ncclGroupEnd());
  }

#if CUDART_VERSION >= 11030
  if (cudaGraphLaunches >= 1) {
    // End cuda graph capture
    for (int i=0; i<args->nGpus; i++) {
      CUDACHECK(cudaStreamEndCapture(args->streams[i], graphs+i));
    }
    // Instantiate cuda graph
    for (int i=0; i<args->nGpus; i++) {
      CUDACHECK(cudaGraphInstantiate(graphExec+i, graphs[i], NULL, NULL, 0));
    }
    // Resync CPU, restart timing, launch cuda graph
    Barrier(args);
    start = std::chrono::high_resolution_clock::now();
    for (int l=0; l<cudaGraphLaunches; l++) {
      for (int i=0; i<args->nGpus; i++) {
        CUDACHECK(cudaGraphLaunch(graphExec[i], args->streams[i]));
      }
    }
  }
#endif

  TESTCHECK(completeColl(args));

  auto delta = std::chrono::high_resolution_clock::now() - start;
  double deltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count();
  deltaSec = deltaSec/(iters*agg_iters);
  if (cudaGraphLaunches >= 1) deltaSec = deltaSec/cudaGraphLaunches;
  Allreduce(args, &deltaSec, average);

#if CUDART_VERSION >= 11030
  if (cudaGraphLaunches >= 1) {
    //destroy cuda graph
    for (int i=0; i<args->nGpus; i++) {
      CUDACHECK(cudaGraphExecDestroy(graphExec[i]));
      CUDACHECK(cudaGraphDestroy(graphs[i]));
    }
  }
#endif

  double algBw, busBw;
  args->collTest->getBw(count, wordSize(type), deltaSec, &algBw, &busBw, args->nProcs*args->nThreads*args->nGpus);

  Barrier(args);

  double maxDelta = 0;
  static __thread int rep = 0;
  rep++;
  if (datacheck) {
      // Initialize sendbuffs, recvbuffs and expected
      TESTCHECK(args->collTest->initData(args, type, op, root, rep, in_place));

#if CUDART_VERSION >= 11030
      if (cudaGraphLaunches >= 1) {
        // Begin cuda graph capture for data check
        for (int i=0; i<args->nGpus; i++) {
          CUDACHECK(cudaStreamBeginCapture(args->streams[i], args->nThreads > 1 ? cudaStreamCaptureModeThreadLocal : cudaStreamCaptureModeGlobal));
        }
      }
#endif

      //test validation in single itertion, should ideally be included into the multi-iteration run
      TESTCHECK(startColl(args, type, op, root, in_place, 0));

#if CUDART_VERSION >= 11030
      if (cudaGraphLaunches >= 1) {
        // End cuda graph capture
        for (int i=0; i<args->nGpus; i++) {
          CUDACHECK(cudaStreamEndCapture(args->streams[i], graphs+i));
        }
        // Instantiate cuda graph
        for (int i=0; i<args->nGpus; i++) {
          CUDACHECK(cudaGraphInstantiate(graphExec+i, graphs[i], NULL, NULL, 0));
        }
        // Launch cuda graph
        for (int i=0; i<args->nGpus; i++) {
          CUDACHECK(cudaGraphLaunch(graphExec[i], args->streams[i]));
        }
      }
#endif

      TESTCHECK(completeColl(args));

#if CUDART_VERSION >= 11030
      if (cudaGraphLaunches >= 1) {
        //destroy cuda graph
        for (int i=0; i<args->nGpus; i++) {
          CUDACHECK(cudaGraphExecDestroy(graphExec[i]));
          CUDACHECK(cudaGraphDestroy(graphs[i]));
        }
      }
#endif

      TESTCHECK(CheckData(args, type, op, root, in_place, &maxDelta));

      //aggregate delta from all threads and procs
      Allreduce(args, &maxDelta, 3);
  }

  double timeUsec = deltaSec*1.0E6;
  char timeStr[100];
  if (timeUsec >= 10000.0) {
    sprintf(timeStr, "%7.0f", timeUsec);
  } else if (timeUsec >= 100.0) {
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
    if (datacheck) CUDACHECK(cudaMalloc(expected, recvBytes));
    return testSuccess;
}

testResult_t run(); // Main function

int main(int argc, char* argv[]) {
  // Make sure everyline is flushed so that we see the progress of the test
  setlinebuf(stdout);

  #if NCCL_VERSION_CODE >= NCCL_VERSION(2,4,0)
    ncclGetVersion(&test_ncclVersion);
  #else
    test_ncclVersion = NCCL_VERSION_CODE;
  #endif
  //printf("# NCCL_VERSION_CODE=%d ncclGetVersion=%d\n", NCCL_VERSION_CODE, test_ncclVersion);
  #if NCCL_VERSION_CODE >= NCCL_VERSION(2,0,0)
    test_opnum = 4;
    test_typenum = 9;
    if (NCCL_VERSION_CODE >= NCCL_VERSION(2,10,0) && test_ncclVersion >= NCCL_VERSION(2,10,0)) {
      test_opnum++; // ncclAvg
      #if defined(__CUDA_BF16_TYPES_EXIST__)
        test_typenum++; // bfloat16
      #endif
    }
    if (NCCL_VERSION_CODE >= NCCL_VERSION(2,11,0) && test_ncclVersion >= NCCL_VERSION(2,11,0)) {
      test_opnum++; // PreMulSum
    }
  #endif

  // Parse args
  double parsed;
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
    {"cudagraph", required_argument, 0, 'G'},
    {"average", required_argument, 0, 'a'},
    {"help", no_argument, 0, 'h'},
    {}
  };

  while(1) {
    int c;
    c = getopt_long(argc, argv, "t:g:b:e:i:f:n:m:w:p:c:o:d:r:z:hG:a:", longopts, &longindex);

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
        parsed = parsesize(optarg);
        if (parsed < 0) {
          fprintf(stderr, "invalid size specified for 'minbytes'\n");
          return -1;
        }
        minBytes = (size_t)parsed;
        break;
      case 'e':
        parsed = parsesize(optarg);
        if (parsed < 0) {
          fprintf(stderr, "invalid size specified for 'maxbytes'\n");
          return -1;
        }
        maxBytes = (size_t)parsed;
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
#if NCCL_MAJOR > 2 || (NCCL_MAJOR >= 2 && NCCL_MINOR >= 2)
        agg_iters = (int)strtol(optarg, NULL, 0);
#else
        fprintf(stderr, "Option -m not supported before NCCL 2.2. Ignoring\n");
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
      case 'G':
#if (NCCL_MAJOR > 2 || (NCCL_MAJOR >= 2 && NCCL_MINOR >= 9)) && CUDART_VERSION >= 11030
        cudaGraphLaunches = strtol(optarg, NULL, 0);
#else
        printf("Option -G (CUDA graph) not supported before NCCL 2.9 + CUDA 11.3. Ignoring\n");
#endif
        break;
      case 'a':
        average = (int)strtol(optarg, NULL, 0);
        break;
      case 'h':
      default:
        if (c != 'h') printf("invalid option '%c'\n", c);
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
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,11,0)
            "[-o,--op <sum/prod/min/max/avg/mulsum/all>] \n\t"
#elif NCCL_VERSION_CODE >= NCCL_VERSION(2,10,0)
            "[-o,--op <sum/prod/min/max/avg/all>] \n\t"
#else
            "[-o,--op <sum/prod/min/max/all>] \n\t"
#endif
            "[-d,--datatype <nccltype/all>] \n\t"
            "[-r,--root <root>] \n\t"
            "[-z,--blocking <0/1>] \n\t"
            "[-G,--cudagraph <num graph launches>] \n\t"
            "[-a,--average <0/1/2/3> report average iteration time <0=RANK0/1=AVG/2=MIN/3=MAX>] \n\t"
            "[-h,--help]\n",
	    basename(argv[0]));
	return 0;
    }
  }
  if (minBytes > maxBytes) {
    fprintf(stderr, "invalid sizes for 'minbytes' and 'maxbytes': %llu > %llu\n",
           (unsigned long long)minBytes,
           (unsigned long long)maxBytes);
    return -1;
  }
#ifdef MPI_SUPPORT
  MPI_Init(&argc, &argv);
#endif
  TESTCHECK(run());
  return 0;
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
  size_t maxMem = ~0;
  for (int i=0; i<nThreads*nGpus; i++) {
    int cudaDev = localRank*nThreads*nGpus+i;
    int rank = proc*nThreads*nGpus+i;
    cudaDeviceProp prop;
    CUDACHECK(cudaGetDeviceProperties(&prop, cudaDev));
    len += snprintf(line+len, MAX_LINE-len, "#   Rank %2d Pid %6d on %10s device %2d [0x%02x] %s\n",
                    rank, getpid(), hostname, cudaDev, prop.pciBusID, prop.name);
    maxMem = std::min(maxMem, prop.totalGlobalMem);
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
  MPI_Allreduce(MPI_IN_PLACE, &maxMem, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
#else
  PRINT("%s", line);
#endif

  // We need sendbuff, recvbuff, expected (when datacheck enabled), plus 1G for the rest.
  size_t memMaxBytes = (maxMem - (1<<30)) / (datacheck ? 3 : 2);
  if (maxBytes > memMaxBytes) {
    maxBytes = memMaxBytes;
    if (proc == 0) printf("#\n# Reducing maxBytes to %ld due to memory limitation\n", maxBytes);
  }

  ncclUniqueId ncclId;
  if (proc == 0) {
    NCCLCHECK(ncclGetUniqueId(&ncclId));
  }
#ifdef MPI_SUPPORT
  MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  cudaStream_t streams[nGpus*nThreads];
  void* sendbuffs[nGpus*nThreads];
  void* recvbuffs[nGpus*nThreads];
  void* expected[nGpus*nThreads];
  size_t sendBytes, recvBytes;

  ncclTestEngine.getBuffSize(&sendBytes, &recvBytes, (size_t)maxBytes, (size_t)nProcs*nGpus*nThreads);

  for (int i=0; i<nGpus*nThreads; i++) {
    CUDACHECK(cudaSetDevice(localRank*nThreads*nGpus+i));
    TESTCHECK(AllocateBuffs(sendbuffs+i, sendBytes, recvbuffs+i, recvBytes, expected+i, (size_t)maxBytes, nProcs*nThreads*nGpus));
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
  CUDACHECK(cudaHostAlloc(&delta, sizeof(double)*nThreads*NUM_BLOCKS, cudaHostAllocPortable | cudaHostAllocMapped));
  int bw_count[nThreads];
  for (int t=0; t<nThreads; t++) {
    bw[t] = 0.0;
    errors[t] = bw_count[t] = 0;
  }

  PRINT("#\n");
  print_header();

  int* sync = (int*)calloc(2, sizeof(int));
  int* barrier = (int*)calloc(2, sizeof(int));
  double* reduce = (double*)calloc(2, sizeof(double));

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
    threads[t].args.reduce = (volatile double*)reduce;
    threads[t].args.sync = (volatile int*)sync;
    threads[t].args.sync_idx = 0;
    threads[t].args.deltaHost = (delta + t*NUM_BLOCKS);
    threads[t].args.errors=errors+t;
    threads[t].args.bw=bw+t;
    threads[t].args.bw_count=bw_count+t;

    threads[t].args.reportErrors = 1;

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
    if (sendbuffs[i]) CUDACHECK(cudaFree((char*)sendbuffs[i]));
    if (recvbuffs[i]) CUDACHECK(cudaFree((char*)recvbuffs[i]));
    if (datacheck) CUDACHECK(cudaFree(expected[i]));
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
