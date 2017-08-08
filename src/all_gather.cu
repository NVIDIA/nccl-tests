/*************************************************************************
 * Copyright (c) 2016-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENCE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"


void print_header() {
  PRINT("# %10s  %12s  %6s  %6s        out-of-place                    in-place\n", "", "", "", "");
  PRINT("# %10s  %12s  %6s  %7s  %5s  %5s  %7s  %7s  %5s  %5s  %7s\n", "bytes", "N", "type", 
      "time", "algbw", "busbw", "res", "time", "algbw", "busbw", "res");
}

void print_line_header (size_t size, size_t count, const char *typeName, const char *opName, int root) {
  PRINT("%12li  %12li  %6s", size, count, typeName);
}

void getCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t *procSharedCount, int *sameExpected, size_t count, int nranks) {
    *sendcount = count/nranks;
    *recvcount = (count/nranks)*nranks;
    *sameExpected = 1;
    *procSharedCount = 0;
    *sendInplaceOffset = count/nranks;
    *recvInplaceOffset = 0;
    *paramcount = *sendcount;
}

void InitRecvResult(struct threadArgs_t* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, int is_first) {
  size_t nBytes = args->nbytes;
  size_t count = nBytes / wordSize(type);
  int proc = args->proc;
  int nThreads = args->nThreads;
  int t = args->thread;
  int nGpus = args->nGpus;

  while (args->sync[args->sync_idx] != t) pthread_yield();

  for (int i=0; i<nGpus; i++) {
    int device;
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    NCCLCHECK(ncclCommCuDevice(args->comms[i], &device));
    CUDACHECK(cudaSetDevice(device));

    void* data = in_place ? (void *)((uintptr_t)args->recvbuffs[i] + args->sendInplaceOffset*rank) : args->sendbuffs[i];

    CUDACHECK(cudaMemcpy((void *)((uintptr_t)args->expectedHost[0] + ((proc*nThreads + t)*nGpus + i)*nBytes), 
                data, 
                nBytes, cudaMemcpyDeviceToHost));

    if (in_place == 0) {
      CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    }
    CUDACHECK(cudaDeviceSynchronize());
  }

  args->sync[args->sync_idx] = t + 1;

  if (t+1 == nThreads) {
#ifdef MPI_SUPPORT
    // Last thread does the MPI allgather
    MPI_Allgather(MPI_IN_PLACE, nBytes*nThreads*nGpus, MPI_BYTE, 
        args->expectedHost[0], 
        nBytes*nThreads*nGpus, MPI_BYTE, MPI_COMM_WORLD);
#endif
    args->sync[args->sync_idx] = 0;
  } else {
    while (args->sync[args->sync_idx]) pthread_yield();
  }

  args->sync_idx=!args->sync_idx;
}

void GetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize * (nranks - 1)) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = 1;
  *busBw = baseBw * factor;
}

void RunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  NCCLCHECK(ncclAllGather(sendbuff, recvbuff, count, type, comm, stream));
}

void RunTest(struct threadArgs_t* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  ncclDataType_t *run_types;
  const char **run_typenames;
  int type_count;

  if ((int)type != -1) { 
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else { 
    type_count = ncclNumTypes;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  for (int i=0; i<type_count; i++) { 
     TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, NULL, 0, 1);
  }   
}
