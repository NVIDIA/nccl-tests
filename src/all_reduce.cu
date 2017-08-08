/*************************************************************************
 * Copyright (c) 2016-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENCE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"

void print_header() {
  PRINT("# %10s  %12s  %6s  %6s        out-of-place                    in-place\n", "", "", "", "");
  PRINT("# %10s  %12s  %6s  %6s  %7s  %5s  %5s  %7s  %7s  %5s  %5s  %7s\n", "bytes", "N", "type", "op",
      "time", "algbw", "busbw", "res", "time", "algbw", "busbw", "res");
}

void print_line_header (size_t size, size_t count, const char *typeName, const char *opName, int root) {
  PRINT("%12li  %12li  %6s  %6s", size, count, typeName, opName);
}

void getCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t *procSharedCount, int *sameExpected, size_t count, int nranks) {
    *sendcount = count;
    *recvcount = count;
    *sameExpected = 1;
    *procSharedCount = 0;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
    *paramcount = *sendcount;
 }

void InitRecvResult(struct threadArgs_t* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, int is_first) {
  size_t count = args->nbytes / wordSize(type);

  while (args->sync[args->sync_idx] != args->thread) pthread_yield();

  for (int i=0; i<args->nGpus; i++) {
    int device;
    NCCLCHECK(ncclCommCuDevice(args->comms[i], &device));
    CUDACHECK(cudaSetDevice(device));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];

    if (is_first && i == 0) {
      CUDACHECK(cudaMemcpy(args->expected[0], data, count*wordSize(type), cudaMemcpyDeviceToHost));
    } else {
      Accumulate(args->expected[0], data, count, type, op);
    }

    if (in_place == 0) {
      CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->nbytes));
    }
    CUDACHECK(cudaDeviceSynchronize());
  }

  args->sync[args->sync_idx] = args->thread + 1;

  if (args->thread+1 == args->nThreads) {
#ifdef MPI_SUPPORT
    // Last thread does the MPI reduction
    if (args->nbytes > 0) {
      void* remote, *remoteHost = malloc(args->nbytes);
      void* myInitialData = malloc(args->nbytes);
      memcpy(myInitialData, args->expectedHost[0], args->nbytes);
      CUDACHECK(cudaHostRegister(remoteHost, args->nbytes, cudaHostRegisterPortable | cudaHostRegisterMapped));
      CUDACHECK(cudaHostGetDevicePointer(&remote, remoteHost, 0));
      for (int i=0; i<args->nProcs; i++) {
        if (i == args->proc) {
          MPI_Bcast(myInitialData, args->nbytes, MPI_BYTE, i, MPI_COMM_WORLD);
          free(myInitialData);
        } else {
          MPI_Bcast(remoteHost, args->nbytes, MPI_BYTE, i, MPI_COMM_WORLD);
          Accumulate(args->expected[0], remote, count, type, op);
          cudaDeviceSynchronize();
        }
      }
      CUDACHECK(cudaHostUnregister(remoteHost));
      free(remoteHost);
    }
#endif
    args->sync[args->sync_idx] = 0;
  } else {
    while (args->sync[args->sync_idx]) pthread_yield();
  }

  args->sync_idx = !args->sync_idx;
}

void GetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(2*(nranks - 1)))/((double)nranks);
  *busBw = baseBw * factor;
}

void RunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, type, op, comm, stream));
}


void RunTest(struct threadArgs_t* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  ncclDataType_t *run_types;
  ncclRedOp_t *run_ops;
  const char **run_typenames, **run_opnames;
  int type_count, op_count;

  if ((int)type != -1) { 
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else { 
    type_count = ncclNumTypes;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  if ((int)op != -1) { 
    op_count = 1;
    run_ops = &op;
    run_opnames = &opName;
  } else { 
    op_count = ncclNumOps;
    run_ops = test_ops;
    run_opnames = test_opnames;
  }

  for (int i=0; i<type_count; i++) { 
      for (int j=0; j<op_count; j++) { 
          TimeTest(args, run_types[i], run_typenames[i], run_ops[j], run_opnames[j], 0, 1);
      }
  }   
}
