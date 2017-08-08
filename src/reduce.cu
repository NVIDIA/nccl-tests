/*************************************************************************
 * Copyright (c) 2016-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENCE.txt for license information
 ************************************************************************/

#include <assert.h>
#include "cuda_runtime.h"
#include "common.h"

void print_header() {
  PRINT("# %10s  %12s  %6s  %6s        out-of-place                    in-place\n", "", "", "", "");
  PRINT("# %10s  %12s  %6s  %6s  %6s %7s  %5s  %5s  %7s  %7s  %5s  %5s  %7s\n", "bytes", "N", "type", "op", "root",
      "time", "algbw", "busbw", "res", "time", "algbw", "busbw", "res");
}

void print_line_header (size_t size, size_t count, const char *typeName, const char *opName, int root) {
  PRINT("%12li  %12li  %6s  %6s  %6i", size, count, typeName, opName, root);
}

void getCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t *procSharedCount, int *sameExpected, size_t count, int nranks) {
    *sendcount = count;
    *recvcount = count;
    *sameExpected = 0;
    *procSharedCount = count;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
    *paramcount = *sendcount;
 }

void InitRecvResult(struct threadArgs_t* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, int is_first) {
  size_t count = args->expectedBytes / wordSize(type);
  int root_gpu = root%args->nGpus;

  assert(args->expectedBytes == args->nbytes);

  while (args->sync[args->sync_idx] != args->thread) pthread_yield();

  for (int i=0; i<args->nGpus; i++) {
    int device;
    NCCLCHECK(ncclCommCuDevice(args->comms[i], &device));
    CUDACHECK(cudaSetDevice(device));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];

    if (is_first && i == 0) {
      CUDACHECK(cudaMemcpy(args->procSharedHost, data, count*wordSize(type), cudaMemcpyDeviceToHost));
    } else {
      Accumulate(args->procShared, data, count, type, op);
    }

    if (in_place == 0) {
      CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    }
    CUDACHECK(cudaDeviceSynchronize());
  }

  args->sync[args->sync_idx] = args->thread + 1;

  if (args->thread+1 == args->nThreads) {
#ifdef MPI_SUPPORT
    int root_proc = root/(args->nThreads*args->nGpus);
    if (args->expectedBytes) {
      // Last thread does the MPI reduction
      if (root_proc == args->proc) { 
        void* temp, *tempHost = malloc(args->expectedBytes);
        CUDACHECK(cudaHostRegister(tempHost, args->expectedBytes, 0));
        CUDACHECK(cudaHostGetDevicePointer(&temp, tempHost, 0));

        for (int i=0; i<args->nProcs; i++) {
          if (i == args->proc) continue;
          MPI_Recv(tempHost, args->expectedBytes, MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          Accumulate(args->procShared, temp, count, type, op);
          CUDACHECK(cudaDeviceSynchronize());
        }

        CUDACHECK(cudaHostUnregister(tempHost));
        free(tempHost);
      } else {
        MPI_Send(args->procSharedHost, args->expectedBytes, MPI_BYTE, root_proc, 0, MPI_COMM_WORLD);
      }
    }
#endif
    args->sync[args->sync_idx] = 0;
  } else {
    while (args->sync[args->sync_idx]) pthread_yield();
  }

  //if root fill expected bytes with reduced data
  // else if in_place, leave fill it with original data, else set to zero
  for (int i=0; i<args->nGpus; i++) {
      int rank = (args->proc*args->nThreads + args->thread)*args->nGpus + i;
      if (rank == root) { 
          memcpy(args->expectedHost[root_gpu], args->procSharedHost, args->expectedBytes); 
      } else { 
         if (in_place == 1) {
              CUDACHECK(cudaMemcpy(args->expectedHost[i], args->recvbuffs[i], args->expectedBytes, cudaMemcpyDeviceToHost));
          } else {
              memset(args->expectedHost[i], 0, args->expectedBytes); 
          }
      } 
  }

  args->sync_idx = !args->sync_idx;
}

void GetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;
  *algBw = baseBw;
  *busBw = baseBw;
}

void RunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  NCCLCHECK(ncclReduce(sendbuff, recvbuff, count, type, op, root, comm, stream));
}


void RunTest(struct threadArgs_t* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  ncclDataType_t *run_types;
  ncclRedOp_t *run_ops;
  const char **run_typenames, **run_opnames;
  int type_count, op_count;
  int begin_root, end_root;

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

  if (root != -1) { 
     begin_root = end_root = root;
  } else { 
     begin_root = 0;
     end_root = args->nProcs*args->nThreads*args->nGpus-1;
  }

  for (int i=0; i<type_count; i++) { 
      for (int j=0; j<op_count; j++) { 
         for (int k=begin_root; k<=end_root; k++) { 
             TimeTest(args, run_types[i], run_typenames[i], run_ops[j], run_opnames[j], k, 1);
         }
      }
  }   
}
