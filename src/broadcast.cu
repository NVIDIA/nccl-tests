/*************************************************************************
 * Copyright (c) 2016-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENCE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"
#include <assert.h>

void print_header() {
  PRINT("# %10s  %12s  %6s  %6s        out-of-place\n", "", "", "", "");
  PRINT("# %10s  %12s  %6s  %6s  %7s  %5s  %5s  %7s\n", "bytes", "N", "type", "root", 
      "time", "algbw", "busbw", "res");
}

void print_line_header (size_t size, size_t count, const char *typeName, const char *opName, int root) {
  PRINT("%12li  %12li  %6s  %6i", size, count, typeName, root);
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
  int root_proc = root/(args->nThreads*args->nGpus);
  int root_thread = (root/args->nGpus)%(args->nThreads);
  int root_gpu = root%args->nGpus;

  assert(args->expectedBytes == args->nbytes);

  if (root_thread == args->thread) {
      if (root_proc == args->proc) {  
         CUDACHECK(cudaMemcpy(args->procSharedHost,
                    args->sendbuffs[root_gpu],
                    args->nbytes, cudaMemcpyDeviceToHost));
      }
#ifdef MPI_SUPPORT 
      MPI_Bcast(args->procSharedHost, args->nbytes, MPI_BYTE, root_proc, MPI_COMM_WORLD);
#endif

      args->sync[0] = 0;
  }

  Barrier(args);

  for (int i=0; i<args->nGpus; i++) {
     int device;
     NCCLCHECK(ncclCommCuDevice(args->comms[i], &device)); 
     CUDACHECK(cudaSetDevice(device));

     //set expected buf to zero at root, copy over source data at others
     if ((root_proc == args->proc) 
         && (root_thread == args->thread) 
         && (root_gpu == i)) { 
         memset(args->expectedHost[i], 0, args->nbytes); 
     } else { 
         memcpy(args->expectedHost[i], args->procSharedHost, args->nbytes);
     }

     //reset recvbufs to zero
     CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->nbytes));
     CUDACHECK(cudaDeviceSynchronize());
  }

  Barrier(args);
}

void GetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = 1;
  *busBw = baseBw * factor;
}

void RunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  int rank; 
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  if (rank == root) { 
      NCCLCHECK(ncclBcast(sendbuff, count, type, root, comm, stream));
  } else { 
      NCCLCHECK(ncclBcast(recvbuff, count, type, root, comm, stream));
  } 
}

void RunTest(struct threadArgs_t* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  ncclDataType_t *run_types;
  const char **run_typenames;
  int type_count;
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

  if (root != -1) { 
     begin_root = end_root = root;
  } else { 
     begin_root = 0;
     end_root = args->nProcs*args->nThreads*args->nGpus-1;
  }

  for (int i=0; i<type_count; i++) { 
       for (int j=begin_root; j<=end_root; j++) {
          TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, NULL, j, 0);
       }
  }   
}
