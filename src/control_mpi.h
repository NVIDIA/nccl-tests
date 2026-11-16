/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef NCCL_TESTS_CONTROL_MPI_H_
#define NCCL_TESTS_CONTROL_MPI_H_

class TestControl {
 public:
  int totalProcs = 1, proc = 0, ncclProcs = 1, ncclProc = 0;
  int localRank = 0, color = 0;

  bool needsComms() const { return false; }
  bool hasProcessTiming() const { return true; }

  testResult_t init(int* argc, char*** argv) {
    MPI_Init(argc, argv);
    return testSuccess;
  }

  testResult_t discoverRanks(const char* hostname, int nThreads, int nGpus, int parallelInit) {
    MPI_Comm_size(MPI_COMM_WORLD, &totalProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc);
    std::vector<uint64_t> hostHashs(totalProcs);
    hostHashs[proc] = getHostHash(hostname);
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs.data(), sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);
    for (int p=0; p<totalProcs; p++) {
      if (p == proc) break;
      if (hostHashs[p] == hostHashs[proc]) localRank++;
    }

    color = getSplitColor(proc);

    MPI_Comm_split(MPI_COMM_WORLD, color, proc, &mpiComm);
    MPI_Comm_size(mpiComm, &ncclProcs);
    MPI_Comm_rank(mpiComm, &ncclProc);
    return testSuccess;
  }

  testResult_t getUniqueId(ncclUniqueId* id) {
    if (ncclProc == 0) NCCLCHECK(ncclGetUniqueId(id));
    MPI_Bcast(id, sizeof(*id), MPI_BYTE, 0, mpiComm);
    MPI_Barrier(MPI_COMM_WORLD); // Ensure Bcast is complete for HCOLL
    return testSuccess;
  }

  testResult_t initComms(ncclComm_t*, cudaStream_t*, int*, int) { return testSuccess; }
  testResult_t releaseComms() { return testSuccess; }

  testResult_t barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
    return testSuccess;
  }

  void gather(double* send, int count, double* recv) {
    if (totalProcs > 1)
      MPI_Gather(send, count, MPI_DOUBLE, recv, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  testResult_t finalize() {
    MPI_Comm_free(&mpiComm);
    MPI_Finalize();
    return testSuccess;
  }

 private:
  MPI_Comm mpiComm;

  template<typename T>
  testResult_t allReduceImpl(T* value, int average) {
    MPI_Datatype type = std::is_same<T, int>::value ? MPI_INT :
                        std::is_same<T, double>::value ? MPI_DOUBLE :
                        std::is_same<T, long long>::value ? MPI_LONG_LONG : MPI_INT64_T;
    MPI_Op op = average == 2 ? MPI_MIN : average == 3 ? MPI_MAX : MPI_SUM;
    MPI_Allreduce(MPI_IN_PLACE, value, 1, type, op, MPI_COMM_WORLD);
    return testSuccess;
  }

 public:
  template<typename T>
  testResult_t allReduce(T* value, int average) {
    static_assert(std::is_same<T, int>::value ||
                  std::is_same<T, double>::value ||
                  (std::is_integral<T>::value && std::is_signed<T>::value && sizeof(T) == 8),
                  "Control reductions support int, signed 64-bit integers, and double");
    if (average == 0) return testSuccess;
    return allReduceImpl(value, average);
  }
};

#endif
