/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef NCCL_TESTS_CONTROL_ENV_H_
#define NCCL_TESTS_CONTROL_ENV_H_

#include <errno.h>
#include <limits>

testResult_t testStreamSynchronize(int ngpus, cudaStream_t* streams, ncclComm_t* comms);

static bool parseEnvInt(const char *name, int *num) {
  const char *env = getenv(name);
  if (env == nullptr) return false;
  char *end;
  errno = 0;
  long value = strtol(env, &end, 10);
  if (errno == ERANGE || end == env || *end != '\0' || value < 0 || value > std::numeric_limits<int>::max()) {
    fprintf(stderr, "Invalid integer value for %s: %s\n", name, env);
    return false;
  }
  *num = (int)value;
  return true;
}

struct envRankNames {
  const char *rank;
  const char *nranks;
};

static int getStandardEnvRankAndSize(int *rank, int *nranks, const char **rankName, const char **nranksName) {
  const envRankNames names[] = {
    {"SLURM_PROCID", "SLURM_NTASKS"},
    {"SLURM_PROCID", "SLURM_NPROCS"},
    {"PMI_RANK", "PMI_SIZE"},
    {"PMIX_RANK", "PMIX_SIZE"},
    {"OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"},
    {"MV2_COMM_WORLD_RANK", "MV2_COMM_WORLD_SIZE"},
    {"RANK", "WORLD_SIZE"}
  };
  bool foundPartial = false;
  const char *partialRankName = nullptr;
  const char *partialNranksName = nullptr;

  for (size_t i = 0; i < sizeof(names)/sizeof(names[0]); i++) {
    const bool haveRank = getenv(names[i].rank) != nullptr;
    const bool haveNranks = getenv(names[i].nranks) != nullptr;
    if (haveRank && haveNranks) {
      if (!parseEnvInt(names[i].rank, rank) ||
          !parseEnvInt(names[i].nranks, nranks)) {
        return -1;
      }
      if (rankName) *rankName = names[i].rank;
      if (nranksName) *nranksName = names[i].nranks;
      return 1;
    }
    if (haveRank || haveNranks) {
      foundPartial = true;
      partialRankName = names[i].rank;
      partialNranksName = names[i].nranks;
    }
  }

  if (foundPartial) {
    fprintf(stderr,
        "Found incomplete launcher rank environment. Set both %s and %s, "
        "or set NCCL_TESTS_COMM_RANK and NCCL_TESTS_COMM_NRANKS.\n",
        partialRankName, partialNranksName);
    return -1;
  }
  return 0;
}

static int getEnvLocalRank(int *localRank) {
  const char *names[] = {
    "NCCL_TESTS_COMM_LOCAL_RANK",
    "LOCAL_RANK",
    "SLURM_LOCALID",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "MV2_COMM_WORLD_LOCAL_RANK"
  };
  for (size_t i = 0; i < sizeof(names)/sizeof(names[0]); i++) {
    if (getenv(names[i])) return parseEnvInt(names[i], localRank) ? 1 : -1;
  }
  return 0;
}

static testResult_t setFallbackLocalRank(int proc, int *localRank, int nThreads, int nGpus) {
  int availableDevices = 0;
  CUDACHECK(cudaGetDeviceCount(&availableDevices));
  const int gpusPerProc = std::max(1, nThreads*nGpus);
  const int localRankSlots = std::max(1, availableDevices/gpusPerProc);
  *localRank = proc % localRankSlots;
  if (proc == 0) {
    fprintf(stderr,
        "No local rank environment found; using global rank modulo %d local GPU slot(s) for device assignment.\n",
        localRankSlots);
  }
  return testSuccess;
}

class TestControl {
 public:
  int totalProcs = 1, proc = 0, ncclProcs = 1, ncclProc = 0;
  int localRank = 0, color = 0;

  bool needsComms() const { return totalProcs > 1; }
  bool hasProcessTiming() const { return false; }

  testResult_t init(int*, char***) { return testSuccess; }

  testResult_t discoverRanks(const char* hostname, int nThreads, int nGpus, int parallelInit) {
    const bool haveCommId = getenv("NCCL_COMM_ID") != nullptr;
    bool haveEnvRank = getenv("NCCL_TESTS_COMM_RANK") != nullptr;
    bool haveEnvNranks = getenv("NCCL_TESTS_COMM_NRANKS") != nullptr;
    if (haveEnvRank != haveEnvNranks) {
      fprintf(stderr, "NCCL_TESTS_COMM_RANK and NCCL_TESTS_COMM_NRANKS must be set together.\n");
      return testInvalidUsage;
    }
    if (haveEnvRank) {
      if (!parseEnvInt("NCCL_TESTS_COMM_RANK", &proc) ||
          !parseEnvInt("NCCL_TESTS_COMM_NRANKS", &totalProcs)) {
        return testInvalidUsage;
      }
      if (totalProcs < 1 || proc < 0 || proc >= totalProcs) {
        fprintf(stderr, "Invalid NCCL_TESTS_COMM_RANK/NCCL_TESTS_COMM_NRANKS: rank %d nranks %d\n", proc, totalProcs);
        return testInvalidUsage;
      }
      if (totalProcs > 1 && !haveCommId) {
        fprintf(stderr, "NCCL_COMM_ID must be set for non-MPI multi-process launches.\n");
        return testInvalidUsage;
      }
    } else if (haveCommId) {
      const char *rankName = nullptr;
      const char *nranksName = nullptr;
      int envResult = getStandardEnvRankAndSize(&proc, &totalProcs, &rankName, &nranksName);
      if (envResult < 0) return testInvalidUsage;
      if (envResult > 0) {
        if (totalProcs < 1 || proc < 0 || proc >= totalProcs) {
          fprintf(stderr, "Invalid %s/%s: rank %d nranks %d\n", rankName, nranksName, proc, totalProcs);
          return testInvalidUsage;
        }
      }
    }

    const int localRankResult = getEnvLocalRank(&localRank);
    if (localRankResult < 0) return testInvalidUsage;
    if (localRankResult == 0 && needsComms() && getenv("NCCL_TESTS_DEVICE") == nullptr) {
      TESTCHECK(setFallbackLocalRank(proc, &localRank, nThreads, nGpus));
    }

    if ((getenv("NCCL_TESTS_SPLIT") || getenv("NCCL_TESTS_SPLIT_MASK")) && totalProcs > 1) {
      fprintf(stderr, "NCCL_TESTS_SPLIT is not supported for non-MPI multi-process launches.\n");
      return testInvalidUsage;
    }
    if (needsComms() && parallelInit) {
      fprintf(stderr, "parallel_init is not supported for non-MPI multi-process launches.\n");
      return testInvalidUsage;
    }
    color = getSplitColor(proc);
    ncclProcs = totalProcs;
    ncclProc = proc;
    return testSuccess;
  }

  testResult_t getUniqueId(ncclUniqueId* id) {
    NCCLCHECK(ncclGetUniqueId(id));
    return testSuccess;
  }

  testResult_t initComms(ncclComm_t* controlComms, cudaStream_t* controlStreams, int* controlGpus, int count) {
    if (!needsComms()) return testSuccess;
    comms = controlComms;
    streams = controlStreams;
    gpus = controlGpus;
    nranks = count;
    CUDACHECK(cudaHostAlloc(&hostBuffer, nranks*sizeof(double), cudaHostAllocPortable));
    deviceBuffers.resize(nranks);
    for (int i=0; i<nranks; i++) {
      CUDACHECK(cudaSetDevice(gpus[i]));
      CUDACHECK(cudaMalloc(&deviceBuffers[i], 2*sizeof(double)));
      CUDACHECK(cudaMemsetAsync(deviceBuffers[i], 0, 2*sizeof(double), streams[i]));
    }
    return testSuccess;
  }

  testResult_t releaseComms() {
    if (!needsComms()) return testSuccess;
    for (int i=0; i<nranks; i++) {
      CUDACHECK(cudaSetDevice(gpus[i]));
      CUDACHECK(cudaFree(deviceBuffers[i]));
    }
    CUDACHECK(cudaFreeHost(hostBuffer));
    return testSuccess;
  }

  testResult_t barrier() {
    int dummy = 0;
    return allReduceImpl(&dummy, /*sum*/4, /*copy*/false);
  }

  void gather(double* send, int count, double* recv) {
    if (recv) memcpy(recv, send, count*sizeof(double));
  }

  testResult_t finalize() { return testSuccess; }

 private:
  ncclComm_t* comms = nullptr;
  cudaStream_t* streams = nullptr;
  int* gpus = nullptr;
  void* hostBuffer = nullptr;
  std::vector<void*> deviceBuffers;
  int nranks = 0;

  template<typename T>
  testResult_t allReduceImpl(T* value, int average, bool copy = true) {
    if (!needsComms()) return testSuccess;
    ncclDataType_t type = std::is_same<T, int>::value ? ncclInt :
                          std::is_same<T, double>::value ? ncclDouble : ncclInt64;
    ncclRedOp_t op = average == 2 ? ncclMin : average == 3 ? ncclMax : ncclSum;
    T identity = average == 2 ? std::numeric_limits<T>::max() :
                 average == 3 ? std::numeric_limits<T>::lowest() : (T)0;
    T* buffer = (T*)hostBuffer;
    if (copy) {
      buffer[0] = *value;
      for (int i=1; i<nranks; i++) buffer[i] = identity;
      for (int i=0; i<nranks; i++) {
        CUDACHECK(cudaSetDevice(gpus[i]));
        CUDACHECK(cudaMemcpyAsync(deviceBuffers[i], buffer+i, sizeof(T),
                                  cudaMemcpyHostToDevice, streams[i]));
      }
    }

    NCCLCHECK(ncclGroupStart());
    ncclResult_t result = ncclSuccess;
    cudaError_t cudaResult = cudaSuccess;
    for (int i=0; i<nranks; i++) {
      cudaResult = cudaSetDevice(gpus[i]);
      if (cudaResult != cudaSuccess) break;
      // The second slot stays zero across barriers, independent of real reductions.
      void* deviceBuffer = (char*)deviceBuffers[i] + (copy ? 0 : sizeof(double));
      result = ncclAllReduce(deviceBuffer, deviceBuffer, 1,
                            type, op, comms[i], streams[i]);
      if (result != ncclSuccess) break;
    }
    ncclResult_t groupResult = ncclGroupEnd();
    CUDACHECK(cudaResult);
    NCCLCHECK(result);
    NCCLCHECK(groupResult);
    if (copy) {
      CUDACHECK(cudaSetDevice(gpus[0]));
      CUDACHECK(cudaMemcpyAsync(buffer, deviceBuffers[0], sizeof(T),
                                cudaMemcpyDeviceToHost, streams[0]));
    }
    for (int i=0; i<nranks; i++) {
      CUDACHECK(cudaSetDevice(gpus[i]));
      TESTCHECK(testStreamSynchronize(1, streams+i, comms+i));
    }
    if (copy) *value = buffer[0];
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
