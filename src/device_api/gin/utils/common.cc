#include "common.h"
#include "gin_context.h"


/*
 * A binary can cover several operations (the put benchmarks select between put,
 * put+signal and put+counter via --gin_op), so name the resolved configuration
 * before the table.
 */
static void printConfiguration(const ginArgs_t* args, const ginBenchmark_t* bench) {
  printf("# %s: rsm=%s", bench->name ? bench->name : "gin benchmark", ncclTestGinRsmName(args->ginRsm));
  if (args->ginStrongSignal) printf(", strong_signal");
  if (args->ginSkipCreditCheck) printf(", skip_credit_check");
  if (args->ginAggregateRequests) printf(", aggregate_requests");
  if (args->ginBidirectional) printf(", bidirectional");
  if (bench->type == GIN_BENCHMARK_TYPE_THROUGHPUT) printf(", ctas=%d, threads=%d", args->numCtas, args->numThreads);
  printf("\n");
}

static void printHeader(const ginBenchmark_t* bench) {
  if (bench->type == GIN_BENCHMARK_TYPE_THROUGHPUT) {
    printf("%12s  %14s  %18s  %20s\n", "Size(B)", "Num_Messages", "Bandwidth(MiB/s)",
           "Message_Rate(MPPS)");
  } else {
    printf("%12s  %8s  %14s\n", "Size(B)", "Iters", "Latency(us)");
  }
}

static void printThroughputInformation(size_t size, double numMessages, double bandwidthMiBps,
                                       double messageRateMpps) {
  printf("%12zu  %14.0f  %18.3f  %20.3f\n", size, numMessages, bandwidthMiBps, messageRateMpps);
}

static void printLatencyInformation(size_t size, int iters, double latencyUs) {
  printf("%12zu  %8d  %14.3f\n", size, iters, latencyUs);
}

static uint64_t getHostHash(const char* string) {
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++) {
    result = ((result << 5) + result) + string[c];
  }
  return result;
}

static void getHostName(char* hostname, int maxlen) {
  memset(hostname, 0, maxlen);
  gethostname(hostname, maxlen);
  for (int i = 0; i < maxlen && hostname[i] != '\0'; i++) {
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}

static size_t nextSize(const ginArgs_t* args, size_t size) {
  if (args->stepBytes != 0) return size + args->stepBytes;
  size_t factor = args->stepFactor;
  if (factor <= 1) factor = 2;
  return size * factor;
}

static double throughputBytesPerMessage(const ginBenchmark_t* bench, size_t size) {
  switch (bench->payloadMode) {
  case GIN_THROUGHPUT_PAYLOAD_NONE:
    return 0.0;
  case GIN_THROUGHPUT_PAYLOAD_FIXED:
    return (double)bench->payloadFixedBytes;
  case GIN_THROUGHPUT_PAYLOAD_SIZE:
  default:
    return (double)size;
  }
}

void ncclTestGinPerfRun(int argc, char** argv, const ginArgs_t* args, const ginBenchmark_t* bench) {

  setlinebuf(stdout);
  int rank = 0, nRanks = 0;
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK_FATAL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK_FATAL(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  if (nRanks != 2) {
    if (rank == 0) fprintf(stderr, "Error: GIN perf benchmarks require exactly 2 MPI ranks (got %d)\n", nRanks);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  uint64_t* hosts = new uint64_t[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hosts[rank] = getHostHash(hostname);
  MPICHECK_FATAL(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                               hosts, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  int dev = 0;
  for (int r = 0; r < rank; r++) {
    if (hosts[r] == hosts[rank]) dev++;
  }
  delete[] hosts;

  CUDACHECK_FATAL(cudaSetDevice(dev));

  ncclUniqueId id;
  if (rank == 0) NCCLCHECK_FATAL(ncclGetUniqueId(&id));
  MPICHECK_FATAL(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  cudaStream_t stream;
  CUDACHECK_FATAL(cudaStreamCreate(&stream));

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 1;
  ncclComm_t comm;
  NCCLCHECK_FATAL(ncclCommInitRankConfig(&comm, nRanks, id, rank, &config));

  ginContext_t ctx;
  ncclTestGinDevCommCreate(comm, args, &ctx);

  const bool bidir = args->ginBidirectional;
  const bool participates = (bench->type == GIN_BENCHMARK_TYPE_PING_PONG || bidir) || rank == 0;
  const bool measures = (rank == 0) || bidir;

  cudaEvent_t start, stop;
  if (measures) {
    CUDACHECK_FATAL(cudaEventCreate(&start));
    CUDACHECK_FATAL(cudaEventCreate(&stop));
  }

  if (rank == 0) {
    printConfiguration(args, bench);
    printHeader(bench);
  }

  for (size_t size = args->minBytes; size <= args->maxBytes;) {
    size_t numElems = size / sizeof(int);

    //-----Warmup-----
    if (args->warmupIters > 0 && participates) {
      bench->run(ctx.dcomm, ctx.hBuf, stream, args, numElems, args->warmupIters);
      CUDACHECK_FATAL(cudaStreamSynchronize(stream));
    }

    //-----Benchmark-----
    MPICHECK_FATAL(MPI_Barrier(MPI_COMM_WORLD));

    if (measures) CUDACHECK_FATAL(cudaEventRecord(start, stream));
    if (participates) bench->run(ctx.dcomm, ctx.hBuf, stream, args, numElems, args->iters);
    if (measures) CUDACHECK_FATAL(cudaEventRecord(stop, stream));
    if (participates) CUDACHECK_FATAL(cudaStreamSynchronize(stream));

    if (bench->type == GIN_BENCHMARK_TYPE_PING || bench->type == GIN_BENCHMARK_TYPE_PING_PONG) {
      // Latency: rank 0 reports (bidirectional is not supported for latency).
      if (rank == 0) {
        float milliseconds = 0.0f;
        CUDACHECK_FATAL(cudaEventElapsedTime(&milliseconds, start, stop));
        double rttUs = (double)milliseconds * 1000.0 / (double)args->iters;
        double latencyUs = rttUs / 2.0;
        printLatencyInformation(size, args->iters, latencyUs);
      }
    } else {
      double numMessages = 0.0, bandwidthMiBps = 0.0, messageRateMpps = 0.0;
      if (measures) {
        float milliseconds = 0.0f;
        CUDACHECK_FATAL(cudaEventElapsedTime(&milliseconds, start, stop));
        double seconds = (double)milliseconds / 1e3;
        numMessages = (double)args->iters * (double)args->numCtas * (double)args->numThreads;
        double bytes = numMessages * throughputBytesPerMessage(bench, size);
        bandwidthMiBps = bytes / (double)(1ULL << 20) / seconds;
        messageRateMpps = numMessages / 1e6 / seconds;
      }

      if (bidir) {
        double localMetrics[3] = {numMessages, bandwidthMiBps, messageRateMpps};
        double summedMetrics[3] = {0.0, 0.0, 0.0};
        MPICHECK_FATAL(MPI_Reduce(localMetrics, summedMetrics, 3, MPI_DOUBLE, MPI_SUM, 0,
                                  MPI_COMM_WORLD));
        numMessages = summedMetrics[0];
        bandwidthMiBps = summedMetrics[1];
        messageRateMpps = summedMetrics[2];
      }

      if (rank == 0) printThroughputInformation(size, numMessages, bandwidthMiBps, messageRateMpps);
    }

    MPICHECK_FATAL(MPI_Barrier(MPI_COMM_WORLD));

    size_t next = nextSize(args, size);
    if (next <= size) break;
    size = next;
  }

  MPICHECK_FATAL(MPI_Barrier(MPI_COMM_WORLD));
  ncclTestGinDevCommDestroy(comm, &ctx);
  NCCLCHECK_FATAL(ncclCommDestroy(comm));
  if (measures) {
    CUDACHECK_FATAL(cudaEventDestroy(start));
    CUDACHECK_FATAL(cudaEventDestroy(stop));
  }
  CUDACHECK_FATAL(cudaStreamDestroy(stream));
  MPICHECK_FATAL(MPI_Finalize());
}
