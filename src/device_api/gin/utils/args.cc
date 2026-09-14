#include "args.h"
#include "gin_context.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <getopt.h>
#include <stdint.h>

static void printUsage(const char* argv0) {
  fprintf(stderr,
    "Usage: %s [OPTIONS]\n"
    "  -b, --minbytes <size>            Minimum message size in bytes (default: %zu; K/M/G suffix OK)\n"
    "  -e, --maxbytes <size>            Maximum message size in bytes (default: 4M; K/M/G suffix OK)\n"
    "  -f <factor>                      Multiplicative step factor for size sweep (default: 2)\n"
    "  -i <stepbytes>                   Additive step in bytes; overrides -f when non-zero (default: 0)\n"
    "  -w <iters>                       Warmup round trips per size point (default: 50, 0 to disable)\n"
    "  -n, --iters <iters>              Measured round trips per size point (default: 500)\n"
    "  -c, --num_ctas <num_ctas>        Number of CTAs (default: 1)\n"
    "  -t, --num_threads <num_threads>  Number of threads per CTA (default: 1)\n"
    "  --gin_skip_credit_check          Skip GIN credit check (default: off)\n"
    "  --gin_op <put|put_signal|put_counter>  GIN operation to benchmark, for binaries covering more than one "
    "(default: put)\n"
    "  --gin_strong_signal              Use a Strong signal instead of Weak, for tests that support both (default: off, i.e. weak)\n"
    "  --gin_ag                         Enable ncclGinOptFlagsAggregateRequests on throughput tests (default: off)\n"
    "  --gin_bd                         Bidirectional bandwidth: both ranks send to each other; reported metric is the sum of each rank's average (throughput tests only, default: off)\n"
    "  --local_mem_type <device|host>   Memory type of local rank (default: device)\n"
    "  --remote_mem_type <device|host>  Memory type of remote rank (default: device)\n"
    "  --gin_rsm <thread|cta|gpu>       GIN resource sharing mode (default: gpu)\n"
    "  --gin_tx_depth <depth>           GIN Send (queue) depth (default: 1024)\n",
    argv0, sizeof(int));
}

static size_t parseSize(const char* str, const char* opt) {
  char* end;
  errno = 0;
  long long val = strtoll(str, &end, 10);
  if (end == str || errno != 0) {
    fprintf(stderr, "Error: invalid value for %s: '%s'\n", opt, str);
    exit(EXIT_FAILURE);
  }
  if (val < 0) {
    fprintf(stderr, "Error: %s must be non-negative (got %lld)\n", opt, val);
    exit(EXIT_FAILURE);
  }
  size_t mult = 1;
  if (*end == 'K' || *end == 'k') { mult = 1024ULL; end++; }
  else if (*end == 'M' || *end == 'm') { mult = 1024ULL * 1024; end++; }
  else if (*end == 'G' || *end == 'g') { mult = 1024ULL * 1024 * 1024; end++; }
  if (*end != '\0') {
    fprintf(stderr,
      "Error: malformed value for %s: '%s' (only K/M/G suffixes are recognised)\n", opt, str);
    exit(EXIT_FAILURE);
  }
  if (val != 0 && (unsigned long long)val > (unsigned long long)SIZE_MAX / mult) {
    fprintf(stderr, "Error: value for %s is too large: '%s'\n", opt, str);
    exit(EXIT_FAILURE);
  }
  return (size_t)val * mult;
}

// Helper function to parse an integer argument
static int parseIntArg(const char* str, const char* opt) {
  char* end;
  errno = 0;
  long val = strtol(str, &end, 10);
  if (end == str || *end != '\0' || errno != 0) {
    fprintf(stderr, "Error: invalid integer for %s: '%s'\n", opt, str);
    exit(EXIT_FAILURE);
  }
  return (int)val;
}
static ncclGinRsm_t parseRsm(const char* str) {
  if (strcmp(str, "thread") == 0) return ncclGinRsmThread;
  if (strcmp(str, "cta") == 0) return ncclGinRsmCta;
  if (strcmp(str, "gpu") == 0) return ncclGinRsmGpu;
  fprintf(stderr,
    "Error: invalid value for --gin_rsm: '%s' (expected thread, cta, or gpu)\n", str);
  exit(EXIT_FAILURE);
}

static ncclGinOp_t parseOp(const char* str) {
  if (strcmp(str, "put") == 0) return ncclGinOpPut;
  if (strcmp(str, "put_signal") == 0) return ncclGinOpPutSignal;
  if (strcmp(str, "put_counter") == 0) return ncclGinOpPutCount;
  fprintf(stderr,
    "Error: invalid value for --gin_op: '%s' (expected put, put_signal, or put_counter)\n", str);
  exit(EXIT_FAILURE);
}

static ncclGinMemoryType_t parseMemoryType(const char* str, bool isLocal) {
  if (strcmp(str, "device") == 0) return ncclGinMemoryDevice;
  if (strcmp(str, "host") == 0) return ncclGinMemoryHost;
  fprintf(stderr, "Error: invalid value for --%s: '%s' (expected device or host)\n", isLocal ? "local_mem_type" : "remote_mem_type", str);
  exit(EXIT_FAILURE);
}

const char* ncclTestGinOpName(ncclGinOp_t op) {
  switch (op) {
  case ncclGinOpPut: return "put";
  case ncclGinOpPutSignal: return "put_signal";
  case ncclGinOpPutCount: return "put_counter";
  default: return "unset";
  }
}

const char* ncclTestGinRsmName(ncclGinRsm_t rsm) {
  switch (rsm) {
  case ncclGinRsmThread: return "thread";
  case ncclGinRsmCta: return "cta";
  case ncclGinRsmGpu: return "gpu";
  default: return "unset";
  }
}

const char* ncclTestGinMemoryTypeName(ncclGinMemoryType_t memory) {
  return memory == ncclGinMemoryHost ? "host" : "device";
}

void ncclTestGinParseArgs(int argc, char** argv, ginArgs_t* args) {
  args->minBytes = sizeof(int);
  args->maxBytes = 4 * 1024 * 1024;
  args->stepFactor = 2;
  args->stepBytes = 0;
  args->warmupIters = 50;
  args->iters = 500;
  args->numCtas = 1;
  args->numThreads = 1;
  args->ginSkipCreditCheck = 0;
  args->ginStrongSignal = false;
  args->ginAggregateRequests = false;
  args->ginBidirectional = false;
  args->localMemoryType = ncclGinMemoryDevice;
  args->remoteMemoryType = ncclGinMemoryDevice;
  args->queueDepth = 1024;
  args->ginRsm = ncclGinRsmUnset;
  args->ginOp = ncclGinOpUnset;

  static const struct option longOpts[] = {
    {"minbytes", required_argument, NULL, 'b'},
    {"maxbytes", required_argument, NULL, 'e'},
    {"iters", required_argument, NULL, 'n'},
    {"gin_skip_credit_check", no_argument, NULL, 0},
    {"gin_op", required_argument, NULL, 0},
    {"gin_strong_signal", no_argument, NULL, 0},
    {"gin_ag", no_argument, NULL, 0},
    {"gin_bd", no_argument, NULL, 0},
    {"local_mem_type", required_argument, NULL, 0},
    {"remote_mem_type", required_argument, NULL, 0},
    {"gin_rsm", required_argument, NULL, 0},
    {"gin_tx_depth", required_argument, NULL, 0},
    {"num_ctas", required_argument, NULL, 'c'},
    {"num_threads", required_argument, NULL, 't'},
    {NULL, 0, NULL, 0}
  };

  int opt, idx = 0;
  while ((opt = getopt_long(argc, argv, "b:e:f:i:w:n:c:t:", longOpts, &idx)) != -1) {
    switch (opt) {
      case 'b': args->minBytes = parseSize(optarg, "-b/--minbytes"); break;
      case 'e': args->maxBytes = parseSize(optarg, "-e/--maxbytes"); break;
      case 'f': args->stepFactor = parseIntArg(optarg, "-f"); break;
      case 'i': args->stepBytes = parseSize(optarg, "-i"); break;
      case 'w': args->warmupIters = parseIntArg(optarg, "-w"); break;
      case 'n': args->iters = parseIntArg(optarg, "-n/--iters"); break;
      case 'c': args->numCtas = parseIntArg(optarg, "-c/--num_ctas"); break;
      case 't': args->numThreads = parseIntArg(optarg, "-t/--num_threads"); break;
      case 0:
        if (strcmp(longOpts[idx].name, "gin_skip_credit_check") == 0) {
          args->ginSkipCreditCheck = 1;
        } else if (strcmp(longOpts[idx].name, "gin_op") == 0) {
          args->ginOp = parseOp(optarg);
        } else if (strcmp(longOpts[idx].name, "gin_strong_signal") == 0) {
          args->ginStrongSignal = true;
        } else if (strcmp(longOpts[idx].name, "gin_ag") == 0) {
          args->ginAggregateRequests = true;
        } else if (strcmp(longOpts[idx].name, "gin_bd") == 0) {
          args->ginBidirectional = true;
        } else if (strcmp(longOpts[idx].name, "local_mem_type") == 0) {
          args->localMemoryType = parseMemoryType(optarg, true);
        } else if (strcmp(longOpts[idx].name, "remote_mem_type") == 0) {
          args->remoteMemoryType = parseMemoryType(optarg, false);
        } else if (strcmp(longOpts[idx].name, "gin_rsm") == 0) {
          args->ginRsm = parseRsm(optarg);
        } else if (strcmp(longOpts[idx].name, "gin_tx_depth") == 0) {
          args->queueDepth = parseIntArg(optarg, "--gin_tx_depth");
        }
        break;
      default:
        printUsage(argv[0]);
        exit(EXIT_FAILURE);
    }
  }

  if (optind < argc) {
    fprintf(stderr, "Error: unexpected positional argument: '%s'\n", argv[optind]);
    printUsage(argv[0]);
    exit(EXIT_FAILURE);
  }

  /* --- validation (all before MPI_Init) --- */
  if (args->minBytes <= 0) {
    fprintf(stderr, "Error: --minbytes must be positive\n");
    exit(EXIT_FAILURE);
  }
  if (args->maxBytes <= 0) {
    fprintf(stderr, "Error: --maxbytes must be positive\n");
    exit(EXIT_FAILURE);
  }
  if (args->minBytes > args->maxBytes) {
    fprintf(stderr,
      "Error: --minbytes (%zu) must be <= --maxbytes (%zu)\n",
      args->minBytes, args->maxBytes);
    exit(EXIT_FAILURE);
  }
  if (args->minBytes % sizeof(int) != 0) {
    fprintf(stderr,
      "Error: --minbytes (%zu) must be a multiple of sizeof(int) (%zu)\n",
      args->minBytes, sizeof(int));
    exit(EXIT_FAILURE);
  }
  if (args->maxBytes % sizeof(int) != 0) {
    fprintf(stderr,
      "Error: --maxbytes (%zu) must be a multiple of sizeof(int) (%zu)\n",
      args->maxBytes, sizeof(int));
    exit(EXIT_FAILURE);
  }
  if (args->stepFactor <= 1 && args->stepBytes == 0) {
    fprintf(stderr, "Error: -f size factor must be > 1 when -i is not specified\n");
    exit(EXIT_FAILURE);
  }
  if (args->stepBytes != 0 && args->stepBytes % sizeof(int) != 0) {
    fprintf(stderr,
      "Error: -i stepbytes (%zu) must be a multiple of sizeof(int) (%zu)\n",
      args->stepBytes, sizeof(int));
    exit(EXIT_FAILURE);
  }
  if (args->warmupIters < 0) {
    fprintf(stderr, "Error: -w warmup_iters must be >= 0\n");
    exit(EXIT_FAILURE);
  }
  if (args->iters <= 0) {
    fprintf(stderr, "Error: -n/--iters must be > 0\n");
    exit(EXIT_FAILURE);
  }
  if (args->numCtas <= 0) {
    fprintf(stderr, "Error: -c/--num_ctas must be > 0\n");
    exit(EXIT_FAILURE);
  }
  if (args->numThreads <= 0) {
    fprintf(stderr, "Error: -t/--num_threads must be > 0\n");
    exit(EXIT_FAILURE);
  }
  if (args->queueDepth <= 0) {
    fprintf(stderr, "Error: --gin_tx_depth must be > 0\n");
    exit(EXIT_FAILURE);
  }
  {
    const size_t maxBuf = ncclTestGinMaxBufferBytes();
    if (args->maxBytes > maxBuf) {
      fprintf(stderr,
        "Error: -e/--maxbytes (%zu) exceeds the %zu-byte per-buffer limit "
        "(20 * 2^31, 40 GiB). Reduce -e.\n",
        args->maxBytes, maxBuf);
      exit(EXIT_FAILURE);
    }
  }
}

static void configureOp(ginArgs_t* args, const ginTestCaps_t* caps) {
  if (caps->opMask == 0) {
    if (args->ginOp != ncclGinOpUnset) {
      fprintf(stderr, "Error: --gin_op is only supported by the put benchmarks\n");
      exit(EXIT_FAILURE);
    }
    return;
  }
  if (args->ginOp == ncclGinOpUnset) args->ginOp = caps->defaultOp;
  if ((caps->opMask & GIN_OP_BIT(args->ginOp)) == 0) {
    fprintf(stderr, "Error: --gin_op %s is not supported by this benchmark (accepted:",
            ncclTestGinOpName(args->ginOp));
    for (int op = ncclGinOpPut; op <= ncclGinOpPutCount; op++) {
      if (caps->opMask & GIN_OP_BIT(op)) fprintf(stderr, " %s", ncclTestGinOpName((ncclGinOp_t)op));
    }
    fprintf(stderr, ")\n");
    exit(EXIT_FAILURE);
  }
}

void ncclTestGinConfigureArgs(ginArgs_t* args, const ginTestCaps_t* caps) {
  if (args->ginRsm == ncclGinRsmUnset) args->ginRsm = caps->defaultRsm;
  if (!caps->allowThreadRsm && args->ginRsm == ncclGinRsmThread) {
    fprintf(stderr, "Error: --gin_rsm thread is not supported by throughput tests (expected cta or gpu)\n");
    exit(EXIT_FAILURE);
  }

  configureOp(args, caps);

  const bool signalsInUse = caps->isSignalOp || args->ginOp == ncclGinOpPutSignal;
  if (!signalsInUse && args->ginStrongSignal) {
    fprintf(stderr,
      "Error: --gin_strong_signal is only supported by the signal tests and by put tests run with "
      "--gin_op put_signal\n");
    exit(EXIT_FAILURE);
  }
  if (!caps->allowAggregateRequests && args->ginAggregateRequests) {
    fprintf(stderr, "Error: --gin_ag is only supported by throughput tests\n");
    exit(EXIT_FAILURE);
  }
  if (!caps->allowBidirectional && args->ginBidirectional) {
    fprintf(stderr, "Error: --gin_bd is only supported by throughput (bandwidth) tests\n");
    exit(EXIT_FAILURE);
  }
  if (!caps->canUseHostMemory && (args->localMemoryType == ncclGinMemoryHost || args->remoteMemoryType == ncclGinMemoryHost)) {
    fprintf(stderr, "Error: --remote_mem_type host/--local_mem_type host is only supported by ginGetBW_perf\n");
    exit(EXIT_FAILURE);
  }
  if (!caps->allowMultiCtaThreads && (args->numCtas != 1 || args->numThreads != 1)) {
    fprintf(stderr, "Error: Setting > 1 cta or thread is only supported by throughput tests\n");
    exit(EXIT_FAILURE);
  }
  if (args->ginAggregateRequests && args->numThreads % 32 != 0) {
    fprintf(stderr,
      "Error: --gin_ag requires -t/--num_threads to be a multiple of 32 (got %d)\n",
      args->numThreads);
    exit(EXIT_FAILURE);
  }
}
