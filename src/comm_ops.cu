/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * NCCL Communicator Operations Performance Test
 *
 * This test measures the performance of NCCL communicator lifecycle operations:
 * initialization, splitting, shrinking, and growing. Unlike other tests in src/,
 * this does not use the common testColl/testEngine framework as it focuses on
 * operations performed on the communicator itself.
 *
 * Tests:
 *   - init:   Measures ncclCommInitRankScalable performance
 *   - split:  Measures ncclCommSplit performance
 *   - shrink: Measures ncclCommShrink performance
 *   - grow:   Measures ncclCommGrow performance
 */

#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <getopt.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <cerrno>
#include <string>
#include <vector>

#ifndef MPI_SUPPORT
int main() {
  printf("This test requires MPI support, but was compiled without it.\n");
  return 1;
}
#else

#include "nccl.h"
#include "mpi.h"

#define ABORT_EXIT do { \
  fprintf(stderr, "Aborting due to error in %s:%d\n", __FILE__, __LINE__); \
  fflush(stderr); fflush(stdout); \
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); \
} while(0)

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Cuda failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    ABORT_EXIT;                                     \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r != ncclSuccess) {                           \
    printf("NCCL failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    ABORT_EXIT;                                     \
  }                                                 \
} while(0)

#define MPICHECK(cmd) do {                         \
  int r = cmd;                                     \
  if (r != MPI_SUCCESS) {                          \
    int err_len;                                   \
    char err_string[MPI_MAX_ERROR_STRING];         \
    MPI_Error_string(r, err_string, &err_len);     \
    printf("MPI failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,err_string); \
    ABORT_EXIT;                                     \
  }                                                 \
} while(0)

int main_ret_val = 0;

void print_help();

enum timer_enums {
  mpi_init,
  mpi_barrier,
  mpi_allgatherv,
  work_prepare,
  nccl_init,
  nccl_split,
  nccl_grow,
  nccl_allreduce,
  nccl_finalize,
  nccl_destroy,
  nccl_abort,
  nccl_shrink,
  cuda_init,
  cuda_stream_sync,
  timer_enum_count,
};

// timer_strings array - order must match timer_enums exactly
const char* const timer_strings[timer_enum_count] = {
  "(MPI) Initialization",           // mpi_init
  "(MPI) Barrier",                  // mpi_barrier
  "(MPI) Allgatherv for ncclUniqueId", // mpi_allgatherv
  "(CUDA) Allocate mem and streams",   // work_prepare
  "NCCL ncclCommInitRankScalable",  // nccl_init
  "NCCL ncclCommSplit",             // nccl_split
  "NCCL ncclCommGrow",              // nccl_grow
  "NCCL ncclAllReduce",             // nccl_allreduce
  "NCCL ncclCommFinalize",          // nccl_finalize
  "NCCL ncclCommDestroy",           // nccl_destroy
  "NCCL ncclCommAbort",             // nccl_abort
  "NCCL ncclCommShrink",            // nccl_shrink
  "(CUDA) Initialization",          // cuda_init
  "(CUDA) Stream Synchronize",      // cuda_stream_sync
};

const int nccl_flag = 0x1 << 1;
const int cuda_flag = 0x1 << 2;
const int mpi_flag  = 0x1 << 3;
// timer_flags array - order must match timer_enums exactly
const int timer_flags[timer_enum_count] = {
  mpi_flag,   // mpi_init
  mpi_flag,   // mpi_barrier
  mpi_flag,   // mpi_allgatherv
  cuda_flag,  // work_prepare
  nccl_flag,  // nccl_init
  nccl_flag,  // nccl_split
  nccl_flag,  // nccl_grow
  nccl_flag,  // nccl_allreduce
  nccl_flag,  // nccl_finalize
  nccl_flag,  // nccl_destroy
  nccl_flag,  // nccl_abort
  nccl_flag,  // nccl_shrink
  cuda_flag,  // cuda_init
  cuda_flag,  // cuda_stream_sync
};
bool timer_type_nccl(int timer_enum) {
  return (timer_flags[timer_enum] & nccl_flag) == nccl_flag;
}

struct timer_record_t {
  int flush_interval;
  bool currently_enabled = true;
  int count = 0;
  MPI_Comm comm = MPI_COMM_NULL;
  int rank = 0;
  int rank_count = 1;

  uint64_t timer_enum_start[timer_enum_count];

  /* locally-written for each timing call. */
  double* local;
  double* local_sq;
  int* id;
  int* participated;

  /* root only, reduced from all ranks. */
  double* max;
  double* sum;
  double* sumsq;
  int* participant_count;

  /* root-only, sorted by timer type. */
  int rounds_by_type[timer_enum_count];
  double sum_by_type[timer_enum_count];
  double sumsq_by_type[timer_enum_count];
  double max_by_type[timer_enum_count];
  int participant_count_by_type[timer_enum_count];
};

void flush_timing(timer_record_t* timing);

inline uint64_t clockNano() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return uint64_t(ts.tv_sec)*1000*1000*1000 + ts.tv_nsec;
}

enum test_kinds {
  test_init,
  test_split,
  test_shrink,
  test_grow
};
enum step_kinds {
  kind_factor,
  kind_step,
  kind_fixed,
};

struct args_t {
  int timing_flush_interval;
  int warmup;
  int resize_kind;
  double resize_num;
  int iterations;
  int work_iterations;
  int test_kind;
  bool aborts;
  int beginranks;
  int endranks;
  int sweep_kind;
  double sweep_num;
  int verbose;
  int nccl_only;
  int share;
  int root_every;
} args;

/* parse a string and return an enum and number.  Valid options are:
     step-<I>
     factor-<F>
     fixed
  There is an assumption that step is always positive and factor is always greater than 1.
*/
int parse_step_kinds(char *arg, int *kind, double *num) {
  char *numpart;
  if (strncmp(optarg, "factor-", 7) == 0) {
    *kind = kind_factor;
    numpart = &arg[7];
  } else if (strncmp(optarg, "step-",5) == 0) {
    *kind = kind_step;
    numpart = &arg[5];
  } else if (strncmp(optarg, "fixed",5) == 0) {
    *kind = kind_fixed;
    *num = 0.0;
    return 0;
  } else {
    fprintf(stderr, "Error: Unknown sweep or resize kind '%s'.  Expected factor-N.NN, step-N, or fixed.\n", arg);
    *num = 0.0;
    return 0;
  }
  errno = 0;
  *num = strtod(numpart, NULL);
  if (errno != 0) {
    fprintf(stderr,"strtod cannot parse (\"%s\"): %s\n", numpart, strerror(errno));
    errno=0;
    return 1;
  }
  if (*num <= 0.0) {
    fprintf(stderr, "Error: invalid number (\"%s\").  Expected positive finite number.\n", numpart);
    return 1;
  }

  if (*kind == kind_factor && *num < 1.0) {
    *num = 1.0/(*num);
  }
  if (*kind == kind_step && *num < 0.0) {
    *num = - (*num);
  }
  return 0;
}

void parse_args(int argc, char **argv) {
  // Set defaults
  args.resize_kind = kind_factor;
  args.resize_num = 2.0;
  args.test_kind = test_init;
  args.aborts = false;
  args.verbose = 0;
  args.timing_flush_interval = 1;
  args.sweep_kind = kind_factor;
  args.sweep_num  = 2;
  args.beginranks = 0;
  args.endranks = 0;
  args.warmup = 1;
  args.nccl_only = 1;
  /* enough for 1 warmup and 1 non-warmup iteration. */
  args.work_iterations = 2;
  args.iterations = 2;
  args.root_every = 0;

  // Check if we have at least the program name and test type
  if (argc < 2) {
    fprintf(stderr, "Error: You must specify the test to execute: init, split, or shrink\n");
    print_help();
    ABORT_EXIT;
  }
  // Reset getopt state and skip the test type argument
  optind = 1;
  // Parse the test type (first positional argument)
  if (strcmp(argv[1], "init") == 0) {
    args.test_kind = test_init;
    optind++;
  } else if (strcmp(argv[1], "split") == 0) {
    args.test_kind = test_split;
    optind++;
  } else if (strcmp(argv[1], "shrink") == 0) {
    args.test_kind = test_shrink;
    optind++;
  } else if (strcmp(argv[1], "grow") == 0) {
    args.test_kind = test_grow;
    optind++;
  } else if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "help") == 0) {
    print_help();
    MPICHECK(MPI_Finalize());
    exit(0);
  } else {
    fprintf(stderr, "Error: Unknown test type '%s'. Must be one of: init, split, or shrink\n", argv[1]);
    print_help();
    ABORT_EXIT;
  }

  // Define long options (alphabetical order)
  static struct option long_options[] = {
    {"abort",          no_argument,       0, 'a'},
    {"all-timers",     no_argument,       0, 'A'},
    {"sweep-begin",    required_argument, 0, 'b'},
    {"sweep-end",      required_argument, 0, 'e'},
    {"flush-interval", required_argument, 0, 'F'},
    {"help",           no_argument,       0, 'h'},
    {"iterations",     required_argument, 0, 'i'},
    {"share",          no_argument,       0, 'S'},
    {"resize",         required_argument, 0, 'r'},
    {"root-every",     required_argument, 0, 'R'},
    {"sweep-comms",    required_argument, 0, 's'},
    {"verbose",        no_argument,       0, 'v'},
    {"warmup",         required_argument, 0, 'w'},
    {"work-iterations",required_argument, 0, 'W'},
    {0, 0, 0, 0}
  };

  // Parse options (alphabetical order by short option)
  int opt;
  int option_index = 0;
  while ((opt = getopt_long(argc, argv, "aAb:e:F:hi:k:r:R:s:Svw:W:", long_options, &option_index)) != -1) {
    switch (opt) {
      case 'a':
        args.aborts = true;
        break;
      case 'b':
        args.beginranks = atoi(optarg);
        break;
      case 'e':
        args.endranks = atoi(optarg);
        break;
      case 's':
        if (parse_step_kinds(optarg, &args.sweep_kind, &args.sweep_num)) {
          print_help();
          ABORT_EXIT;
        }
        break;
      case 'F':
        args.timing_flush_interval = atoi(optarg);
        if (args.timing_flush_interval <= 0) {
          fprintf(stderr, "Error: Flush interval must be greater than 0\n");
          ABORT_EXIT;
        }
        break;
      case 'h':
        print_help();
        MPICHECK(MPI_Finalize());
        exit(0);
        break;
      case 'i':
        args.iterations = atoi(optarg);
        break;
      case 'r':
        if (parse_step_kinds(optarg, &args.resize_kind, &args.resize_num)) {
          print_help();
          ABORT_EXIT;
        }
        break;
      case 'A':
        args.nccl_only = 0;
        break;
      case 'R':
        args.root_every = atoi(optarg);
        break;
      case 'S':
        args.share = true;
        break;
      case 'v':
        args.verbose = 1;
        break;
      case 'w':
        args.warmup = atoi(optarg);
        if (args.warmup < 0) {
          fprintf(stderr, "Error: Warmup iterations must be greater than or equal to 0\n");
          ABORT_EXIT;
        }
        break;
      case 'W':
        args.work_iterations = atoi(optarg);
        break;
      default:
        print_help();
        fprintf(stderr, "Error: Unknown option '%s'\n", argv[optind-1]);
        ABORT_EXIT;
    }
  }

  int nranks;
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

  if (args.beginranks == 0) {
    args.beginranks = nranks;
  }
  if (args.endranks == 0) {
    args.endranks = 1;
  }

  args.beginranks = std::min(std::max(args.beginranks, 0), nranks);
  args.endranks = std::min(std::max(args.endranks, 0), nranks);

  bool going_up = args.beginranks < args.endranks;
  if (args.sweep_kind == kind_factor) {
    bool _going_up = args.sweep_num > 1.0;
    if (going_up != _going_up)
      args.sweep_num = 1.0 / args.sweep_num;
  } else if (args.sweep_kind == kind_step) {
    bool _going_up = args.sweep_num > 0;
    if (going_up != _going_up)
      args.sweep_num = -args.sweep_num;
  }

  if (args.test_kind == test_split && args.share && args.aborts) {
    fprintf(stderr, "Error: split test with both --share and --abort is not valid. Disabling --abort.\n");
    ABORT_EXIT;
  }

}

static inline void record_timer(timer_record_t* timing, uint64_t elapsed, int timer_enum, bool record) {
  if (!timing->currently_enabled) return;
  assert(timer_enum >= 0 && timer_enum < timer_enum_count);
  if (timing->count < timing->flush_interval) {
    timing->local[timing->count] = (double)elapsed;
    timing->id[timing->count] = timer_enum;
    timing->participated[timing->count] = record;
    timing->local_sq[timing->count] = pow(timing->local[timing->count], 2.0);
  }
  timing->count++;
  if (timing->count == timing->flush_interval) {
    flush_timing(timing);
  }
}

static inline void set_timer_enabled(timer_record_t* timing, bool enabled) {
  timing->currently_enabled = enabled;
}

static inline void start_timer(timer_record_t* timing, int timer_enum) {
  if (!timing->currently_enabled) return;
  timing->timer_enum_start[timer_enum] = clockNano();
}

static inline void stop_timer(timer_record_t* timing, int timer_enum, bool record=true) {
  if (!timing->currently_enabled) return;
  uint64_t elapsed;
  if (record) {
    elapsed = clockNano() - timing->timer_enum_start[timer_enum];
  } else {
    elapsed = 0;
  }
  record_timer(timing, elapsed, timer_enum, record);
}

inline void skip_timer(timer_record_t* timing, int timer_enum) {
  record_timer(timing, 0, timer_enum, false);
}

struct work_t {
  int *ddata;
  int *ddata_out;
  int *hdata;
  cudaStream_t stream;
} work_ctx;

/* buffer size isn't important for this test. */
const int worksize = 1024;

/* Create the stream and allocate the work buffers.*/
void prepare_work(timer_record_t* timing) {
  start_timer(timing, work_prepare);
  // CUDA stream creation
  CUDACHECK(cudaStreamCreateWithFlags(&work_ctx.stream, cudaStreamNonBlocking));
  CUDACHECK(cudaMalloc(&work_ctx.ddata, worksize*sizeof(int)));
  CUDACHECK(cudaMalloc(&work_ctx.ddata_out, worksize*sizeof(int)));
  work_ctx.hdata = (int*) malloc(worksize*sizeof(int));
  stop_timer(timing, work_prepare);
}

/* Execute ncclAllReduce and cudaStreamSynchronize.*/
void do_work(timer_record_t* timing, ncclComm_t comm) {
  if (comm == NULL) {
    skip_timer(timing, nccl_allreduce);
    skip_timer(timing, cuda_stream_sync);
    return;
  }
  start_timer(timing, nccl_allreduce);
  NCCLCHECK(ncclAllReduce(work_ctx.ddata, work_ctx.ddata_out, worksize, ncclInt, ncclSum, comm, work_ctx.stream));
  stop_timer(timing, nccl_allreduce);
  start_timer(timing, cuda_stream_sync);
  CUDACHECK(cudaStreamSynchronize(work_ctx.stream));
  stop_timer(timing, cuda_stream_sync);
}

/* Free the work buffers and destroy the stream.*/
void cleanup_work() {
  CUDACHECK(cudaFree(work_ctx.ddata));
  CUDACHECK(cudaFree(work_ctx.ddata_out));
  free(work_ctx.hdata);
  CUDACHECK(cudaStreamDestroy(work_ctx.stream));
}

static int rootCount(int nranks) {
  if (args.root_every > 0) {
    return 1 + ((nranks-1) / args.root_every);
  } else {
    return 1;
  }
}
/*
 * Determines if a given rank should host a root for scalable API initialization.
 *
 * Returns: 1 if this rank hosts a root, 0 otherwise
 */
static int isRoot(int rank) {
  if (args.root_every > 0)
    return (rank % args.root_every) == 0;
  else
    return rank == 0;
}

void timing_print_verbose(timer_record_t* timing) {
  int nranks;
  if (args.verbose == 0) return;
  MPICHECK(MPI_Comm_size(timing->comm, &nranks));
  int nrec = timing->count < timing->flush_interval ? timing->count : timing->flush_interval;
  for (int i=0; i<nrec; i++) {
    if (timing->participant_count[i] == 0) continue;
    if (args.nccl_only && !timer_type_nccl(timing->id[i])) continue;
    double avg = timing->sum[i] / timing->participant_count[i];
    double A = timing->sumsq[i] / timing->participant_count[i];
    double B = pow(timing->sum[i] / timing->participant_count[i], 2.0);
    double stddev = sqrt(A - B);
    printf("%36s: Max %9.3f ms, Avg %8.3f ms, Stddev %8.3f ms, [%4d/%4d]\n", \
      timer_strings[timing->id[i]], timing->max[i]*1e-6, avg*1e-6, stddev*1e-6, (int)timing->participant_count[i], nranks);
  }
  fflush(stdout);
}

/*
 * Performs MPI reductions to gather timing statistics from all ranks.
 *
 * Computes aggregate statistics across all ranks
 * and accumulates them into per-timer-type arrays for summary reporting.
 *
 * Results are stored on rank 0 only. Also validates that all ranks recorded
 * the same number of timing events.
 */
void gather_timing(timer_record_t* timing) {
  int *all_counts = (int*)malloc(sizeof(int) * timing->rank_count);

  if (timing->comm == MPI_COMM_NULL) {
    all_counts[0] = timing->count;
    for (int i=0; i<timing->flush_interval; i++) {
      timing->max[i] = timing->local[i];
      timing->sum[i] = timing->local[i];
      timing->participant_count[i] = timing->participated[i];
      timing->sumsq[i] = timing->local_sq[i];
    }
  } else {
    MPICHECK(MPI_Gather(&timing->count, 1, MPI_INT, all_counts, 1, MPI_INT, 0, timing->comm));
    MPICHECK(MPI_Reduce(timing->local, timing->max, timing->flush_interval, MPI_DOUBLE, MPI_MAX, 0, timing->comm));
    MPICHECK(MPI_Reduce(timing->local, timing->sum, timing->flush_interval, MPI_DOUBLE, MPI_SUM, 0, timing->comm));
    MPICHECK(MPI_Reduce(timing->participated, timing->participant_count, timing->flush_interval, MPI_INT, MPI_SUM, 0, timing->comm));
    MPICHECK(MPI_Reduce(timing->local_sq, timing->sumsq, timing->flush_interval, MPI_DOUBLE, MPI_SUM, 0, timing->comm));
  }

  if (timing->rank == 0) {
    bool all_match = true;
    for (int i=0; i<timing->rank_count; i++) {
      if (all_counts[i] != timing->count) {
        printf("INTERNAL TEST ERROR: Rank %d: timer_records_count mismatch: %d != %d\n", i, all_counts[i], timing->count);
        all_match = false;
        main_ret_val = 1;
      }
    }
    for (int jj=0; jj<timing->count; jj++) {
      int id = timing->id[jj];
      if (timing->participant_count[jj] > 0) {
        timing->sum_by_type[id] += timing->sum[jj];
        timing->sumsq_by_type[id] += timing->sumsq[jj];
        timing->max_by_type[id] = std::max(timing->max_by_type[id], timing->max[jj]);
        timing->participant_count_by_type[id] += timing->participant_count[jj];
        timing->rounds_by_type[id]++;
      }
    }
    if (all_match) {
      // printf("Sanity check pass: All ranks recorded %d timers\n", timing->count);
    }
    if (timing->count > timing->flush_interval) {
      printf("ERROR: Max timer records (%d) exceeded limit (%d)\n", timing->count, timing->flush_interval);
      main_ret_val = 1;
    }
  }
  free(all_counts);
}

void flush_timing(timer_record_t* timing) {
  if (timing->comm != MPI_COMM_NULL) {
    int rank;
    gather_timing(timing);

    MPICHECK(MPI_Comm_rank(timing->comm, &rank));
    if (rank == 0) {
      timing_print_verbose(timing);
    }
    MPICHECK(MPI_Barrier(timing->comm));
  }
  timing->count = 0;
}

/*
 * Configures NCCL communicator settings based on command-line arguments.
 */
void configure_communicator(ncclConfig_t* config) {
  config->splitShare = args.share ? 1 : 0;
  config->shrinkShare = args.share ? 1 : 0;
}

/* The main test function for the init test.
 *
 * Each iteration creates a new communicator and performs some work on it
 * before destroying it.
 */
void main_init(int argc, char **argv, timer_record_t* timing, MPI_Comm mpicomm) {
  ncclUniqueId commId;
  ncclComm_t ncclcomm;
  int nranks, rank;
  int nroots;

  set_timer_enabled(timing, 0 >= args.warmup);

  MPICHECK(MPI_Comm_size(mpicomm, &nranks));
  MPICHECK(MPI_Comm_rank(mpicomm, &rank));

  start_timer(timing, mpi_barrier);
  MPICHECK(MPI_Barrier(mpicomm));
  stop_timer(timing, mpi_barrier);

  nroots = rootCount(nranks);
  ncclUniqueId *id_array= (ncclUniqueId*)calloc(nroots, NCCL_UNIQUE_ID_BYTES);

  for (int j=0; j<args.iterations + args.warmup; j++) {
    set_timer_enabled(timing, j >= args.warmup);

    // NCCL Communicator creation
    if (isRoot(rank)){
      NCCLCHECK(ncclGetUniqueId(&commId));
    }

    int* recv_count = (int*)malloc(sizeof(int) * nranks);
    int* recv_displ = (int*)malloc(sizeof(int) * nranks);
    int c=0;
    for (int i=0; i<nranks; ++i){
      int peer_size = isRoot(i) ? NCCL_UNIQUE_ID_BYTES : 0;
      recv_displ[i] = c;
      recv_count[i] = peer_size;
      c+= peer_size;
    }
    start_timer(timing, mpi_allgatherv);
    MPICHECK(MPI_Allgatherv(&commId,recv_count[rank],MPI_CHAR,id_array,recv_count,recv_displ,MPI_CHAR,mpicomm));
    stop_timer(timing, mpi_allgatherv);
    free(recv_count);
    free(recv_displ);

    start_timer(timing, mpi_barrier);
    MPICHECK(MPI_Barrier(mpicomm));
    stop_timer(timing, mpi_barrier);

    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    configure_communicator(&config);
    start_timer(timing, nccl_init);
    NCCLCHECK(ncclCommInitRankScalable(&ncclcomm, nranks, rank, nroots, id_array, &config));
    stop_timer(timing, nccl_init);

    for (int i=0; i<args.work_iterations; i++) {
      do_work(timing, ncclcomm);
    }

    if (args.aborts) {
      start_timer(timing, nccl_abort);
      NCCLCHECK(ncclCommAbort(ncclcomm));
      stop_timer(timing, nccl_abort);
    } else {
      start_timer(timing, nccl_finalize);
      NCCLCHECK(ncclCommFinalize(ncclcomm));
      stop_timer(timing, nccl_finalize);
      start_timer(timing, nccl_destroy);
      NCCLCHECK(ncclCommDestroy(ncclcomm));
      stop_timer(timing, nccl_destroy);
    }

  }

  free(id_array);
}

/*
 * Determines the color (group ID) for a rank during communicator split/shrink/grow operations.
 *
 * In all cases we return up to two colors (0 or 1).  When factor=3.0 the first third will
 * return 0, while the rest return 1.  When step=5, the first 5 will return 0, and the rest return 1.
 *
 * For grow operations the assumption is only color==0 will join the initial comm, and the
 * remaining will join during grow.
 */
static inline int pick_color(int rank, int count) {
  int color;
  if (args.resize_kind == kind_factor) {
    color = rank >= count/args.resize_num;
  } else {
    color = rank >= count - args.resize_num;
  }
  return color;
}

/*
 * Main test function for both split and shrink tests.
 *
 * Create the NCCL communicator, then perform args.iterations of splits or shrinks from
 * that same parent communicator each time.
 *
 * Each new communicator will have some work performed on it.
 * Note: the parent communicator will not have work performed on it.
 */
void main_split(int argc, char **argv, timer_record_t* timing, MPI_Comm mpicomm) {
  ncclUniqueId commId;
  ncclComm_t ncclcomm;
  int nranks, rank;

  set_timer_enabled(timing, 0 >= args.warmup);

  MPICHECK(MPI_Comm_size(mpicomm, &nranks));
  MPICHECK(MPI_Comm_rank(mpicomm, &rank));

  /* cannot split a single rank */
  if (nranks == 1) return;

  start_timer(timing, mpi_barrier);
  MPICHECK(MPI_Barrier(mpicomm));
  stop_timer(timing, mpi_barrier);

  {
    int nroots = rootCount(nranks);
    ncclUniqueId *id_array= (ncclUniqueId*)calloc(nroots, NCCL_UNIQUE_ID_BYTES);
    // NCCL Communicator creation
    if (isRoot(rank)){
      NCCLCHECK(ncclGetUniqueId(&commId));
    }

    int* recv_count = (int*)malloc(sizeof(int) * nranks);
    int* recv_displ = (int*)malloc(sizeof(int) * nranks);
    int c=0;
    for (int i=0; i<nranks; ++i){
      int peer_size = isRoot(i) ? NCCL_UNIQUE_ID_BYTES : 0;
      recv_displ[i] = c;
      recv_count[i] = peer_size;
      c+= peer_size;
    }
    start_timer(timing, mpi_allgatherv);
    MPICHECK(MPI_Allgatherv(&commId,recv_count[rank],MPI_CHAR,id_array,recv_count,recv_displ,MPI_CHAR,mpicomm));
    stop_timer(timing, mpi_allgatherv);

    free(recv_count);
    free(recv_displ);

    start_timer(timing, nccl_init);
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    configure_communicator(&config);
    NCCLCHECK(ncclCommInitRankScalable(&ncclcomm, nranks, rank, nroots, id_array, &config));
    stop_timer(timing, nccl_init);
    free(id_array);
  }

  int *exclude_ranks = (int*)malloc(sizeof(int) * nranks);

  ncclComm_t split_child;

  /* Do Shrink/Split iterations. */
  for (int j=0; j<args.iterations + args.warmup; j++) {
    set_timer_enabled(timing, j >= args.warmup);

    int ncclRank, ncclCount;
    int exclude_count = 0;
    int shrink_flag = args.aborts ? NCCL_SHRINK_ABORT : NCCL_SHRINK_DEFAULT;

    MPICHECK(MPI_Barrier(mpicomm));

    NCCLCHECK(ncclCommUserRank(ncclcomm, &ncclRank));
    NCCLCHECK(ncclCommCount(ncclcomm, &ncclCount));

    /* it is not possible to split/shrink a single rank */
    if (ncclCount == 1) break;

    int color=0;
    int new_rank=0;

    color = pick_color(ncclRank, ncclCount);
    for (int other_rank=0; other_rank<ncclCount; other_rank++) {
      if (color != pick_color(other_rank, ncclCount) ) {
        exclude_ranks[exclude_count++] = other_rank;
      }
    }

    split_child = NULL;
    if (args.test_kind == test_split) {
      start_timer(timing, nccl_split);
      NCCLCHECK(ncclCommSplit(ncclcomm, color, new_rank, &split_child, NULL));
      stop_timer(timing, nccl_split);
    } else if (args.test_kind == test_shrink) {
      start_timer(timing, nccl_shrink);
      if (color == 0) {
        NCCLCHECK(ncclCommShrink(ncclcomm, exclude_ranks, exclude_count, &split_child, NULL, shrink_flag));
        stop_timer(timing, nccl_shrink);
      } else {
        skip_timer(timing, nccl_shrink);
      }
    }

    /* do_work is safe for null communicators, and is required so that we
    appropriately skip the timers. */
    for (int i=0; i<args.work_iterations; i++) {
      do_work(timing, split_child);
    }

    /* Excluded shrink ranks leave split_child == NULL; skip teardown timers
     * so all ranks stay in lockstep without timing no-ops. */
    if (split_child != NULL) {
      if (args.aborts) {
        start_timer(timing, nccl_abort);
        NCCLCHECK(ncclCommAbort(split_child));
        stop_timer(timing, nccl_abort);
      } else {
        start_timer(timing, nccl_finalize);
        NCCLCHECK(ncclCommFinalize(split_child));
        stop_timer(timing, nccl_finalize);
        start_timer(timing, nccl_destroy);
        NCCLCHECK(ncclCommDestroy(split_child));
        stop_timer(timing, nccl_destroy);
      }
    } else {
      if (args.aborts) {
        skip_timer(timing, nccl_abort);
      } else {
        skip_timer(timing, nccl_finalize);
        skip_timer(timing, nccl_destroy);
      }
    }
  }

  /* Parent communicator was created once for the split/shrink loop. */
  if (args.aborts) {
    start_timer(timing, nccl_abort);
    NCCLCHECK(ncclCommAbort(ncclcomm));
    stop_timer(timing, nccl_abort);
  } else {
    start_timer(timing, nccl_finalize);
    NCCLCHECK(ncclCommFinalize(ncclcomm));
    stop_timer(timing, nccl_finalize);
    start_timer(timing, nccl_destroy);
    NCCLCHECK(ncclCommDestroy(ncclcomm));
    stop_timer(timing, nccl_destroy);
  }
  free(exclude_ranks);
}

void main_grow(int argc, char **argv, timer_record_t* timing, MPI_Comm mpicomm) {
  ncclUniqueId commId_smaller, commId_bigger;
  ncclComm_t ncclComm_smaller, ncclComm_bigger;
  int nranks, rank, color;
  int nroots;

  set_timer_enabled(timing, 0 >= args.warmup);

  MPICHECK(MPI_Comm_size(mpicomm, &nranks));
  MPICHECK(MPI_Comm_rank(mpicomm, &rank));

  int initial_count = 0;
  color = pick_color(rank, nranks);
  for (int other_rank=0; other_rank<nranks; other_rank++) {
    initial_count += (pick_color(other_rank, nranks) == 0);
  }

  /* cannot grow from nothing! */
  if (initial_count == 0) return;
  /* no room to grow!*/
  if (initial_count == nranks) return;

  start_timer(timing, mpi_barrier);
  MPICHECK(MPI_Barrier(mpicomm));
  stop_timer(timing, mpi_barrier);

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  configure_communicator(&config);

  MPI_Comm mpi_color_comm;
  MPICHECK(MPI_Comm_split(mpicomm, color==0, rank, &mpi_color_comm));
  nroots = rootCount(initial_count);
  if (color == 0) {
    ncclUniqueId *id_array = (ncclUniqueId*)calloc(nroots, NCCL_UNIQUE_ID_BYTES);
    // NCCL Communicator creation
    if (isRoot(rank)){
      NCCLCHECK(ncclGetUniqueId(&commId_smaller));
    }

    int* recv_count = (int*)malloc(sizeof(int) * initial_count);
    int* recv_displ = (int*)malloc(sizeof(int) * initial_count);
    int c=0;
    for (int i=0; i<initial_count; ++i){
      int peer_size = isRoot(i) ? NCCL_UNIQUE_ID_BYTES : 0;
      recv_displ[i] = c;
      recv_count[i] = peer_size;
      c+= peer_size;
    }
    start_timer(timing, mpi_allgatherv);
    MPICHECK(MPI_Allgatherv(&commId_smaller,recv_count[rank],MPI_CHAR,id_array,recv_count,recv_displ,MPI_CHAR,mpi_color_comm));
    stop_timer(timing, mpi_allgatherv);
    free(recv_count);
    free(recv_displ);

    start_timer(timing, nccl_init);
    NCCLCHECK(ncclCommInitRankScalable(&ncclComm_smaller, initial_count, rank, nroots, id_array, &config));
    stop_timer(timing, nccl_init);
    free(id_array);
  } else {
    skip_timer(timing, mpi_allgatherv);
    skip_timer(timing, nccl_init);
    ncclComm_smaller = NULL;
  }

  /* do_work before grow. */
  for (int i=0; i<args.work_iterations; i++) {
    do_work(timing, ncclComm_smaller);
  }

  /* Do Grow iterations. */
  for (int j=0; j<args.iterations + args.warmup; j++) {
    set_timer_enabled(timing, j >= args.warmup);

    MPICHECK(MPI_Barrier(mpicomm));

    if (rank == 0) {
      /* assumption: rank0 is always part of color=0*/
      NCCLCHECK(ncclCommGetUniqueId(ncclComm_smaller, &commId_bigger));
    }
    MPICHECK(MPI_Bcast(&commId_bigger, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, mpicomm));

    start_timer(timing, nccl_grow);
    if (color == 0) {
      /* existing ranks provide comm and pass -1 for rank */
      NCCLCHECK(ncclCommGrow(ncclComm_smaller, nranks, &commId_bigger, -1, &ncclComm_bigger, &config));
    } else {
      /* new ranks have NULL input comm and set their rank */
      NCCLCHECK(ncclCommGrow(NULL, nranks, &commId_bigger, rank, &ncclComm_bigger, &config));
    }
    stop_timer(timing, nccl_grow);

    /* do_work after grow. */
    for (int i=0; i<args.work_iterations; i++) {
      do_work(timing, ncclComm_bigger);
    }

    if (args.aborts) {
      start_timer(timing, nccl_abort);
      NCCLCHECK(ncclCommAbort(ncclComm_bigger));
      stop_timer(timing, nccl_abort);
    } else {
      start_timer(timing, nccl_finalize);
      NCCLCHECK(ncclCommFinalize(ncclComm_bigger));
      stop_timer(timing, nccl_finalize);
      start_timer(timing, nccl_destroy);
      NCCLCHECK(ncclCommDestroy(ncclComm_bigger));
      stop_timer(timing, nccl_destroy);
    }
  }

  if (color == 0) {
    if (args.aborts) {
      start_timer(timing, nccl_abort);
      NCCLCHECK(ncclCommAbort(ncclComm_smaller));
      stop_timer(timing, nccl_abort);
    } else {
      start_timer(timing, nccl_finalize);
      NCCLCHECK(ncclCommFinalize(ncclComm_smaller));
      stop_timer(timing, nccl_finalize);
      start_timer(timing, nccl_destroy);
      NCCLCHECK(ncclCommDestroy(ncclComm_smaller));
      stop_timer(timing, nccl_destroy);
    }
  } else {
    if (args.aborts) {
      skip_timer(timing, nccl_abort);
    } else {
      skip_timer(timing, nccl_finalize);
      skip_timer(timing, nccl_destroy);
    }
  }
}

void timing_set_comm(timer_record_t* timing, MPI_Comm comm) {
  /* this routine only exists to time things prior to MPI_Init, and then
  add the comm in later. */
  if (timing->comm != MPI_COMM_NULL) {
    printf("ERROR: Timing already has a communicator! %p\n", timing->comm);
    ABORT_EXIT;
  }
  timing->comm = comm;
  MPICHECK(MPI_Comm_rank(comm, &timing->rank));
  MPICHECK(MPI_Comm_size(comm, &timing->rank_count));
}

/*
 * Allocates and initializes a timer_record_t structure for performance
 * measurements.
 *
 * The structure tracks timing data across multiple iterations and ranks:
 * - Local arrays store per-iteration timing on each rank
 * - Reduced arrays (on rank 0) aggregate data across all ranks
 * - Type-sorted arrays accumulate statistics per timer type
 *
 * When done, free it with timing_free().
 *
 * Note: ALL ranks in the MPI Communicator _must_ call the same set of
 * stop_timer or skip_timer calls (with the same types), otherwise
 * inconsistent results and/or deadlock occurs.
 */
timer_record_t* timing_allocate(MPI_Comm comm, int flush) {
  timer_record_t* timing = (timer_record_t*)malloc(sizeof(timer_record_t));
  timing->flush_interval = flush;
  timing->currently_enabled = true;
  // Allocate locally-written arrays
  timing->local = (double*)malloc(sizeof(double) * timing->flush_interval);
  timing->local_sq = (double*)malloc(sizeof(double) * timing->flush_interval);
  timing->id = (int*)malloc(sizeof(int) * timing->flush_interval);
  timing->participated = (int*)malloc(sizeof(int) * timing->flush_interval);
  timing->comm = MPI_COMM_NULL;
  timing->count = 0;

  if (comm != MPI_COMM_NULL) {
    timing_set_comm(timing, comm);
  } else {
    timing->rank = 0;
    timing->rank_count = 1;
  }
  if (timing->rank == 0) {
    // Allocate root-only reduced arrays
    timing->max = (double*)malloc(sizeof(double) * timing->flush_interval);
    timing->sum = (double*)malloc(sizeof(double) * timing->flush_interval);
    timing->sumsq = (double*)malloc(sizeof(double) * timing->flush_interval);
    timing->participant_count = (int*)malloc(sizeof(int) * timing->flush_interval);
  } else {
    timing->max = NULL;
    timing->sum = NULL;
    timing->sumsq = NULL;
    timing->participant_count = NULL;
  }

  // Initialize type-sorted arrays to zero
  for (int i=0; i<timer_enum_count; i++) {
    timing->rounds_by_type[i] = 0;
    timing->sum_by_type[i] = 0;
    timing->sumsq_by_type[i] = 0;
    timing->max_by_type[i] = 0;
    timing->participant_count_by_type[i] = 0;
  }
  return timing;
}

/*
 * Prints summary statistics for multiple timing contexts (e.g., different communicator sizes).
 *
 * For each timer type and each timing context:
 * - Performs final gathering of timing data
 * - Computes average, max, and standard deviation across all samples
 * - Prints formatted summary line with statistics
 * - Can be filtered to show only NCCL operations with --nccl-only
 *
 * Output includes: communicator size, operation name, max/avg/stddev times,
 * number of collectives performed, and total samples collected.
 */
void timing_print_summary_grouped(timer_record_t** timings, int num_timings) {
  for (int k=0; k<num_timings; k++) {
    gather_timing(timings[k]);
  }
  for (int i=0; i<timer_enum_count; i++) {
    if (args.nccl_only && !timer_type_nccl(i)) continue;
    for (int k=0; k<num_timings; k++) {
      timer_record_t* timing = timings[k];
      if (timing->rank != 0) continue;
      int64_t rounds_counted = timing->rounds_by_type[i];
      if (rounds_counted <= 0) continue;
      double avg = (double)timing->sum_by_type[i] / timing->participant_count_by_type[i];
      double stddev;
      if (timing->participant_count_by_type[i] > 1) {
        double A = timing->sumsq_by_type[i] / timing->participant_count_by_type[i];
        double B = pow(timing->sum_by_type[i] / timing->participant_count_by_type[i], 2.0);
        stddev = sqrt(A - B);
      } else {
        stddev = 0;
      }
      double avg_ranks = timing->participant_count_by_type[i] / rounds_counted;
      printf("SUMMARY: Comm size %4d, %36s: Max %8.3f ms, Avg %8.3f ms, Stddev %8.3f ms, Collectives %3ld\n",
        timing->rank_count,
        timer_strings[i], timing->max_by_type[i]*1e-6, avg*1e-6, stddev*1e-6,
        rounds_counted);
    }
  }
  if (args.root_every > 0 && num_timings && timings[0]->rank == 0) {
    printf("Note: Scalable Comm Initialization was performed with a root for every %d ranks.\n",args.root_every);
  }
}

void timing_print_summary(timer_record_t* timing) {
  timing_print_summary_grouped(&timing, 1);
}

/* Deallocate and free resources associated with the timing record. */
void timing_free(timer_record_t** timing_ptr) {
  timer_record_t* timing = *timing_ptr;
  free(timing->local);
  free(timing->local_sq);
  free(timing->id);
  free(timing->participated);
  if (timing->max != NULL) free(timing->max);
  if (timing->sum != NULL) free(timing->sum);
  if (timing->sumsq != NULL) free(timing->sumsq);
  if (timing->participant_count != NULL) free(timing->participant_count);
  free(timing);
  *timing_ptr = NULL;
}

/*
 * Calculates the next rank count to test based on sweep parameters.
 * Supports both linear stepping (stepranks) and multiplicative stepping (stepfactor).
 * Ensures progress by adding/subtracting 1 if the calculated next value equals current.
 */
 int next_step(int current) {
  int next;
  if (args.sweep_kind == kind_fixed) {
    return 0;
  }
  if (args.sweep_kind == kind_factor) {
    next = (int)roundf(current * args.sweep_num);
  }
  if (args.sweep_kind == kind_step) {
    next = current + (int)roundf(args.sweep_num);
  }
  if (next == current) {
    next += (args.beginranks < args.endranks) ? 1 : -1;
  }
  return next;
}

/*
 * Determines whether to continue testing based on current rank count.
 * Handles both ascending (beginranks < endranks) and descending iteration.
 */
int keep_testing(int current) {
  if (args.beginranks == args.endranks) {
    return current == args.endranks;
  }
  if (args.beginranks < args.endranks) {
    return current <= args.endranks;
  } else {
    return current >= args.endranks;
  }
}

int main(int argc, char **argv) {
  timer_record_t *timing_prempi;
  int rank, nDevs;

  std::vector<timer_record_t*> all_timings;

  /* we want to time a few things before MPI_Init, but we use MPI routines in
    parse_args as well. so allocate a timing object with enough buffer space to
    hold our pre-MPI timing events, then we can add the communicator later. */
  timing_prempi = timing_allocate(MPI_COMM_NULL, 16);

  start_timer(timing_prempi, cuda_init);
  CUDACHECK(cudaGetDeviceCount(&nDevs));
  stop_timer(timing_prempi, cuda_init);

  start_timer(timing_prempi, mpi_init);
  MPICHECK(MPI_Init(&argc, &argv));
  stop_timer(timing_prempi, mpi_init);
  timing_set_comm(timing_prempi, MPI_COMM_WORLD);

  parse_args(argc, argv);

  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  // We have to set our device before NCCL init.  Device is constant for all
  // tests, based on rank in MPI_COMM_WORLD.
  CUDACHECK(cudaSetDevice(rank % nDevs));
  prepare_work(timing_prempi);

  all_timings.push_back(timing_prempi);

  int nrank_test = args.beginranks;
  while (keep_testing(nrank_test)) {
    MPI_Comm subcomm = MPI_COMM_NULL;
    timer_record_t *subtimer;
    int color = 0;
    if (rank >= nrank_test) color = MPI_UNDEFINED;
    MPICHECK(MPI_Comm_split(MPI_COMM_WORLD, color, rank, &subcomm));

    if (subcomm != MPI_COMM_NULL) {
      subtimer = timing_allocate(subcomm, args.timing_flush_interval);
      if (args.test_kind == test_init) {
        main_init(argc, argv, subtimer, subcomm);
      } else if (args.test_kind == test_split) {
        main_split(argc, argv, subtimer, subcomm);
      } else if (args.test_kind == test_shrink) {
        main_split(argc, argv, subtimer, subcomm);
      } else if (args.test_kind == test_grow) {
        main_grow(argc, argv, subtimer, subcomm);
      }
      all_timings.push_back(subtimer);
    }
    nrank_test = next_step(nrank_test);
  }

  cleanup_work();

  timing_print_summary_grouped(all_timings.data(), all_timings.size());

  for (size_t i=0; i<all_timings.size(); i++) {
    if (all_timings[i]->comm != MPI_COMM_NULL && all_timings[i]->comm != MPI_COMM_WORLD) {
      MPICHECK(MPI_Comm_free(&all_timings[i]->comm));
    }
    timing_free(&all_timings[i]);
  }
  all_timings.clear();

  MPICHECK(MPI_Finalize());
  return main_ret_val;
}

void print_help() {
  int rank;
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  if (rank == 0) {
    printf("Usage: comm_ops_perf <test> <test-options>\n");
    printf(" Tests: init, split, shrink, grow\n");
    printf("  init:   Time ncclCommInitRank\n");
    printf("  split:  Time ncclCommSplit\n");
    printf("  shrink: Time ncclCommShrink\n");
    printf("  grow:   Time ncclCommGrow\n");
    printf(" Test Options:\n");
    printf("    -a|--abort: Don't finalize, instead abort communicators.\n");
    printf("    -A|--all-timers: Show timing for All operations, not just NCCL.\n");
    printf("    -s|--sweep-comms (factor-N|step-N|fixed) how to sweep through comm size. (Default: factor-2)\n");
    printf("    -b|--sweep-begin <ranks>: Ranks count to start with. (Default: NRanks)\n");
    printf("    -e|--sweep-end <ranks>: Ranks count to end with. (Default: 1)\n");
    printf("    -F|--flush-interval <interval>: flush timing every <interval> timing calls.\n");
    printf("          Default to 1, which introduces barriers and with -v flag, prints events as they happen.\n");
    printf("          Larger numbers remove artificial barriers, and prints in batches.\n");
    printf("    -i|--iterations <iterations>: number of iterations to run after warmup. (Default: %d).\n", args.iterations);
    printf("    -r|--resize factor-N|step-N:  How much to resize for split/shrink/grow.  (Default: factor-2)\n");
    printf("    -R|--root-every <n>: Each <n> ranks gets a root in ncclCommInitRankScalable (Default to 0 => only 1 root).\n");
    printf("    -S|--share: configure communicator with splitShare=1 and shrinkShare=1 (Default to %s).\n", args.share ? "true" : "false");
    printf("    -v|--verbose: print verbose timing output (timeseries).\n");
    printf("    -w|--warmup <iterations>: Run <iterations> rounds before enabling timing. (Default to %d).\n", args.warmup);
    printf("Examples:\n");
    printf("  mpirun ./comm_ops_perf init\n");
    printf("  mpirun ./comm_ops_perf split --sweep-comms fixed\n");
    printf("  mpirun ./comm_ops_perf shrink\n");
    printf("  mpirun ./comm_ops_perf grow --resize factor-1.5\n");
    printf("For Detailed output: Add --all-timers and --verbose.\n");
    printf("Note begin/end ranks can be increasing or decreasing, and the comm sweep will be adjusted accordingly.\n");
    printf("\n");
  }
}

#endif
