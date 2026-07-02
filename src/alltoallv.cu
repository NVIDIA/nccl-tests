/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * AlltoAllv Performance Test
 * ==========================
 *
 * This test benchmarks irregular alltoall (alltoallv) communication patterns,
 * where each rank sends different amounts of data to different peers.
 *
 * USAGE
 * -----
 *
 * The test integrates with the standard NCCL perf test harness and accepts
 * all standard perf test command-line arguments.
 *
 * The test supports two modes for specifying traffic patterns:
 *
 * 1. Generated Pattern Mode (default)
 *    When no traffic matrix file is provided, the test automatically generates
 *    a distance-weighted distribution where more distant ranks (in rank ID
 *    space) receive proportionally more traffic than nearby ranks. While
 *    per-peer distribution is imbalanced (different amounts to different peers),
 *    each rank sends and receives the same total amount of data.
 *
 *    Environment variable settings:
 *    - NCCL_TESTS_ALLTOALLV_SPREAD: Controls the distance-weighted spread factor.
 *      Range: 0.0 to 1.0. Default: 1.0
 *        * 0.0 = uniform distribution (equal traffic to all peers)
 *        * 1.0 = fully distance-weighted (more traffic to distant ranks)
 *        * Intermediate values blend uniform and distance-weighted distributions
 *
 *    Impact of spread on traffic distribution (example: rank 0 sending to 4 ranks):

 *      spread=0.0 (uniform):       spread=1.0 (distance-weighted):
 *      rank 0 -> rank 0: ###       rank 0 -> rank 0: (none)
 *      rank 0 -> rank 1: ###       rank 0 -> rank 1: ##
 *      rank 0 -> rank 2: ###       rank 0 -> rank 2: ####
 *      rank 0 -> rank 3: ###       rank 0 -> rank 3: ######
 *
 *    Data size sweeps:
 *    The -b/-e/-f parameters control the sweep bounds and step size. Each
 *    message size in the sweep represents the total amount each rank sends to
 *    all peers combined, with the spread factor controlling how this total is
 *    distributed among peers.
 *
 * 2. Matrix File Mode
 *    When NCCL_TESTS_ALLTOALLV_MATRIX_FILE is provided, the test reads an explicit
 *    traffic matrix file that defines the exact byte counts for each
 *    rank-to-rank communication.
 *
 *    Environment variable settings:
 *    - NCCL_TESTS_ALLTOALLV_MATRIX_FILE: Path to a whitespace-separated matrix file.
 *      Required for matrix mode. The file must contain a square matrix where
 *      cell [i][j] represents bytes sent from rank i to rank j.
 *
 *    - NCCL_TESTS_ALLTOALLV_MATRIX_SCALE: Scale factor applied to all matrix values.
 *      Default: 1.0 (no scaling). Useful for testing different traffic
 *      volumes with the same pattern.
 *
 *    Matrix file format:
 *    - Whitespace-separated numeric values, interpreted as bytes
 *    - Must be square (rows == cols)
 *    - Matrix dimension must be >= number of ranks
 *    - Only the first nranks rows/cols are used if matrix is larger
 *    - All values are aligned to 16-byte boundaries automatically
 *
 *    Example matrix file (3 ranks):
 *      100  200  150
 *      50   100  75
 *      300  150  200
 *
 *    IMPORTANT caveats for matrix mode:
 *    1. -b and -e must be equal. Sweeps are not supported since per-peer sizes
 *       come directly from the matrix file, not from the -b/-e parameters.
 *    2. -e must be at least the maximum total send or receive bytes across all
 *       ranks in the matrix. The test harness allocates buffers for all ranks
 *       based on the -e value, so it must accommodate the largest workload.
 *
 * OUTPUT AND ANALYSIS
 * -------------------
 *
 * The test outputs standard NCCL perf test metrics (algorithmic and bus
 * bandwidth), but these are based on the rank with the most send/recv bytes
 * and the maximum time across ranks. This approach is most representative of
 * collective performance in imbalanced workloads. Use the -a 3 option to report
 * max time across ranks rather than the default average time, which can be skewed
 * by inactive ranks.
 *
 * Optional per-rank bandwidth summary:
 *   Set NCCL_TESTS_ALLTOALLV_PRINT_SUMMARY=1 to print detailed per-rank statistics
 *   including send_bytes, recv_bytes, alg_bw, and bus_bw for each rank.
 *   Useful for analyzing imbalanced workloads.
 *
 * EXAMPLE RUNS
 * ------------
 *
 * Generated pattern with distance-weighted distribution (default):
 *   alltoallv_perf -b 1M -e 32M -f 2
 *   Tests imbalanced alltoall where distant ranks receive more traffic.
 *
 * Generated pattern with uniform distribution:
 *   NCCL_TESTS_ALLTOALLV_SPREAD=0.0 alltoallv_perf -b 1M -e 32M -f 2
 *   Tests balanced alltoall where all peers receive equal traffic.
 *
 * Matrix file mode with explicit traffic pattern:
 *   NCCL_TESTS_ALLTOALLV_MATRIX_FILE=traffic.txt alltoallv_perf -b 32M -e 32M -f 2
 *   Tests arbitrary communication pattern defined in traffic.txt.
 * 
 */

#include "cuda_runtime.h"
#include "common.h"
#include <stdio.h>
#include <ctype.h>
#include <errno.h>
#include <atomic>
#include <mutex>
#include <stdint.h>
#include <stdlib.h>

#define PRINT if (is_main_thread) printf

#define MAX(a,b) ((a) > (b) ? (a) : (b))

static int alltoallv_use_generated_pattern = 0;
static int alltoallv_print_summary = 0;
static char* traffic_matrix_file = NULL;
static size_t* traffic_matrix_data = NULL;
static int traffic_matrix_dim = 0;
static double traffic_matrix_scale = 1.0; // scale factor applied to traffic matrix values
static double distance_weighted_spread = 1.0; // 0.0 => uniform, 1.0 => fully distance-weighted (default)

static std::mutex alltoallv_lock;
static int alltoallv_inited = 0;

// atomics used for error flag, since init can be multi-threaded
static std::atomic<int> alltoallv_error_set{0};
static inline int AlltoAllvHasError() {
  return alltoallv_error_set.load(std::memory_order_relaxed);
}
static inline void AlltoAllvSetError() {
  alltoallv_error_set.store(1, std::memory_order_relaxed);
}

static void AlltoAllvParseEnv() {
  const char* matrix_file_env = getenv("NCCL_TESTS_ALLTOALLV_MATRIX_FILE");
  if (matrix_file_env) traffic_matrix_file = (char*)matrix_file_env;
  const char* print_summary_env = getenv("NCCL_TESTS_ALLTOALLV_PRINT_SUMMARY");
  if (print_summary_env) alltoallv_print_summary = atoi(print_summary_env) != 0;
  const char* spread_env = getenv("NCCL_TESTS_ALLTOALLV_SPREAD");
  if (spread_env) {
    errno = 0;
    char* end = NULL;
    double v = strtod(spread_env, &end);
    if (errno != ERANGE && end != spread_env && *end == '\0') distance_weighted_spread = v;
  }
  if (distance_weighted_spread < 0.0) distance_weighted_spread = 0.0;
  if (distance_weighted_spread > 1.0) distance_weighted_spread = 1.0;
  const char* scale_env = getenv("NCCL_TESTS_ALLTOALLV_MATRIX_SCALE");
  if (scale_env) {
    errno = 0;
    char* end = NULL;
    double v = strtod(scale_env, &end);
    if (errno != ERANGE && end != scale_env && *end == '\0') traffic_matrix_scale = v;
  }
  if (traffic_matrix_scale < 0.0) traffic_matrix_scale = 0.0;
}

// read traffic matrix from file, return 0 on success, 1 on error
static int AlltoAllvReadTrafficMatrix() {
  FILE* f = NULL;
  char* buf = NULL;
  size_t* tmp_matrix_data = NULL;
  size_t fileLen = 0;
  char* p = NULL;
  char* end = NULL;
  double v = 0.0;
  long double scaled = 0.0;
  size_t rows = 0, cols = 0, tmp_cols = 0;
  size_t n_elts = 0;
  size_t k = 0;
  long pos = 0;
  int rc = 1;

  f = fopen(traffic_matrix_file, "rb");
  if (!f) {
    PRINT("Unable to open alltoallv traffic matrix file %s.\n", traffic_matrix_file);
    goto exit;
  }

  if (fseek(f, 0, SEEK_END) != 0) {
    PRINT("Unable to seek alltoallv traffic matrix file %s.\n", traffic_matrix_file);
    goto exit;
  }
  pos = ftell(f);
  if (pos < 0) {
    PRINT("Unable to tell alltoallv traffic matrix file size %s.\n", traffic_matrix_file);
    goto exit;
  }
  fileLen = (size_t)pos;
  if (fseek(f, 0, SEEK_SET) != 0) {
    PRINT("Unable to rewind alltoallv traffic matrix file %s.\n", traffic_matrix_file);
    goto exit;
  }

  buf = (char*)malloc(fileLen + 1);
  if (!buf) {
    goto exit;
  }
  if (fread(buf, 1, fileLen, f) != fileLen) {
    PRINT("Unable to read alltoallv traffic matrix file %s.\n", traffic_matrix_file);
    goto exit;
  }
  buf[fileLen] = '\0';

  // ---------- First pass: count rows/cols ----------
  p = buf;
  while (*p) {
    if (isspace((unsigned char)*p)) {
      if (*p == '\n' && tmp_cols > 0) {
        if (rows == 0) cols = tmp_cols;
        else if (tmp_cols != cols) {
          PRINT("Invalid alltoallv traffic matrix column count in file %s row %zu. Expected %zu columns, got %zu.\n", traffic_matrix_file, rows, cols, tmp_cols);
          goto exit;
        }
        tmp_cols = 0;
        rows++;
      }
      p++;
      continue;
    }

    errno = 0;
    end = NULL;
    v = strtod(p, &end);
    if (end == p || errno == ERANGE) {
      PRINT("Invalid alltoallv traffic matrix value at row %zu col %zu in file %s.\n", rows, tmp_cols, traffic_matrix_file);
      goto exit;
    }
    if (*end && !isspace((unsigned char)*end)) {
      PRINT("Invalid alltoallv traffic matrix value at row %zu col %zu in file %s.\n", rows, tmp_cols, traffic_matrix_file);
      goto exit;
    }
    if (v < 0.0) {
      PRINT("Invalid alltoallv traffic matrix value at row %zu col %zu in file %s. Value must be non-negative.\n", rows, tmp_cols, traffic_matrix_file);
      goto exit;
    }
    scaled = (long double)v * (long double)traffic_matrix_scale;
    if (scaled > (long double)SIZE_MAX) {
      PRINT("Invalid alltoallv traffic matrix value at row %zu col %zu in file %s. Value overflow (value=%g scale=%g).\n",
            rows, tmp_cols, traffic_matrix_file, v, traffic_matrix_scale);
      goto exit;
    }

    tmp_cols++;
    p = end;
  }
  if (tmp_cols > 0) {
    if (rows == 0) cols = tmp_cols;
    else if (tmp_cols != cols) {
      PRINT("Invalid alltoallv traffic matrix column count in file %s row %zu. Expected %zu columns, got %zu.\n", traffic_matrix_file, rows, cols, tmp_cols);
      goto exit;
    }
    rows++;
  }

  if (rows == 0) {
    PRINT("Invalid alltoallv traffic matrix, file %s is empty.\n", traffic_matrix_file);
    goto exit;
  }
  if (rows != cols) {
    PRINT("Invalid alltoallv traffic matrix, file %s must be square (got %zux%zu).\n", traffic_matrix_file, rows, cols);
    goto exit;
  }
  n_elts = rows * cols;

  tmp_matrix_data = (size_t*)malloc(n_elts * sizeof(*tmp_matrix_data));
  if (!tmp_matrix_data) {
    goto exit;
  }

  // ---------- Second pass: parse values ----------
  p = buf;
  k = 0;
  while (*p) {
    if (isspace((unsigned char)*p)) {
      p++;
      continue;
    }
    if (k >= n_elts) {
      PRINT("Invalid alltoallv traffic matrix, file %s has too many values.\n", traffic_matrix_file);
      goto exit;
    }

    end = NULL;
    v = strtod(p, &end);
    scaled = (long double)v * (long double)traffic_matrix_scale;
    if (end == p) {
      PRINT("Invalid alltoallv traffic matrix value at idx=%zu in file %s.\n", k, traffic_matrix_file);
      goto exit;
    }

    // align per-peer traffic to 16-byte boundary for consistency with other perf tests
    tmp_matrix_data[k++] = ((size_t)scaled) & ~(size_t)15;
    p = end;
  }
  if (k != n_elts) {
    PRINT("Invalid alltoallv traffic matrix, file %s has too few values (got %zu, expected %zu).\n",
          traffic_matrix_file, k, n_elts);
    goto exit;
  }

  // success
  traffic_matrix_dim = (int)rows;
  traffic_matrix_data = tmp_matrix_data;
  tmp_matrix_data = NULL; // don't free on success, transfer ownership to global variable
  rc = 0;

exit:
  if (f) fclose(f);
  free(buf);
  free(tmp_matrix_data);
  return rc;
}

static void AlltoAllvInit(int nranks) {
  std::lock_guard<std::mutex> lock(alltoallv_lock);

  if (!alltoallv_inited) {
    AlltoAllvParseEnv();
    if (traffic_matrix_file) {
      alltoallv_use_generated_pattern = 0;
      if (AlltoAllvReadTrafficMatrix()) AlltoAllvSetError();
    } else {
      alltoallv_use_generated_pattern = 1;
    }
    alltoallv_inited = 1;
  }

  // If matrix mode is active and initialization succeeded, ensure this comm size fits.
  if (traffic_matrix_file && !AlltoAllvHasError() && nranks > traffic_matrix_dim) {
    PRINT("Invalid alltoallv traffic matrix, requested nranks (%d) is larger than matrix size (%d) in file %s.\n",
          nranks, traffic_matrix_dim, traffic_matrix_file);
    AlltoAllvSetError();
  }

}

// get peer-to-peer bytes from traffic matrix
static inline size_t AlltoAllvGetPeerBytesFromMatrix(int i, int j) {
  return traffic_matrix_data[i * traffic_matrix_dim + j];
}

// get peer-to-peer bytes using distance-weighted formula
static inline size_t AlltoAllvGetPeerBytesDistanceWeighted(int i, int j, int nranks, size_t total_bytes) {
  if (nranks == 1) return total_bytes & ~(size_t)15;
  int distance = (j - i + nranks) % nranks;
  size_t sum_distances = (size_t)nranks * (nranks - 1) / 2;
  double uniform = (double)total_bytes / (double)nranks;
  double dist = (double)distance * (double)total_bytes / (double)sum_distances;
  double bytes = (1.0 - distance_weighted_spread) * uniform + distance_weighted_spread * dist;
  // align to per-peer chunk size to 16-byte boundary as in other tests
  return (size_t)bytes & ~(size_t)15;
}

static void AlltoAllvComputeMaxCounts(size_t *maxSendCount, size_t *maxRecvCount, size_t eltSize, int nranks, size_t total_bytes) {
  if (alltoallv_use_generated_pattern) {
    // Generated patterns are symmetric & balanced: compute one rank's total (O(N))
    size_t rank_total = 0;
    for (int j = 0; j < nranks; j++) {
      rank_total += AlltoAllvGetPeerBytesDistanceWeighted(0, j, nranks, total_bytes) / eltSize;
    }
    *maxSendCount = *maxRecvCount = rank_total;
  } else {
    // File-based patterns: scan all ranks for max (O(N²))
    size_t max_s = 0, max_r = 0;
    for (int i = 0; i < nranks; i++) {
      size_t rank_s = 0, rank_r = 0;
      for (int j = 0; j < nranks; j++) {
        rank_s += AlltoAllvGetPeerBytesFromMatrix(i, j) / eltSize;
        rank_r += AlltoAllvGetPeerBytesFromMatrix(j, i) / eltSize;
      }
      max_s = MAX(max_s, rank_s);
      max_r = MAX(max_r, rank_r);
    }
    *maxSendCount = max_s;
    *maxRecvCount = max_r;
  }
}

void AlltoAllvGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  *paramcount = alltoallv_use_generated_pattern ? (count / nranks) & ~(16/eltSize - 1) : 0;
  size_t total_bytes = (*paramcount) * nranks * eltSize;
  AlltoAllvComputeMaxCounts(sendcount, recvcount, eltSize, nranks, total_bytes);
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
}

testResult_t AlltoAllvInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount;
  size_t recvcount;
  size_t elt_size = wordSize(type);
  int nranks, rank;
  void* data;

  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    NCCLCHECK(ncclCommUserRank(args->comms[i], &rank));
    NCCLCHECK(ncclCommCount(args->comms[i], &nranks));
    // args->nbytes is per-peer bytes (from paramcount), so we reconstruct total_bytes by multiplying by nranks.
    // in generated_mode, this total_bytes is needed to calculate the distance-weighted pattern.
    size_t total_bytes = alltoallv_use_generated_pattern ? args->nbytes * nranks : 0;

    // zero out full recv and expected buffers, as expectedBytes is based on max across
    // ranks (not this rank's exact value) and unused tails must ultimately compare equal
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    CUDACHECK(cudaMemset(args->expected[i], 0, args->expectedBytes));

    // prepare this rank's send buffer
    sendcount = 0;
    for (int dst = 0; dst < nranks; dst++) {
      size_t peer_bytes = alltoallv_use_generated_pattern ?
        AlltoAllvGetPeerBytesDistanceWeighted(rank, dst, nranks, total_bytes) :
        AlltoAllvGetPeerBytesFromMatrix(rank, dst);
      sendcount += peer_bytes / elt_size;
    }
    if (sendcount > 0) {
      data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
      TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, 33*rep + rank, 1, 0));
    }

    // prepare this rank's expected buffer, which is a concatenation of variable-length
    // segments received from each source in rank order
    size_t recvoffset = 0;
    for (int src = 0; src < nranks; src++) {
      size_t peer_bytes = alltoallv_use_generated_pattern ?
        AlltoAllvGetPeerBytesDistanceWeighted(src, rank, nranks, total_bytes) :
        AlltoAllvGetPeerBytesFromMatrix(src, rank);
      recvcount = peer_bytes / elt_size;
      if (recvcount == 0) continue;
      // offset within the source's send stream where my segment begins (in elements)
      size_t src_offset = 0;
      for (int k = 0; k < rank; k++) {
        size_t prior_bytes = alltoallv_use_generated_pattern ?
          AlltoAllvGetPeerBytesDistanceWeighted(src, k, nranks, total_bytes) :
          AlltoAllvGetPeerBytesFromMatrix(src, k);
        src_offset += prior_bytes / elt_size;
      }
      TESTCHECK(InitData((char*)args->expected[i] + recvoffset, recvcount, src_offset, type, ncclSum, 33*rep + src, 1, 0));
      recvoffset += recvcount * elt_size;
    }
    CUDACHECK(cudaDeviceSynchronize());
  }

  // We don't support in-place alltoallv
  args->reportErrors = in_place ? 0 : 1;
  return testSuccess;
}

static void AlltoAllvPrintRankBandwidths(int nranks, size_t total_bytes, double sec) {
  if (alltoallv_use_generated_pattern) {
    printf("# NCCL_TESTS_ALLTOALLV_PRINT_SUMMARY: mode=generated nranks=%d total_bytes=%zu spread=%.3f time_usec=%.2f\n",
           nranks, total_bytes, distance_weighted_spread, sec * 1.0E6);
  } else {
    printf("# NCCL_TESTS_ALLTOALLV_PRINT_SUMMARY: mode=matrix nranks=%d matrix_dim=%d scale=%.3f time_usec=%.2f\n",
           nranks, traffic_matrix_dim, traffic_matrix_scale, sec * 1.0E6);
  }

  for (int rank = 0; rank < nranks; rank++) {
    size_t rank_send_bytes = 0, rank_recv_bytes = 0;
    for (int peer = 0; peer < nranks; peer++) {
      if (alltoallv_use_generated_pattern) {
        rank_send_bytes += AlltoAllvGetPeerBytesDistanceWeighted(rank, peer, nranks, total_bytes);
        rank_recv_bytes += AlltoAllvGetPeerBytesDistanceWeighted(peer, rank, nranks, total_bytes);
      } else {
        rank_send_bytes += AlltoAllvGetPeerBytesFromMatrix(rank, peer);
        rank_recv_bytes += AlltoAllvGetPeerBytesFromMatrix(peer, rank);
      }
    }

    size_t rank_max_bytes = MAX(rank_send_bytes, rank_recv_bytes);
    double rank_alg_bw = (double)rank_max_bytes / 1.0E9 / sec;
    double factor = ((double)(nranks-1))/((double)(nranks));
    double rank_bus_bw = rank_alg_bw * factor;

    printf("#   rank=%d send_bytes=%zu recv_bytes=%zu alg_bw=%.2f bus_bw=%.2f\n",
          rank, rank_send_bytes, rank_recv_bytes, rank_alg_bw, rank_bus_bw);
  }
}

void AlltoAllvGetBw(size_t count, size_t typesize, double sec, double* algBw, double* busBw, int nranks) {
  size_t total_bytes = count * nranks * typesize;
  size_t max_sendcount, max_recvcount;
  AlltoAllvComputeMaxCounts(&max_sendcount, &max_recvcount, typesize, nranks, total_bytes);
  size_t max_bytes = MAX(max_sendcount, max_recvcount) * typesize;
  double baseBw = (double)max_bytes / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks-1))/((double)(nranks));
  *busBw = baseBw * factor;

  if (alltoallv_print_summary && is_main_thread) {
    // print per-rank bandwidth summary (rank 0 computes and prints all ranks)
    AlltoAllvPrintRankBandwidths(nranks, total_bytes, sec);
  }
}

testResult_t AlltoAllvRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl) {
  if (deviceImpl == 0) {
    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,7,0)
    int nRanks, ncclRank;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    NCCLCHECK(ncclCommUserRank(comm, &ncclRank));
    NCCLCHECK(ncclGroupStart());

    // 'count' is per-peer elements (from paramcount) -- for generated patterns, we reconstruct total_bytes
    // by multiplying by nranks and use this to compute the distance-weighted pattern.
    // for matrix patterns, 'count' is not used (sizes come directly from the matrix).
    size_t elt_size = wordSize(type);
    size_t total_bytes = count * nRanks * elt_size;
    for (int r=0; r<nRanks; r++) {
      size_t send_bytes, recv_bytes;
      if (alltoallv_use_generated_pattern) {
        send_bytes = AlltoAllvGetPeerBytesDistanceWeighted(ncclRank, r, nRanks, total_bytes);
        recv_bytes = AlltoAllvGetPeerBytesDistanceWeighted(r, ncclRank, nRanks, total_bytes);
      } else {
        send_bytes = AlltoAllvGetPeerBytesFromMatrix(ncclRank, r);
        recv_bytes = AlltoAllvGetPeerBytesFromMatrix(r, ncclRank);
      }

      size_t send_count = send_bytes / elt_size;
      size_t recv_count = recv_bytes / elt_size;
      if (send_count > 0) {
        NCCLCHECK(ncclSend(sptr, send_count, type, r, comm, stream));
        sptr += send_count * elt_size;
      }
      if (recv_count > 0) {
        NCCLCHECK(ncclRecv(rptr, recv_count, type, r, comm, stream));
        rptr += recv_count * elt_size;
      }
    }
    NCCLCHECK(ncclGroupEnd());
#else
  PRINT("NCCL 2.7 or later is needed for alltoallv. This test was compiled with %d.%d.\n", NCCL_MAJOR, NCCL_MINOR);
  return testNcclError;
#endif
  } else {
    return testNotImplemented;
  }
  return testSuccess;
}

struct testColl alltoAllvTest = {
  "AlltoAllv",
  AlltoAllvGetCollByteCount,
  AlltoAllvInitData,
  AlltoAllvGetBw,
  AlltoAllvRunColl
};

void AlltoAllvGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  *sendcount = *recvcount = 0;

  AlltoAllvInit(nranks);
  if (AlltoAllvHasError()) return;

  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AlltoAllvGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);

  if (!alltoallv_use_generated_pattern) {
    // if using traffic matrix, validate that the max buffer size is large enough
    size_t total_bytes_req = MAX(*sendcount, *recvcount);
    if (count < total_bytes_req) {
      std::lock_guard<std::mutex> lock(alltoallv_lock);
      if (!AlltoAllvHasError()) {
        if (is_main_proc)
          printf("maxBytes (-e) must be at least %zu bytes as required by traffic matrix file %s (got %zu). Increase -e.\n",
                 total_bytes_req, traffic_matrix_file, count);
        AlltoAllvSetError();
      }
      *sendcount = *recvcount = 0;
      return;
    }
  }
}

testResult_t AlltoAllvRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &alltoAllvTest;
  ncclDataType_t *run_types;
  const char **run_typenames;
  int type_count;

  if (!alltoallv_use_generated_pattern) {
    if (AlltoAllvHasError()) {
      // this catches any potential errors that occurred earlier while loading the traffic matrix
      return testInternalError;
    }
    if (args->minbytes != args->maxbytes) {
      PRINT("-b and -e options must be set to the same value (e.g., -b 32M -e 32M) when using alltoallv traffic matrix mode.\n");
      return testInternalError;
    }
  }

  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = test_typenum;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  for (int i=0; i<type_count; i++) {
    TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "none", -1));
  }
  return testSuccess;
}

NCCL_WEAK struct testEngine ncclTestEngine = {
  /* .getBuffSize = */ AlltoAllvGetBuffSize,
  /* .runTest = */ AlltoAllvRunTest
};
