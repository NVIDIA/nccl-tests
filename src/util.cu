/*************************************************************************
 * Copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
// This contains an utlities to handle output both to stdout and to
// json files.
//
// An ad-hoc, libc-based approach to writing json has been adopted to
// keep things simple and to avoid injecting a dependency on the
// library for an external JSON utility.
//
// However, this means that the code is a brittle to changes and care
// should be taken when adding/removing things. We also essentially
// give up when passed non-ASCII strings and non-printable characters
// except some of the usual ones.

#include "nccl.h"
#include "util.h"
#include <assert.h>
#include <errno.h>
#include <string>
#include <iomanip>

#define PRINT if (is_main_thread) printf

extern int nThreads;
extern int nGpus;
extern size_t minBytes;
extern size_t maxBytes;
extern size_t stepBytes;
extern size_t stepFactor;
extern int datacheck;
extern int warmup_iters;
extern int iters;
extern int agg_iters;
extern int parallel_init;
extern int blocking_coll;
extern int cudaGraphLaunches;

static FILE *json_report_fp;
static thread_local bool write_json;

#define JSON_FILE_VERSION 1

#define TIME_STRING_FORMAT "%Y-%m-%d %H:%M:%S"

typedef enum {
  JSON_NONE, // A pseudo-state meaning that the document is empty
  JSON_KEY,
  JSON_OBJECT_EMPTY,
  JSON_OBJECT_SOME,
  JSON_LIST_EMPTY,
  JSON_LIST_SOME,
} json_state_t;

// We use these statics to maintain a stack of states where we are writing.
// the init_json_output function gets this set up, and it's the finalize_json_output function's job to clean this up.
json_state_t *states = nullptr;
size_t state_cap = 0; // Allocated stack capacity
size_t state_n = 0;   // # of items in the stack.

// This tries to sanitize/quote a string from 'in' into 'out',
// assuming 'out' has length 'lim'.  We mainly quote ",/,\,\t,\n, and
// bail if we encounter non-printable stuff or non-ASCII stuff.
// 'in' should be null-terminated, of course.
//
// We return false if we were not able to copy all of 'in', either for
// length reasons or for unhandled characters.
static bool sanitizeJson(char out[], int lim, const char *in) {
  int c = 0;
  while(*in) {
    if(c+1 >= lim) {
      out[c] = 0;
      return false;
    }
    switch(*in) {
    case '"':
    case '\\':
    case '/':
    case '\t':
    case '\n':
      if(c + 2 > lim) {
        out[c] = 0;
        return false;
      }

      out[c++] = '\\';
      if(*in == '\n') {
        out[c++] = 'n';
      }
      else if( *in == '\t') {
        out[c++] = 't';
      }
      else {
        out[c++] = *in;
      }
      break;
    default:
      if (*in >= 0x7F || *in <= 0x1F) {
        out[c] = 0;
        return false;
      }
      out[c++] = *in;
      break;
    }
    ++in;
  }
  out[c] = 0;
  return true;
}

// Push state onto the state stack. Reallocate for extra storage if needed.
// Because JSON_NONE is a pseudo-state, don't allow it to be pushed.
static void jsonPushState(json_state_t state) {
  assert(state != JSON_NONE);
  if(state_cap <= (state_n+1)) {
    state_cap = max((size_t)16, state_cap*2);
    states = (json_state_t *)realloc(states, sizeof(json_state_t)*state_cap);
    assert(states);
  }
  states[state_n++] = state;
}

// Return the current state at the top of the stack
static json_state_t jsonCurrState() {
  if(state_n == 0) {
    return JSON_NONE;
  }
  return states[state_n-1];
}

// Replace the stack with state (equivalent to a pop & push if stack is not empty)
static void jsonReplaceState(json_state_t state) {
  assert(state != JSON_NONE);
  assert(state_n != 0);
  states[state_n-1] = state;
}

// Pop the top state off the stack, or return that the state is empty
static json_state_t jsonPopState() {
  if(state_n == 0) {
    return JSON_NONE;
  }
  return states[--state_n];
}

// Emit a key and separator. Santize the key.
// This is only acceptable if the top state is an object
// Emit a ',' separator of we aren't the first item.
static void jsonKey(const char *name) {
  switch(jsonCurrState()) {
  case JSON_OBJECT_EMPTY:
    jsonReplaceState(JSON_OBJECT_SOME);
    break;
  case JSON_OBJECT_SOME:
    fprintf(json_report_fp, ",");
    break;
  default:
    assert(0);
    break;
  }
  char tmp[2048];
  sanitizeJson(tmp, sizeof(tmp), name);
  fprintf(json_report_fp, "\"%s\":", tmp);
  jsonPushState(JSON_KEY);
}

// Helper function for inserting values.
// Only acceptable after keys, top-level, or in lists.
// Emit preceeding ',' if in a list and not first item.
static void jsonValHelper() {
  switch(jsonCurrState()) {
  case JSON_LIST_EMPTY:
    jsonReplaceState(JSON_LIST_SOME);
    break;
  case JSON_LIST_SOME:
    fprintf(json_report_fp, ",");
    break;
  case JSON_KEY:
    jsonPopState();
    break;
  case JSON_NONE:
    break;
  default:
    assert(0);
  }
}

// Start an object
static void jsonStartObject() {
  jsonValHelper();
  fprintf(json_report_fp, "{");
  jsonPushState(JSON_OBJECT_EMPTY);
}

// Close an object
static void jsonFinishObject() {
  switch(jsonPopState()) {
  case JSON_OBJECT_EMPTY:
  case JSON_OBJECT_SOME:
    break;
  default:
    assert(0);
  }
  fprintf(json_report_fp, "}");
}

// Start a list
static void jsonStartList() {
  jsonValHelper();
  fprintf(json_report_fp, "[");
  jsonPushState(JSON_LIST_EMPTY);
}

// Close a list
static void jsonFinishList() {
  switch(jsonPopState()) {
  case JSON_LIST_EMPTY:
  case JSON_LIST_SOME:
    break;
  default:
    assert(0);
  }
  fprintf(json_report_fp, "]");
}

// Write a null value
static void jsonNull() {
  jsonValHelper();
  fprintf(json_report_fp, "null");
}

// Write a (sanititzed) string
static void jsonStr(const char *str) {
  if(str == nullptr) {
    jsonNull();
    return;
  }
  jsonValHelper();
  char tmp[2048];
  sanitizeJson(tmp, sizeof(tmp), str);
  fprintf(json_report_fp, "\"%s\"", tmp);
}

// Write a bool as "true" or "false" strings.
static void jsonBool(bool val) {
  jsonStr(val ? "true" : "false");
}

// Write an integer value
static void jsonInt(const int val) {
  jsonValHelper();
  fprintf(json_report_fp, "%d", val);
}

// Write a size_t value
static void jsonSize_t(const size_t val) {
  jsonValHelper();
  fprintf(json_report_fp, "%zu", val);
}

// Write a double value
static void jsonDouble(const double val) {
  jsonValHelper();
  if(val != val) {
    fprintf(json_report_fp, "\"nan\"");
  }
  else {
    fprintf(json_report_fp, "%lf", val);
  }
}

// Fill buff with a formatted time string corresponding to 'now.
// Write len or fewer bytes.
void formatNow(char *buff, int len) {
  time_t now;
  time(&now);
  struct tm *timeinfo = localtime(&now);

  strftime(buff, len, TIME_STRING_FORMAT, timeinfo);
}

// We provide some status line to stdout.
// The JSON stream is left with a trailing comma and the top-level
// object open for the next set of top-level items (config and
// results).

// This uses unguarded 'printf' rather than the PRINT() macro because
// is_main_thread is not set up at this point.
void jsonOutputInit(const char *in_path,
                    int argc, char **argv,
                    char **envp) {
  if(in_path == nullptr) {
    return;
  }

  #ifdef MPI_SUPPORT
  int proc;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  if(proc != 0) {
    return;
  }
  #endif

  char *try_path = strdup(in_path);
  int try_count = 0;
  json_report_fp = fopen(try_path, "wx");
  while(json_report_fp == NULL) {
    if(errno != EEXIST) {
      printf("# skipping json output; %s not accessible\n", try_path);
      free(try_path);
      return;
    }
    free(try_path);
    if(asprintf(&try_path, "%s.%d", in_path, try_count++) == -1) {
      printf("# skipping json output; failed to probe destination\n");
      return;
    }
    json_report_fp = fopen(try_path, "wx");
  }

  printf("# Writing JSON output to %s\n", try_path);
  free(try_path);

  write_json = true;

  jsonStartObject(); // will be closed finalize_json_output

  jsonKey("version"); jsonInt(JSON_FILE_VERSION);

  jsonKey("start_time");
  {
    char timebuffer[128];
    formatNow(timebuffer, sizeof(timebuffer));
    jsonStr(timebuffer);
  }

  jsonKey("args");
  jsonStartList();
  for(int i = 0; i < argc; i++) {
    jsonStr(argv[i]);
  }
  jsonFinishList();

  jsonKey("env");
  jsonStartList();
  for(char **e = envp; *e; e++) {
    jsonStr(*e);
  }
  jsonFinishList();
  jsonKey("nccl_version"); jsonInt(test_ncclVersion);
}

void jsonIdentifyWriter(bool is_writer) {
  write_json &= is_writer;
}

// This cleans up the json output, finishing the object and closing the file.
// If we were not writing json output, we don't do anything.
void jsonOutputFinalize() {
  if(write_json) {

    jsonKey("end_time");
    char timebuffer[128];
    formatNow(timebuffer, sizeof(timebuffer));
    jsonStr(timebuffer);

    jsonFinishObject();

    assert(jsonCurrState() == JSON_NONE);
    free(states);
    states = nullptr;
    state_n = 0;
    state_cap = 0;

    fclose(json_report_fp);
    json_report_fp = nullptr;
  }
}

struct rankInfo_t {
  int rank;
  int group;
  int pid;
  char hostname[1024];
  int device;
  char device_hex[128];
  char devinfo[1024];
};

// Helper function to parse the device info lines passed via MPI to the root rank.
// This fills 'rank' with the parsed contents of 'instring'.
static int parseRankInfo(rankInfo_t *rank, const char *instring) {
  int end;
  sscanf(instring,
         "#  Rank %d Group %d Pid %d on %1024s device %d [%128[^]]] %1024[^\n]\n%n",
         &rank->rank,
         &rank->group,
         &rank->pid,
         rank->hostname,
         &rank->device,
         rank->device_hex,
         rank->devinfo,
         &end);
  return end;
}

static void jsonRankInfo(const rankInfo_t *ri) {
  jsonStartObject();
  jsonKey("rank");        jsonInt(ri->rank);
  jsonKey("group");       jsonInt(ri->group);
  jsonKey("pid");         jsonInt(ri->pid);
  jsonKey("hostname");    jsonStr(ri->hostname);
  jsonKey("device");      jsonInt(ri->device);
  jsonKey("device_hex");  jsonStr(ri->device_hex);
  jsonKey("device_info"); jsonStr(ri->devinfo);
  jsonFinishObject();
}

// Write the start of a benchmark output line containing the bytes &
// op type, both to stdout and to json if we are writing there.
void writeBenchmarkLinePreamble(size_t nBytes, size_t nElem, const char typeName[], const char opName[], int root) {
  char rootName[100];
  sprintf(rootName, "%6i", root);
  PRINT("%12li  %12li  %8s  %6s  %6s", nBytes, nElem, typeName, opName, rootName);

  if(write_json) {
    jsonStartObject();
    jsonKey("size");  jsonSize_t(nBytes);
    jsonKey("count"); jsonSize_t(nElem);
    jsonKey("type");  jsonStr(typeName);
    jsonKey("redop"); jsonStr(opName);
    jsonKey("root");  jsonStr(rootName);
  }
}

// Finish a result record we were writing to stdout/json
void writeBenchmarkLineTerminator(int actualIters, const char *name) {
  PRINT("\n");
  if(write_json) {
    jsonKey("actual_iterations"); jsonInt(actualIters);
    jsonKey("experiment_name");   jsonStr(name);
    jsonFinishObject();
  }
}

// Handle a cases where we don't write out of place results
void writeBenchMarkLineNullBody() {
  PRINT("                                ");  // only do in-place for trace replay
  if(write_json) {
    jsonKey("out_of_place"); jsonNull();
  }
}

void getFloatStr(double value, int width, char* str) {
  int power = 0;
  for (uint64_t val = 1; value >= val; val *= 10) power++;

  if (power < width-2) sprintf(str, "%*.2f", width, value);
  else if (power < width-1) sprintf(str, "%*.1f", width, value);
  else if (power < width+1) sprintf(str, "%*.0f", width, value);
  else if (width >= 7) sprintf(str, "%*.1e", width, value);
  else if (width >= 8) sprintf(str, "%*.2e", width, value);
  else sprintf(str, "%*.0e", width, value);
}

// Write the performance-related payload to stdout/json.
// We call this function twice at the top level per test: once for out-of-place, and once for in-place.
// The Json output assumes out-of-place happens first.
void writeBenchmarkLineBody(double timeUsec, double algBw, double busBw, bool reportErrors, int64_t wrongElts, bool report_cputime, bool report_timestamps, bool out_of_place) {
  char timeStr[8];
  getFloatStr(timeUsec, 7, timeStr);

  char algBwStr[7];
  getFloatStr(algBw, 6, algBwStr);

  char busBwStr[7];
  getFloatStr(busBw, 6, busBwStr);

  if (reportErrors) {
    PRINT("  %7s  %6s  %6s  %6g", timeStr, algBwStr, busBwStr, (double)wrongElts);
  } else {
    PRINT("  %7s  %6s  %6s    N/A", timeStr, algBwStr, busBwStr);
  }

  if (!out_of_place && report_timestamps) {
    char timebuffer[128];
    formatNow(timebuffer, sizeof(timebuffer));
    PRINT("%21s", timebuffer);
  }

  if(write_json) {
    jsonKey(out_of_place ? "out_of_place" : "in_place");
    jsonStartObject();
    jsonKey(report_cputime ? "cpu_time" : "time"); jsonDouble(timeUsec);
    jsonKey("alg_bw");                             jsonDouble(algBw);
    jsonKey("bus_bw");                             jsonDouble(busBw);
    jsonKey("nwrong");                             (reportErrors ? jsonDouble((double)wrongElts) : jsonNull());
    jsonFinishObject();
  }
}

// This writes out a report about the run parameters and devices
// involved to stdout and json.  For MPI, this will use a collective
// to gather from each rank to the root.

// Root then consumes this output, printing raw lines for stdout and
// parsing them for JSON for proper formatting.

// Perhaps actually sending records around instead of formatted
// strings would be smarter/easier, but I chose to adapt what was
// already in place.
testResult_t writeDeviceReport(size_t *maxMem, int localRank, int proc, int totalProcs, int color, const char hostname[], const char *program_name) {
  PRINT("# nccl-tests version %s nccl-headers=%d nccl-library=%d\n", NCCL_TESTS_VERSION, NCCL_VERSION_CODE, test_ncclVersion);
  PRINT("# Collective test starting: %s\n", program_name);
  PRINT("# nThread %d nGpus %d minBytes %ld maxBytes %ld step: %ld(%s) warmup iters: %d iters: %d agg iters: %d validation: %d graph: %d\n",
        nThreads, nGpus, minBytes, maxBytes,
        (stepFactor > 1)?stepFactor:stepBytes, (stepFactor > 1)?"factor":"bytes",
        warmup_iters, iters, agg_iters, datacheck, cudaGraphLaunches);
  if (blocking_coll) PRINT("# Blocking Enabled: wait for completion and barrier after each collective \n");
  if (parallel_init) PRINT("# Parallel Init Enabled: threads call into NcclInitRank concurrently \n");
  PRINT("#\n");

  if(write_json) {
    jsonKey("config");
    jsonStartObject();
    jsonKey("nthreads");      jsonInt(nThreads);
    jsonKey("ngpus");         jsonInt(nGpus);
    jsonKey("minimum_bytes"); jsonSize_t(minBytes);
    jsonKey("maximum_bytes"); jsonSize_t(maxBytes);
    if(stepFactor > 1) {
      jsonKey("step_factor");   jsonInt(stepFactor);
    }
    else {
      jsonKey("step_bytes");  jsonSize_t(stepBytes);
    }

    jsonKey("warmup_iters");          jsonInt(warmup_iters);
    jsonKey("iterations");            jsonInt(iters);
    jsonKey("aggregated_iterations"); jsonInt(agg_iters);
    jsonKey("validation");            jsonInt(datacheck);
    jsonKey("graph");                 jsonInt(cudaGraphLaunches);
    jsonKey("blocking_collectives");  jsonBool(blocking_coll);
    jsonKey("parallel_init");         jsonBool(parallel_init);
  }

  PRINT("# Using devices\n");
#define MAX_LINE 2048
  char line[MAX_LINE];
  int len = 0;
  const char* envstr = getenv("NCCL_TESTS_DEVICE");
  const int gpu0 = envstr ? atoi(envstr) : -1;
  int available_devices;
  CUDACHECK(cudaGetDeviceCount(&available_devices));
  for (int i=0; i<nThreads*nGpus; i++) {
    const int cudaDev = (gpu0 != -1 ? gpu0 : localRank*nThreads*nGpus) + i;
    const int rank = proc*nThreads*nGpus+i;
    cudaDeviceProp prop;
    if (cudaDev >= available_devices) {
      fprintf(stderr, "Invalid number of GPUs: %d requested but only %d were found.\n",
              (gpu0 != -1 ? gpu0 : localRank*nThreads*nGpus) + nThreads*nGpus, available_devices);
      fprintf(stderr, "Please check the number of processes and GPUs per process.\n");
      return testNotImplemented;
    }
    CUDACHECK(cudaGetDeviceProperties(&prop, cudaDev));
    if (len < MAX_LINE) {
      len += snprintf(line+len, MAX_LINE-len, "#  Rank %2d Group %2d Pid %6d on %10s device %2d [%04x:%02x:%02x] %s\n",
                      rank, color, getpid(), hostname, cudaDev, prop.pciDomainID, prop.pciBusID, prop.pciDeviceID, prop.name);
    }
    *maxMem = std::min(*maxMem, prop.totalGlobalMem);
  }
  if (len >= MAX_LINE) {
    strcpy(line+MAX_LINE-5, "...\n");
  }

#if MPI_SUPPORT
  char *lines = (proc == 0) ? (char *)malloc(totalProcs*MAX_LINE) : NULL;
  // Gather all output in rank order to root (0)
  MPI_Gather(line, MAX_LINE, MPI_BYTE, lines, MAX_LINE, MPI_BYTE, 0, MPI_COMM_WORLD);
  if (proc == 0) {
    if(write_json) {
      jsonKey("devices");
      jsonStartList();
    }
    for (int p = 0; p < totalProcs; p++) {
      PRINT("%s", lines+MAX_LINE*p);
      if(write_json) {
        rankInfo_t rankinfo;
        parseRankInfo(&rankinfo, lines + MAX_LINE*p);
        jsonRankInfo(&rankinfo);
      }
    }
    if(write_json) {
      jsonFinishList();
    }
    free(lines);
  }
  MPI_Allreduce(MPI_IN_PLACE, maxMem, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
#else
  PRINT("%s", line);
  if(write_json) {
    rankInfo_t rankinfo;
    parseRankInfo(&rankinfo, line);
    jsonKey("devices");
    jsonStartList();
    jsonRankInfo(&rankinfo);
    jsonFinishList();
  }
#endif
  if(write_json) {
    jsonFinishObject();
  }

  return testSuccess;
}

// Write a result header to stdout/json.
// Json results object and contained table list are left open
void writeResultHeader(bool report_cputime, bool report_timestamps) {
  const char* tsLbl  = report_timestamps ? "timestamp" : "";
  const int tsPad = report_timestamps ? 19 : 0;
  const char* tsFmt = report_timestamps ? TIME_STRING_FORMAT : "";
  const char* timeStr = report_cputime ? "cputime" : "time";
  PRINT("#\n");
  PRINT("# %10s  %12s  %8s  %6s  %6s           out-of-place                       in-place          \n", "", "", "", "", "");
  PRINT("# %10s  %12s  %8s  %6s  %6s  %7s  %6s  %6s  %6s  %7s  %6s  %6s  %6s %*s\n", "size", "count", "type", "redop", "root",
        timeStr, "algbw", "busbw", "#wrong", timeStr, "algbw", "busbw", "#wrong", tsPad, tsLbl);
  PRINT("# %10s  %12s  %8s  %6s  %6s  %7s  %6s  %6s  %6s  %7s  %6s  %6s  %6s %*s\n", "(B)", "(elements)", "", "", "",
        "(us)", "(GB/s)", "(GB/s)", "", "(us)", "(GB/s)", "(GB/s)", "", tsPad, tsFmt);

  if(write_json) {
    jsonKey("results"); jsonStartList();
  }
}

// Write the footer for results to stdout/json.
// We close the table list and write out the summary items.
// Results object is left open for errors.
void writeResultFooter(const int errors[], const double bw[], double check_avg_bw, const char *program_name) {

  if(write_json) {
    jsonFinishList();
  }

  PRINT("# %-20s : %d %s\n", "Out of bounds values", errors[0], errors[0] ? "FAILED" : "OK");
  PRINT("# %-20s : %g %s\n", "Avg bus bandwidth", bw[0], check_avg_bw == -1 ? "" : (bw[0] < check_avg_bw*(0.9) ? "FAILED" : "OK"));
  PRINT("#\n");
  PRINT("# Collective test concluded: %s\n", program_name);

  if(write_json) {
    jsonKey("out_of_bounds");
    jsonStartObject();
    jsonKey("count");      jsonInt(errors[0]);
    jsonKey("okay");       jsonBool(errors[0] == 0);
    jsonFinishObject();
    jsonKey("average_bus_bandwidth");
    jsonStartObject();
    jsonKey("bandwidth"); jsonDouble(bw[0]);
    jsonKey("okay");       check_avg_bw == -1 ? jsonStr("unchecked") : jsonBool(bw[0] >= check_avg_bw*(0.9));
    jsonFinishObject();
  }
}

std::string getMemString(double amount) {
  std::string postfix = " B";
  if (abs(amount) >= 1024.0*1024.0*1024.0) {
    postfix = " GB";
    amount /= 1024.0 * 1024.0 * 1024.0;
  } else if (abs(amount) >= 1024.0*1024.0) {
    postfix = " MB";
    amount /= 1024.0 * 1024.0;
  } else if (abs(amount) >= 1024.0) {
    postfix = " KB";
    amount /= 1024.0;
  }
  int precision = 0;
  if (abs(amount) < 10.0) {
    precision = 2;
  } else if (abs(amount) < 100.0) {
    precision = 1;
  }
  std::stringstream ss;
  ss << std::fixed << std::setprecision(precision) << amount << postfix;
  return ss.str();
}

void writeMemInfo(memInfo_t* memInfos, int numMemInfos) {

  std::stringstream ss;
  uint64_t maxAmount = 0;
  for (int i = 0; i < numMemInfos; i++) {
    ss << memInfos[i].name << " "
      << getMemString(memInfos[i].amount)
      << " ";
    if (i < numMemInfos - 1) {
      ss << "| ";
    }
    maxAmount += memInfos[i].amount;
  }
  ss << "| Total  " << getMemString(maxAmount);
  PRINT("# %-20s : %s\n", "GPU memory usage", ss.str().c_str());
}

// Write out remaining errors to stdout/json.
void writeErrors() {
  const char *error = ncclGetLastError(NULL);
  if(error && strlen(error) > 0) {
    PRINT("# error: %s\n", error);
  } else {
    PRINT("\n");
  }
  if(write_json) {
    jsonKey("errors");
    jsonStartList();
    if(error) {
      jsonStr(error);
    }
    jsonFinishList();
  }
}

void finalizeFooter() {
  PRINT("#\n");
}
