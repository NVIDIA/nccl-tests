/*************************************************************************
 * Copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef __UTIL_H__
#define __UTIL_H__

#include "common.h"

struct memInfo_t {
  int64_t amount;
  const char* name;
};

// Try to set up JSON file output. If MPI is used, only rank 0 will proceed.
// This should be called by only a single thread.
// If 'in_path' is NULL, we stop.
// Otherwise, we borrow 'in_path' and try to open it as a new file.
// If it already exists, we probe for new files by appending integers
// until we succeed.
// Then we write argv and envp to the json output, santizing them. We also
// write the nccl version.
// The top-level object remains open for the rest of the output.
void jsonOutputInit(const char *path, int argc, char **argv, char **envp);

// Should be called to identify main thread after threads are started to ensure we don't duplicate output
void jsonIdentifyWriter(bool is_writer);

// Write end time and close top-level object. Reset json state and close output file.
void jsonOutputFinalize();

void writeBenchmarkLinePreamble(size_t nBytes, size_t nElem, const char typeName[], const char opName[], int root);
void writeBenchmarkLineTerminator(int actualIters, const char *name);
void writeBenchMarkLineNullBody();
void writeBenchmarkLineBody(double timeUsec, double algBw, double busBw, bool reportErrors, int64_t wrongElts, bool report_cputime, bool report_timestamps, bool out_of_place);
testResult_t writeDeviceReport(size_t *maxMem, int localRank, int proc, int totalProcs, int color, const char hostname[], const char *program_name);
void writeResultHeader(bool report_cputime, bool report_timestamps);
void writeResultFooter(const int errors[], const double bw[], double check_avg_bw, const char *program_name);
void finalizeFooter();
void writeMemInfo(memInfo_t* memInfos, int numMemInfos);
void writeErrors();

#endif
