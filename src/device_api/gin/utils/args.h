#pragma once

#include <stddef.h>

typedef enum {
  ncclGinRsmUnset = 0,
  ncclGinRsmThread,
  ncclGinRsmCta,
  ncclGinRsmGpu,
} ncclGinRsm_t;

/*
 * GIN operation under test, selected by --gin_op. A benchmark binary links one
 * test implementation per operation it supports and dispatches to the matching
 * one, so e.g. the put binaries cover plain put, put+signal and put+counter.
 */
typedef enum {
  ncclGinOpUnset = 0,
  ncclGinOpPut,
  ncclGinOpPutSignal,
  ncclGinOpPutCount,
} ncclGinOp_t;

#define GIN_OP_BIT(op) (1u << (unsigned)(op))

typedef struct {
  size_t minBytes;
  size_t maxBytes;
  int stepFactor;
  size_t stepBytes;
  int warmupIters;
  int iters;
  int numCtas;
  int numThreads;
  int ginSkipCreditCheck;
  bool ginStrongSignal;
  bool ginAggregateRequests;
  bool ginBidirectional;
  int queueDepth;
  ncclGinRsm_t ginRsm;
  ncclGinOp_t ginOp;
} ginArgs_t;

/*
 * What a benchmark binary actually implements. Anything a binary does not
 * declare here is rejected by ncclTestGinConfigureArgs with a diagnostic
 * rather than being silently ignored.
 *
 *   defaultRsm / allowThreadRsm – --gin_rsm default (gpu) and whether thread
 *                                 mode is usable (latency yes, throughput no).
 *   opMask                      – OR of GIN_OP_BIT() values --gin_op accepts;
 *                                 0 for binaries with a single operation
 *                                 (the get and signal tests).
 *   defaultOp                   – used when --gin_op is omitted.
 *   isSignalOp                  – the binary's operation is itself a signal,
 *                                 so --gin_strong_signal applies regardless
 *                                 of --gin_op.
 *   allowAggregateRequests      – --gin_ag (throughput only).
 *   allowBidirectional          – --gin_bd (throughput only).
 *   allowMultiCtaThreads        – -c/-t other than 1 (throughput only;
 *                                 latency requires a single CTA and thread).
 */
typedef struct {
  ncclGinRsm_t defaultRsm;
  bool allowThreadRsm;
  unsigned opMask;
  ncclGinOp_t defaultOp;
  bool isSignalOp;
  bool allowAggregateRequests;
  bool allowBidirectional;
  bool allowMultiCtaThreads;
} ginTestCaps_t;

void ncclTestGinParseArgs(int argc, char** argv, ginArgs_t* args);
void ncclTestGinConfigureArgs(ginArgs_t* args, const ginTestCaps_t* caps);
const char* ncclTestGinOpName(ncclGinOp_t op);
const char* ncclTestGinRsmName(ncclGinRsm_t rsm);
