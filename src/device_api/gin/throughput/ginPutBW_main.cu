#include "common.h"
#include "args.h"
#include "ginThroughputTests.h"

/* Bandwidth / message rate of the put family: put, put+signal or put+counter. */
int main(int argc, char** argv) {
  ginArgs_t args;
  ncclTestGinParseArgs(argc, argv, &args);

  ginTestCaps_t caps = {};
  caps.defaultRsm = ncclGinRsmGpu;
  caps.allowThreadRsm = false;
  caps.opMask = GIN_OP_BIT(ncclGinOpPut) | GIN_OP_BIT(ncclGinOpPutSignal) | GIN_OP_BIT(ncclGinOpPutCount);
  caps.defaultOp = ncclGinOpPut;
  caps.isSignalOp = false;
  caps.allowAggregateRequests = true;
  caps.allowBidirectional = true;
  caps.allowMultiCtaThreads = true;
  ncclTestGinConfigureArgs(&args, &caps);

  ginBenchmark_t bench = {};
  bench.type = GIN_BENCHMARK_TYPE_THROUGHPUT;
  switch (args.ginOp) {
  case ncclGinOpPutSignal:
    bench.run = ncclTestGinPutSignalBWLaunch;
    bench.name = "GIN PUT_SIGNAL BANDWIDTH/MESSAGE RATE";
    break;
  case ncclGinOpPutCount:
    bench.run = ncclTestGinPutCounterBWLaunch;
    bench.name = "GIN PUT_COUNTER BANDWIDTH/MESSAGE RATE";
    break;
  default:
    bench.run = ncclTestGinPutBWLaunch;
    bench.name = "GIN PUT BANDWIDTH/MESSAGE RATE";
    break;
  }

  ncclTestGinPerfRun(argc, argv, &args, &bench);
  return 0;
}
