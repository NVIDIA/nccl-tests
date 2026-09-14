#include "common.h"
#include "args.h"
#include "ginThroughputTests.h"

/* Bandwidth / message rate of get. Single operation, so --gin_op does not apply. */
int main(int argc, char** argv) {
  ginArgs_t args;
  ncclTestGinParseArgs(argc, argv, &args);

  ginTestCaps_t caps = {};
  caps.defaultRsm = ncclGinRsmGpu;
  caps.allowThreadRsm = false;
  caps.opMask = 0;
  caps.defaultOp = ncclGinOpUnset;
  caps.isSignalOp = false;
  caps.allowAggregateRequests = true;
  caps.allowBidirectional = true;
  caps.allowMultiCtaThreads = true;
  ncclTestGinConfigureArgs(&args, &caps);

  ginBenchmark_t bench = {};
  bench.run = ncclTestGinGetBWLaunch;
  bench.name = "GIN GET BANDWIDTH/MESSAGE RATE";
  bench.type = GIN_BENCHMARK_TYPE_THROUGHPUT;

  ncclTestGinPerfRun(argc, argv, &args, &bench);
  return 0;
}
