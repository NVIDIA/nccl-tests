#include "common.h"
#include "args.h"
#include "ginThroughputTests.h"

/* Bandwidth / message rate of the put family: put, put+signal or put+counter. */
int main(int argc, char** argv) {
  ginArgs_t args;
  ncclTestGinParseArgs(argc, argv, &args);

  ginTestCaps_t caps = {};
  caps.defaultRsm = GIN_RSM_GPU;
  caps.allowThreadRsm = false;
  caps.opMask = GIN_OP_BIT(GIN_OP_PUT) | GIN_OP_BIT(GIN_OP_PUT_SIGNAL) | GIN_OP_BIT(GIN_OP_PUT_COUNTER);
  caps.defaultOp = GIN_OP_PUT;
  caps.isSignalOp = false;
  caps.allowAggregateRequests = true;
  caps.allowBidirectional = true;
  caps.allowMultiCtaThreads = true;
  ncclTestGinConfigureArgs(&args, &caps);

  ginBenchmark_t bench = {};
  bench.type = GIN_BENCHMARK_TYPE_THROUGHPUT;
  switch (args.ginOp) {
  case GIN_OP_PUT_SIGNAL:
    bench.run = ncclTestGinPutSignalBWLaunch;
    bench.name = "GIN PUT_SIGNAL BANDWIDTH/MESSAGE RATE";
    break;
  case GIN_OP_PUT_COUNTER:
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
