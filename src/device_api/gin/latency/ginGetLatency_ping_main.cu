#include "common.h"
#include "args.h"
#include "ginLatencyTests.h"

/* Ping latency of get. Single operation, so --gin_op does not apply. */
int main(int argc, char** argv) {
  ginArgs_t args;
  ncclTestGinParseArgs(argc, argv, &args);

  ginTestCaps_t caps = {};
  caps.defaultRsm = ncclGinRsmGpu;
  caps.allowThreadRsm = true;
  caps.opMask = 0;
  caps.defaultOp = ncclGinOpUnset;
  caps.isSignalOp = false;
  caps.allowAggregateRequests = false;
  caps.allowBidirectional = false;
  caps.allowMultiCtaThreads = false;
  ncclTestGinConfigureArgs(&args, &caps);

  ginBenchmark_t bench = {};
  bench.run = ncclTestGinGetLatencyPingLaunch;
  bench.name = "GIN GET PING LATENCY";
  bench.type = GIN_BENCHMARK_TYPE_PING;

  ncclTestGinPerfRun(argc, argv, &args, &bench);
  return 0;
}
