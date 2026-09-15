#include "common.h"
#include "args.h"
#include "ginLatencyTests.h"

/* Ping latency of signal. Single operation, so --gin_op does not apply. */
int main(int argc, char** argv) {
  ginArgs_t args;
  ncclTestGinParseArgs(argc, argv, &args);

  ginTestCaps_t caps = {};
  caps.defaultRsm = GIN_RSM_GPU;
  caps.allowThreadRsm = true;
  caps.opMask = 0;
  caps.defaultOp = GIN_OP_UNSET;
  caps.isSignalOp = true;
  caps.allowAggregateRequests = false;
  caps.allowBidirectional = false;
  caps.allowMultiCtaThreads = false;
  ncclTestGinConfigureArgs(&args, &caps);

  ginBenchmark_t bench = {};
  bench.run = ncclTestGinSignalLatencyPingLaunch;
  bench.name = "GIN SIGNAL PING LATENCY";
  bench.type = GIN_BENCHMARK_TYPE_PING;

  ncclTestGinPerfRun(argc, argv, &args, &bench);
  return 0;
}
