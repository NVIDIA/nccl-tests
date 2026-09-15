#include "common.h"
#include "args.h"
#include "ginLatencyTests.h"

/* Ping latency of the put family: put, put+signal or put+counter. */
int main(int argc, char** argv) {
  ginArgs_t args;
  ncclTestGinParseArgs(argc, argv, &args);

  ginTestCaps_t caps = {};
  caps.defaultRsm = GIN_RSM_GPU;
  caps.allowThreadRsm = true;
  caps.opMask = GIN_OP_BIT(GIN_OP_PUT) | GIN_OP_BIT(GIN_OP_PUT_SIGNAL) | GIN_OP_BIT(GIN_OP_PUT_COUNTER);
  caps.defaultOp = GIN_OP_PUT;
  caps.isSignalOp = false;
  caps.allowAggregateRequests = false;
  caps.allowBidirectional = false;
  caps.allowMultiCtaThreads = false;
  ncclTestGinConfigureArgs(&args, &caps);

  ginBenchmark_t bench = {};
  bench.type = GIN_BENCHMARK_TYPE_PING;
  switch (args.ginOp) {
  case GIN_OP_PUT_SIGNAL:
    bench.run = ncclTestGinPutSignalLatencyPingLaunch;
    bench.name = "GIN PUT_SIGNAL PING LATENCY";
    break;
  case GIN_OP_PUT_COUNTER:
    bench.run = ncclTestGinPutCounterLatencyPingLaunch;
    bench.name = "GIN PUT_COUNTER PING LATENCY";
    break;
  default:
    bench.run = ncclTestGinPutLatencyPingLaunch;
    bench.name = "GIN PUT PING LATENCY";
    break;
  }

  ncclTestGinPerfRun(argc, argv, &args, &bench);
  return 0;
}
