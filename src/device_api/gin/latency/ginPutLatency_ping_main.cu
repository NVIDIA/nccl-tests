#include "common.h"
#include "args.h"
#include "ginLatencyTests.h"

/* Ping latency of the put family: put, put+signal or put+counter. */
int main(int argc, char** argv) {
  ginArgs_t args;
  ncclTestGinParseArgs(argc, argv, &args);

  ginTestCaps_t caps = {};
  caps.defaultRsm = ncclGinRsmGpu;
  caps.allowThreadRsm = true;
  caps.opMask = GIN_OP_BIT(ncclGinOpPut) | GIN_OP_BIT(ncclGinOpPutSignal) | GIN_OP_BIT(ncclGinOpPutCount);
  caps.defaultOp = ncclGinOpPut;
  caps.isSignalOp = false;
  caps.allowAggregateRequests = false;
  caps.allowBidirectional = false;
  caps.allowMultiCtaThreads = false;
  ncclTestGinConfigureArgs(&args, &caps);

  ginBenchmark_t bench = {};
  bench.type = GIN_BENCHMARK_TYPE_PING;
  switch (args.ginOp) {
  case ncclGinOpPutSignal:
    bench.run = ncclTestGinPutSignalLatencyPingLaunch;
    bench.name = "GIN PUT_SIGNAL PING LATENCY";
    break;
  case ncclGinOpPutCount:
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
