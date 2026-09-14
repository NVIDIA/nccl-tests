#include "common.h"
#include "args.h"
#include "ginLatencyTests.h"

/*
 * Ping-pong latency of the put family. There is no put+counter variant here: a
 * counter reports local completion, which cannot drive the peer's turn.
 */
int main(int argc, char** argv) {
  ginArgs_t args;
  ncclTestGinParseArgs(argc, argv, &args);

  ginTestCaps_t caps = {};
  caps.defaultRsm = ncclGinRsmGpu;
  caps.allowThreadRsm = true;
  caps.opMask = GIN_OP_BIT(ncclGinOpPut) | GIN_OP_BIT(ncclGinOpPutSignal);
  caps.defaultOp = ncclGinOpPut;
  caps.isSignalOp = false;
  caps.allowAggregateRequests = false;
  caps.allowBidirectional = false;
  caps.allowMultiCtaThreads = false;
  caps.canUseHostMemory = false;
  ncclTestGinConfigureArgs(&args, &caps);

  ginBenchmark_t bench = {};
  bench.type = GIN_BENCHMARK_TYPE_PING_PONG;
  switch (args.ginOp) {
  case ncclGinOpPutSignal:
    bench.run = ncclTestGinPutSignalLatencyPingPongLaunch;
    bench.name = "GIN PUT_SIGNAL PING-PONG LATENCY";
    break;
  default:
    bench.run = ncclTestGinPutLatencyPingPongLaunch;
    bench.name = "GIN PUT PING-PONG LATENCY";
    break;
  }

  ncclTestGinPerfRun(argc, argv, &args, &bench);
  return 0;
}
