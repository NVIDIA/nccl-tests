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
  caps.defaultRsm = GIN_RSM_GPU;
  caps.allowThreadRsm = true;
  caps.opMask = GIN_OP_BIT(GIN_OP_PUT) | GIN_OP_BIT(GIN_OP_PUT_SIGNAL);
  caps.defaultOp = GIN_OP_PUT;
  caps.isSignalOp = false;
  caps.allowAggregateRequests = false;
  caps.allowBidirectional = false;
  caps.allowMultiCtaThreads = false;
  ncclTestGinConfigureArgs(&args, &caps);

  ginBenchmark_t bench = {};
  bench.type = GIN_BENCHMARK_TYPE_PING_PONG;
  switch (args.ginOp) {
  case GIN_OP_PUT_SIGNAL:
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
