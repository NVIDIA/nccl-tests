#include "common.h"
#include "args.h"
#include "ginThroughputTests.h"

/*
 * Message rate of signal. Signals carry no payload, so bandwidth is reported as
 * zero and only the message rate column is meaningful.
 */
int main(int argc, char** argv) {
  ginArgs_t args;
  ncclTestGinParseArgs(argc, argv, &args);

  ginTestCaps_t caps = {};
  caps.defaultRsm = GIN_RSM_GPU;
  caps.allowThreadRsm = false;
  caps.opMask = 0;
  caps.defaultOp = GIN_OP_UNSET;
  caps.isSignalOp = true;
  caps.allowAggregateRequests = true;
  caps.allowBidirectional = true;
  caps.allowMultiCtaThreads = true;
  ncclTestGinConfigureArgs(&args, &caps);

  ginBenchmark_t bench = {};
  bench.run = ncclTestGinSignalBWLaunch;
  bench.name = "GIN SIGNAL MESSAGE RATE";
  bench.type = GIN_BENCHMARK_TYPE_THROUGHPUT;
  bench.payloadMode = GIN_THROUGHPUT_PAYLOAD_NONE;

  ncclTestGinPerfRun(argc, argv, &args, &bench);
  return 0;
}
