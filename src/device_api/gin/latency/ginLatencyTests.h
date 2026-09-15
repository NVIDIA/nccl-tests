#pragma once

#include "common.h"

/*
 * Launch entry points for the latency tests, one per GIN operation, each
 * implemented by the like-named .cu file in this directory. The *_main.cu
 * runners select one of these according to --gin_op; the linker then pulls only
 * the referenced implementations out of the category archive.
 *
 * Every entry point matches ginRunFn_t so it can be assigned to
 * ginBenchmark_t::run directly.
 */

void ncclTestGinPutLatencyPingLaunch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                     const ginArgs_t* args, size_t numElems, int iters);
void ncclTestGinPutSignalLatencyPingLaunch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                           const ginArgs_t* args, size_t numElems, int iters);
void ncclTestGinPutCounterLatencyPingLaunch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                            const ginArgs_t* args, size_t numElems, int iters);

void ncclTestGinGetLatencyPingLaunch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                     const ginArgs_t* args, size_t numElems, int iters);

void ncclTestGinSignalLatencyPingLaunch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                        const ginArgs_t* args, size_t numElems, int iters);

void ncclTestGinPutLatencyPingPongLaunch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                         const ginArgs_t* args, size_t numElems, int iters);
void ncclTestGinPutSignalLatencyPingPongLaunch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                               const ginArgs_t* args, size_t numElems, int iters);

void ncclTestGinSignalLatencyPingPongLaunch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                            const ginArgs_t* args, size_t numElems, int iters);
