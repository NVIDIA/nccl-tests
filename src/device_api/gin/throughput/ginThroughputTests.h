#pragma once

#include "common.h"

/*
 * Launch entry points for the throughput (bandwidth / message rate) tests, one
 * per GIN operation, each implemented by the like-named .cu file in this
 * directory. The *_main.cu runners select one of these according to --gin_op;
 * the linker then pulls only the referenced implementations out of the category
 * archive.
 *
 * Every entry point matches ginRunFn_t so it can be assigned to
 * ginBenchmark_t::run directly.
 */

void ncclTestGinPutBWLaunch(ncclDevComm dcomm, ncclDevResourceHandle devBufHandle, cudaStream_t stream,
    const ginArgs_t* args, size_t numElems, int iters);
void ncclTestGinPutSignalBWLaunch(ncclDevComm dcomm, ncclDevResourceHandle devBufHandle, cudaStream_t stream,
    const ginArgs_t* args, size_t numElems, int iters);
void ncclTestGinPutCounterBWLaunch(ncclDevComm dcomm, ncclDevResourceHandle devBufHandle, cudaStream_t stream,
    const ginArgs_t* args, size_t numElems, int iters);

void ncclTestGinGetBWLaunch(ncclDevComm dcomm, ncclDevResourceHandle devBufHandle, cudaStream_t stream,
    const ginArgs_t* args, size_t numElems, int iters);

void ncclTestGinSignalBWLaunch(ncclDevComm dcomm, ncclDevResourceHandle devBufHandle, cudaStream_t stream,
    const ginArgs_t* args, size_t numElems, int iters);
