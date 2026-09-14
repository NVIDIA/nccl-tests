# GIN Device API P2P Performance Tests

These tests measure latency and bandwidth / message rate of NCCL GIN
point-to-point device API operations (`put`, `get`, `signal`, and combined
variants).

They require NCCL 2.30.7+, MPI, and must always run with exactly
**2 ranks** (one GPU each).

## Build

From the root of the `nccl-tests` repository, build the tests with MPI enabled:

```shell
$ make -j MPI=1 MPI_HOME=/path/to/mpi NCCL_HOME=/path/to/nccl
```

If CUDA is not installed in `/usr/local/cuda`, set `CUDA_HOME`. Binaries are
written under `build/device_api/gin/`.

You can also build a single binary, for example:

```shell
$ make -C src -j MPI=1 MPI_HOME=/path/to/mpi NCCL_HOME=/path/to/nccl \
    build/device_api/gin/throughput/ginPutBW_perf
```

With CMake, configure and build from the repository root:

```shell
$ cmake -S . -B build -DNCCL_HOME=/path/to/nccl
$ cmake --build build -j
```

## Usage

Process count is managed by MPI and is not passed as a test argument. Use
exactly two ranks:

```shell
$ mpirun -np 2 ./build/device_api/gin/<category>/<binary>_perf [OPTIONS]
```

### Quick examples

Default put bandwidth size sweep (4B to 4M, factor 2):

```shell
$ mpirun -np 2 ./build/device_api/gin/throughput/ginPutBW_perf
```

put+signal bandwidth with 16 CTAs and 32 threads per CTA:

```shell
$ mpirun -np 2 ./build/device_api/gin/throughput/ginPutBW_perf \
    --gin_op put_signal -c 16 -t 32
```

put ping-pong latency from 4 B to 1 MiB:

```shell
$ mpirun -np 2 ./build/device_api/gin/latency/ginPutLatency_pingPong_perf \
    -b 4 -e 1M -f 2
```

### Binaries

| Category | Binary | Operations |
| --- | --- | --- |
| Throughput | `ginPutBW_perf` | `--gin_op put` (default), `put_signal`, or `put_counter` |
| Throughput | `ginGetBW_perf` | get |
| Throughput | `ginSignalBW_perf` | signal |
| Latency | `ginPutLatency_ping_perf` | `--gin_op put` (default), `put_signal`, or `put_counter` |
| Latency | `ginPutLatency_pingPong_perf` | `--gin_op put` (default) or `put_signal` |
| Latency | `ginGetLatency_ping_perf` | get |
| Latency | `ginSignalLatency_ping_perf` | signal |
| Latency | `ginSignalLatency_pingPong_perf` | signal |

put+counter is supported on ping latency and bandwidth only (not ping-pong): a
counter reports local completion and cannot drive the peer's turn.

### Output

Throughput tests print message size, message count, bandwidth (MiB/s), and message rate
(MPPS).

Latency tests print size, iteration count, and one-way latency (us).

Rank 0 also prints the resolved configuration (RSM, CTA/thread counts for
throughput, and any enabled optional flags) above the results table.

## Arguments

All binaries share the same argument parser. Flags that a binary does not
support are rejected rather than ignored.

* Sizes to scan
  * `-b, --minbytes <size>` minimum size in bytes. Default: `sizeof(int)`. `K`/`M`/`G` suffixes OK.
  * `-e, --maxbytes <size>` maximum size / registered buffer size in bytes. Default: 4M.
    The size sweep runs through `-e`. For throughput, threads partition that
    buffer into `maxbytes / size` slots (`threadIdx.x % slots`); when `size`
    reaches `-e`, there is only one slot and threads reuse the same region.
  * `-f <factor>` multiplicative step between sizes. Default: 2.
  * `-i <stepbytes>` additive step in bytes; overrides `-f` when non-zero. Default: 0.
* Performance
  * `-w <iters>` warmup iterations per size (not timed). Default: 50; `0` disables.
  * `-n, --iters <iters>` measured iterations per size. Default: 500.
* Launch geometry (mainly for throughput)
  * `-c, --num_ctas <n>` number of CTAs. Default: 1.
  * `-t, --num_threads <n>` number of threads per CTA. Default: 1.
    Latency tests require `-c 1` and `-t 1` (the defaults); other values are rejected.
* GIN options
  * `--gin_op <put|put_signal|put_counter>` operation for put-family binaries. Default: `put`.
  * `--gin_rsm <thread|cta|gpu>` resource sharing mode.
    Default: `gpu` for all tests.
    Throughput rejects `thread`.
  * `--gin_tx_depth <depth>` GIN send queue depth. Default: 1024.
  * `--gin_skip_credit_check` skip GIN credit checks. Default: off.
  * `--gin_strong_signal` use Strong instead of Weak signals where supported. Default: Weak.
  * `--gin_ag` enable `ncclGinOptFlagsAggregateRequests` (throughput only). Default: off.
    Requires `-t` to be a multiple of 32.
  * `--gin_bd` bidirectional bandwidth: both ranks send; reported metric is the
    sum of each rank's average (throughput only). Default: off.
  * `--remote_mem_type <device|host>` selects device memory (default) or CPU-backed host
    memory for the remote rank (source memory in GET or destination memory in PUT).
  * `--local_mem_type <device|host>` selects device memory (default) or CPU-backed host
    memory for the local rank (destination memory in GET or source memory in PUT).

### Optional optimizations

By default, optimization knobs above are off. Pass them explicitly when you
want a more aggressive configuration, for example:

```shell
$ mpirun -np 2 ./build/device_api/gin/throughput/ginPutBW_perf \
    --gin_op put_signal -c 16 -t 32 --gin_rsm cta --gin_ag --gin_skip_credit_check
```
