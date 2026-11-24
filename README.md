# NCCL Tests

These tests check both the performance and the correctness of [NCCL](http://github.com/nvidia/nccl) operations.

## Build

To build the tests, just type `make` or `make -j`

If CUDA is not installed in `/usr/local/cuda`, you may specify `CUDA_HOME`. Similarly, if NCCL is not installed in `/usr`, you may specify `NCCL_HOME`.

```shell
$ make CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl
```

NCCL tests can use MPI to work on multiple processes, hence multiple nodes. If you want to compile the tests with MPI support, you need to set `MPI=1` and set `MPI_HOME` to the path where MPI is installed.

```shell
$ make MPI=1 MPI_HOME=/path/to/mpi CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl
```

In some distributions, MPI's header files may not reside in `$MPI_HOME/include`. In this case, you can set `MPI_INCLUDE` to the path where MPI's header files are installed. 

```shell
$ make MPI=1 MPI_HOME=/path/to/mpi MPI_INCLUDE=/path/to/mpi/headers CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl
```

You can also add a suffix to the name of the generated binaries with `NAME_SUFFIX`. For example when compiling with the MPI versions you could use:

```shell
$ make MPI=1 NAME_SUFFIX=_mpi MPI_HOME=/path/to/mpi CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl
```

This will generate test binaries with names such as `all_reduce_perf_mpi`.

## Usage

NCCL tests can run on multiple processes, multiple threads, and multiple CUDA devices per thread. With MPI, the number of processes is managed by MPI and is therefore not passed to the tests as an argument. `NCCL_COMM_ID` is required for non-MPI multi-process launches; supply process count and rank through `NCCL_TESTS_COMM_NRANKS` and `NCCL_TESTS_COMM_RANK`, or recognized launcher variables. The total number of ranks (=CUDA devices) will be equal to `(number of processes)*(number of threads)*(number of GPUs per thread)`.

### Quick examples

#### Single process (single node only)

Run on single node with 8 GPUs (`-g 8`), scanning from 8 Bytes to 128MiB (Mebibytes), doubling between each test (`-f 2`) :

```shell
$ ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
```

#### Multi-process with MPI

Run 64 MPI processes on nodes with 8 GPUs each, for a total of 64 GPUs spread across 8 nodes.
Scanning from 8 Bytes to 8GiB (Gibibytes), doubling between each test (`-f 2`).
(NB: The nccl-tests binaries must be compiled with `MPI=1` for this case)

```shell
$ mpirun -np 64 -N 8 ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1
```

#### Multi-process without MPI

Non-MPI builds can also be launched as multiple processes by setting the same NCCL bootstrap ID and one process rank per process:

```shell
$ NCCL_COMM_ID=node0:45678 NCCL_TESTS_COMM_NRANKS=2 NCCL_TESTS_COMM_RANK=0 NCCL_TESTS_COMM_LOCAL_RANK=0 ./build/all_reduce_perf -b 8 -e 128M -g 1
$ NCCL_COMM_ID=node0:45678 NCCL_TESTS_COMM_NRANKS=2 NCCL_TESTS_COMM_RANK=1 NCCL_TESTS_COMM_LOCAL_RANK=1 ./build/all_reduce_perf -b 8 -e 128M -g 1
```

`NCCL_COMM_ID` uses NCCL's native `<host>:<port>` bootstrap format and must be identical for all processes. IPv6 literals should use bracket syntax, for example `[2001:db8::1]:45678`. `NCCL_TESTS_COMM_NRANKS` and `NCCL_TESTS_COMM_RANK` replace the MPI process count and process rank; the total NCCL rank count remains `(NCCL_TESTS_COMM_NRANKS)*(number of threads)*(number of GPUs per thread)`. `NCCL_TESTS_COMM_LOCAL_RANK` is optional, and common launcher variables such as `LOCAL_RANK`, `SLURM_LOCALID`, `OMPI_COMM_WORLD_LOCAL_RANK`, and `MV2_COMM_WORLD_LOCAL_RANK` are also recognized.

If no local rank environment variable is found, non-MPI multi-process launches fall back to assigning devices from `NCCL_TESTS_COMM_RANK` or launcher rank modulo the number of visible per-process GPU slots. This assumes compatible block placement; supply a local rank explicitly for cyclic or irregular placement.

NOTE: `NCCL_TESTS_SPLIT` and `-p,--parallel_init` are currently only supported with MPI launches.

For non-MPI launches, `-I,--per_iter_timing` reports the maximum iteration time across all ranks; separate per-process timing rows require MPI.

For non-MPI launches, `-J,--output_file` creates a JSON file only on process rank zero. Device metadata is local to rank zero, not gathered across processes; startup device reports from other processes remain available on stdout.

When `NCCL_COMM_ID` is set, non-MPI builds can also derive the process rank and process count from common launcher variables: `SLURM_PROCID` with `SLURM_NTASKS` or `SLURM_NPROCS`, `PMI_RANK` with `PMI_SIZE`, `PMIX_RANK` with `PMIX_SIZE`, `OMPI_COMM_WORLD_RANK` with `OMPI_COMM_WORLD_SIZE`, `MV2_COMM_WORLD_RANK` with `MV2_COMM_WORLD_SIZE`, or `RANK` with `WORLD_SIZE`. Explicit `NCCL_TESTS_COMM_RANK` and `NCCL_TESTS_COMM_NRANKS` take precedence.

For SLURM allocations, `scripts/slurm-nccl-tests.sh` can either launch the test with `srun` or run as the command wrapper inside an existing `srun` launch. Bootstrap discovery runs inside the actual step, using `SLURM_STEP_NODELIST` or, as a fallback, `SLURM_JOB_NODELIST`. A supplied `NCCL_COMM_ID` skips host discovery and does not require `scontrol`:

```shell
$ scripts/slurm-nccl-tests.sh --ntasks=2 --gpus-per-task=1 -- ./build/all_reduce_perf -b 8 -e 128M -g 1
$ srun --kill-on-bad-exit=1 --ntasks=2 --gpus-per-task=1 scripts/slurm-nccl-tests.sh ./build/all_reduce_perf -b 8 -e 128M -g 1
```

The SLURM helper is quiet by default; pass `-v,--verbose` or set `NCCL_TESTS_SLURM_VERBOSE=1` to print the derived `NCCL_COMM_ID` and nodelist diagnostics.

Launcher mode defaults to `--kill-on-bad-exit=1`; include that option when invoking `srun` yourself so a failed rank terminates its peers. `NCCL_TESTS_COMM_PORT` overrides the derived bootstrap port and must be in 1-65535.

The default bootstrap port incorporates both the SLURM job and step IDs to separate concurrent steps within an allocation. It is not a collision-free port reservation; use an explicit port when needed.

When `CUDA_VISIBLE_DEVICES` contains a single device, the wrapper defaults `NCCL_TESTS_COMM_LOCAL_RANK` to `0`, preserving an explicit override. For task-isolated sets of multiple GPUs, set `NCCL_TESTS_COMM_LOCAL_RANK=0` explicitly and match `-g` to the visible allocation. When all node GPUs are visible to every task, leave local-rank assignment to the launcher instead.

For Kubernetes, [examples/kubernetes.yaml](examples/kubernetes.yaml) provides an annotated two-node Indexed Job and headless Service. It assigns one process and GPU per pod, injects rank variables directly, and uses rank-zero pod DNS for bootstrap. Replace the image and binary path before applying it; the example uses TCP, with notes for adapting it to RDMA.

### Performance

See the [Performance](doc/PERFORMANCE.md) page for explanation about numbers, and in particular the "busbw" column.

### Arguments

Collective perf tests support the same set of arguments :

* Number of GPUs
  * `-t,--nthreads <num threads>` number of threads per process. Default : 1.
  * `-g,--ngpus <GPUs per thread>` number of gpus per thread. Default : 1.
* Sizes to scan
  * `-b,--minbytes <min size in bytes>` minimum size to start with. Default : 32M (Mebibytes).
  * `-e,--maxbytes <max size in bytes>` maximum size to end at. Default : 32M (Mebibytes).
  * Increments can be either fixed or a multiplication factor. Only one of those should be used.
    * `-i,--stepbytes <increment size>` fixed increment between sizes. Default : 1M (Mebibytes).
    * `-f,--stepfactor <increment factor>` multiplication factor between sizes. Default : disabled.
* NCCL operations arguments
  * `-o,--op <sum/prod/min/max/avg/all>` Specify which reduction operation to perform. Only relevant for reduction operations like Allreduce, Reduce or ReduceScatter. Default : Sum.
  * `-d,--datatype <nccltype/all>` Specify which datatype to use. Default : Float.
  * `-r,--root <root/all>` Specify which root to use. Only for operations with a root like broadcast or reduce. Default : 0.
* Performance
  * `-n,--iters <iteration count>` number of iterations. Default : 20.
  * `-w,--warmup_iters <warmup iteration count>` number of warmup iterations (not timed). Default : 1.
  * `-m,--agg_iters <aggregation count>` number of operations to aggregate together in each iteration. Default : 1.
  * `-N,--run_cycles <cycle count>` run & print each cycle. Default : 1; 0=infinite.
  * `-a,--average <0/1/2/3>` Report performance as an average across all ranks. <0=Rank0,1=Avg,2=Min,3=Max>. Default : 1.
  * `-I,--per_iter_timing <0/1>` collect per-iteration CUDA event timings and print summary columns.
    `i_p99` uses nearest-rank percentile and may equal `i_max` with fewer than 100 samples.
    Incompatible with CUDA graph capture (`-G`). Default : 0.
  * `-K,--per_iter_skip <count>` exclude leading samples from `-I` summary statistics.
    Raw per-iteration JSON data remains complete. Default : 0.
* Test operation
  * `-p,--parallel_init <0/1>` use threads to initialize NCCL in parallel. Default : 0.
  * `-c,--check <check iteration count>` perform count iterations, checking correctness of results on each iteration. This can be quite slow on large numbers of GPUs. Default : 1.
  * `-z,--blocking <0/1/2/3>` collective blocking mode. Default: 0.
    * `0` : non-blocking (default)
    * `1` : wait and barrier after each inner iteration (`-m`)
    * `2` : wait after each inner iteration, no barrier
    * `3` : wait and barrier after each outer iteration (`-n`); reported time excludes barrier. Incompatible with CUDA graph capture (`-G`).
  * `-G,--cudagraph <num graph launches>` Capture iterations as a CUDA graph and then replay specified number of times. Default : 0.
  * `-C,--report_cputime <0/1>` Report CPU time instead of latency. Default : 0.
  * `-R,--local_register <0/1/2>` enable local (1) or symmetric (2) buffer registration on send/recv buffers. Default : 0.
  * `-D,--device_implementation <implementation number>` use custom device API implementation. Not every collective has a custom device API implementations (currently just all\_reduce and alltoall). Default : 0 (use traditional NCCL host implementation). Note: values > 0 require symmetric memory registration (`-R 2`).
  * `-H,--host_rma_implementation <num RMA contexts>` use host one-sided RMA (`ncclPutSignal`/`ncclWaitSignal`) instead of the built-in collective. Supported by sendrecv, alltoall, all\_gather, broadcast, gather, and scatter. `1` uses a single context; values `>1` distribute RMA traffic across contexts and require NCCL >= 2.31. Requires symmetric registration (`-R 2`). Mutually exclusive with `-D`.
  * `-V,--device_cta_count <number>` number of CTAs for device API implementation. Must be positive and less than 128. Default : 16.
  * `-S,--report_timestamps <0/1>` Add timestamp (`"%Y-%m-%d %H:%M:%S"`) to each performance report line. Default : 0.
  * `-J,--output_file <file>` Write [JSON] output to filepath. Infer type from suffix (only `json` supported presently).
  * `-T,--timeout <time in seconds>` timeout each test after specified number of seconds. Default : disabled.
  * `-M,--memory <0/1>` enable memory usage report. Default : 0.
  * `-u,--unalign <index of first element>` Misalign source and destination buffers. Default : 0.
  * `-U,--tuning <0/1>` report NCCL tuning information. Requires NCCL 2.28 or newer; symmetric collective identification and kernel variant reporting require NCCL 2.31 or newer. Default : 0.

### AlltoAll hybrid diagnostic control

On Linux with NCCL 2.28.7 or newer and GIN support, `alltoall_perf -D 4`
uses GIN for remote peers and vectorized LSA copies (with a scalar fallback)
for local peers. `-D 6` uses the same device-communicator requirements and
peer mapping, but adds a CTA-cooperative `gin.flush()` after the remote puts
and before each CTA's local copies. This provides a flush-before-LSA control
for investigating contention between the two paths. It is a diagnostic mode;
the extra completion wait removes overlap within each CTA and is not a
recommended performance optimization.

Compare the modes with the same ranks, effective message size, and CTA count.
For example, on two nodes with four GPUs each:

```shell
mpirun -np 8 -H nodeA:4,nodeB:4 --map-by ppr:4:node \
  ./build/alltoall_perf -t 1 -g 1 -d float -b 512K -e 512K \
  -R 2 -D 4 -V 32 -n 100 -w 20 -c 5
# Repeat with -D 6 for the flush-before-LSA control.
```

This example has 64 KiB per peer (512 KiB total per rank). Both modes request
one GIN context; increasing `-V` changes the CTA count, not the requested
context count. `-D 3` is a separate GIN-only comparison that also sends to
local peers through GIN, so its transport mix differs. AllToAll verification
covers out-of-place results; the in-place column is not checked.

### AlltoAll hybrid with lean synchronization

With NCCL 2.30 or newer, `-D 7` is a hybrid variant with the same transport
split as `-D 4` (GIN for remote peers, the vectorized LSA copy for local peers)
and one GIN context. It changes how remote traffic is issued and synchronized:

- One warp posts all remote puts, waits for the incoming signals and flushes,
  instead of spreading puts over CTAs and flushing in every CTA.
- Each peer's block is sent as `min(4, ceil(bytes per peer / 4 MiB))` puts.
- Only the CTA that writes to remote peers runs a GIN world barrier at the
  start; every CTA runs an LSA barrier for same-node peers. At the end, the
  signal wait covers remote data, so only the LSA barrier is used, instead of
  a combined LSA+GIN barrier at both ends.

Run it like `-D 4`, for example `-R 2 -D 7 -V 32`.

### Running multiple operations in parallel

NCCL tests allow to partition the set of GPUs into smaller sets, each executing the same operation in parallel.
To split the GPUs, NCCL will compute a "color" for each rank, based on the `NCCL_TESTS_SPLIT` environment variable, then all ranks
with the same color will end up in the same group. The resulting group is printed next to each GPU at the beginning of the test.

NOTE: This split mode is currently supported only for MPI launches.

`NCCL_TESTS_SPLIT` takes the following syntax: `<operation><value>`. Operation can be `AND`, `OR`, `MOD` or `DIV`. The `&`, `|`, `%`, and `/` symbols are also supported. The value can be either decimal, hexadecimal (prefixed by `0x`) or binary (prefixed by `0b`).

`NCCL_TESTS_SPLIT_MASK="<value>"` is equivalent to `NCCL_TESTS_SPLIT="&<value>"`.

Here are a few examples:

 - `NCCL_TESTS_SPLIT="AND 0x7"` or `NCCL_TESTS_SPLIT="MOD 8"`: On systems with 8 GPUs, run 8 parallel operations, each with 1 GPU per node (purely communicating over the inter-node network)

- `NCCL_TESTS_SPLIT="OR 0x7"` or `NCCL_TESTS_SPLIT="DIV 8"`: On systems with 8 GPUs, run one operation per node, purely intra-node.

- `NCCL_TESTS_SPLIT="AND 0x1"` or `NCCL_TESTS_SPLIT="MOD 2"`: Run two operations, each operation using every other rank.

Note that the reported bandwidth is per group, hence to get the total bandwidth used by all groups, one must multiply by the number of groups.

### Communicator operations (`comm_ops_perf`)

Benchmarks NCCL communicator init, split, shrink, and grow latency. Requires `MPI=1`; does not use the collective arguments above.

```shell
$ mpirun -np 8 ./build/comm_ops_perf init
$ mpirun -np 8 ./build/comm_ops_perf split --sweep-comms fixed
$ mpirun -np 8 ./build/comm_ops_perf grow --resize factor-1.5
```

Pass a test mode (`init`, `split`, `shrink`, or `grow`) followed by options such as `-i` (timed iterations), `-w` (untimed warmup iterations), `-s` / `-b` / `-e` (comm-size sweep), `-r` (resize for split/shrink/grow), `-S` (share resources), and `-a` (abort path). Output is per-operation latency summaries in ms; use `--help` for details.

### GIN device API microbenchmarks (`device_api/gin`)

Measures latency and bandwidth / message rate of NCCL GIN point-to-point device API operations (`put`, `get`, `signal`, and combined variants). Requires `MPI=1`, NCCL 2.30.7 or newer, and exactly 2 ranks (one GPU each). Against older NCCL these binaries are simply left out of the build.

See [src/device_api/gin/README.md](src/device_api/gin/README.md) for the full list of benchmarks and their options.

### CuTe DSL device API microbenchmarks (`device_api/dsl`)

Measures NCCL GIN put and LSACopy ping latency, ping-pong latency, and bandwidth from Python CuTe DSL kernels. Requires NCCL4Py, CuPy, MPI, and exactly two ranks. GIN runs across nodes; LSACopy runs on two GPUs in the same LSA team.

See [src/device_api/dsl/README.md](src/device_api/dsl/README.md) for dependencies, launch commands, and options.

## Copyright

NCCL tests are provided under the BSD license. All source code and accompanying documentation is copyright (c) 2016-2026, NVIDIA CORPORATION. All rights reserved.
