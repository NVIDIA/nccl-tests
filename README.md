# NCCL Tests

These tests check both the performance and the correctness of [NCCL](http://github.com/nvidia/nccl) operations.

## Build

To build the tests, just type `make` or `make -j`

If CUDA is not installed in `/usr/local/cuda`, you may specify `CUDA_HOME`. Similarly, if NCCL is not installed in `/usr`, you may specify `NCCL_HOME`.

```shell
$ make CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl
```

NCCL tests rely on MPI to work on multiple processes, hence multiple nodes. If you want to compile the tests with MPI support, you need to set `MPI=1` and set `MPI_HOME` to the path where MPI is installed.

```shell
$ make MPI=1 MPI_HOME=/path/to/mpi CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl
```

You can also add a suffix to the name of the generated binaries with `NAME_SUFFIX`. For example when compiling with the MPI versions you could use:

```shell
$ make MPI=1 NAME_SUFFIX=_mpi MPI_HOME=/path/to/mpi CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl
```

This will generate test binaries with names such as `all_reduce_perf_mpi`.

## Usage

NCCL tests can run on multiple processes, multiple threads, and multiple CUDA devices per thread. The number of process is managed by MPI and is therefore not passed to the tests as argument. The total number of ranks (=CUDA devices) will be equal to `(number of processes)*(number of threads)*(number of GPUs per thread)`.

### Quick examples

Run on single node with 8 GPUs (`-g 8`), scanning from 8 Bytes to 128MiB (Mebibytes), doubling between each test (`-f 2`) :

```shell
$ ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
```

Run 64 MPI processes on nodes with 8 GPUs each, for a total of 64 GPUs spread across 8 nodes.
Scanning from 8 Bytes to 32GiB (Gibibytes), doubling between each test (`-f 2`).
(NB: The nccl-tests binaries must be compiled with `MPI=1` for this case)

```shell
$ mpirun -np 64 -N 8 ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1
```

### Performance

See the [Performance](doc/PERFORMANCE.md) page for explanation about numbers, and in particular the "busbw" column.

### Arguments

All tests support the same set of arguments :

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
  * `-a,--average <0/1/2/3>` Report performance as an average across all ranks (MPI=1 only). <0=Rank0,1=Avg,2=Min,3=Max>. Default : 1.
* Test operation
  * `-p,--parallel_init <0/1>` use threads to initialize NCCL in parallel. Default : 0.
  * `-c,--check <check iteration count>` perform count iterations, checking correctness of results on each iteration. This can be quite slow on large numbers of GPUs. Default : 1.
  * `-z,--blocking <0/1>` Make NCCL collective blocking, i.e. have CPUs wait and sync after each collective. Default : 0.
  * `-G,--cudagraph <num graph launches>` Capture iterations as a CUDA graph and then replay specified number of times. Default : 0.
  * `-C,--report_cputime <0/1>` Report CPU time instead of latency. Default : 0.
  * `-R,--local_register <0/1/2>` enable local (1) or symmetric (2) buffer registration on send/recv buffers. Default : 0.
  * `-S,--report_timestamps <0/1>` Add timestamp (`"%Y-%m-%d %H:%M:%S"`) to each performance report line. Default : 0.
  * `-J,--output_file <file>` Write [JSON] output to filepath. Infer type from suffix (only `json` supported presently).
  * `-T,--timeout <time in seconds>` timeout each test after specified number of seconds. Default : disabled.
  * `-M,--memory_report <0/1>` enable memory usage report. Default : 0

### Running multiple operations in parallel

NCCL tests allow to partition the set of GPUs into smaller sets, each executing the same operation in parallel.
To split the GPUs, NCCL will compute a "color" for each rank, based on the `NCCL_TESTS_SPLIT` environment variable, then all ranks
with the same color will end up in the same group. The resulting group is printed next to each GPU at the beginning of the test.

`NCCL_TESTS_SPLIT` takes the following syntax: `<operation><value>`. Operation can be `AND`, `OR`, `MOD` or `DIV`. The `&`, `|`, `%`, and `/` symbols are also supported. The value can be either decimal, hexadecimal (prefixed by `0x`) or binary (prefixed by `0b`).

`NCCL_TESTS_SPLIT_MASK="<value>"` is equivalent to `NCCL_TESTS_SPLIT="&<value>"`.

Here are a few examples:

 - `NCCL_TESTS_SPLIT="AND 0x7"` or `NCCL_TESTS_SPLIT="MOD 8"`: On systems with 8 GPUs, run 8 parallel operations, each with 1 GPU per node (purely communicating over the inter-node network)

- `NCCL_TESTS_SPLIT="OR 0x7"` or `NCCL_TESTS_SPLIT="DIV 8"`: On systems with 8 GPUs, run one operation per node, purely intra-node.

- `NCCL_TESTS_SPLIT="AND 0x1"` or `NCCL_TESTS_SPLIT="MOD 2"`: Run two operations, each operation using every other rank.

Note that the reported bandwidth is per group, hence to get the total bandwidth used by all groups, one must multiply by the number of groups.

## Copyright

NCCL tests are provided under the BSD license. All source code and accompanying documentation is copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
