# CuTe DSL device API performance tests

`gin_put.py` and `lsa_copy.py` measure ping latency, ping-pong latency, and
bandwidth from CuTe DSL kernels. JIT compilation happens before timing.
Ping-pong reports one-way latency as half of the median round-trip time;
bandwidth reports bytes sent from rank 0 to rank 1 in GB/s.

## Setup

These scripts run directly from source and require Linux, MPI, CuPy, NCCL4Py
with CuTe device API support, and the matching NCCL host library and device
bitcode. For a standalone environment, install the requirements for the CUDA
major version in use. They include `nccl4py[cu12]>=0.7.0rc1,<0.8` or
`nccl4py[cu13]>=0.7.0rc1,<0.8` and the corresponding packaged NCCL runtime. From the
repository root:

```shell
python -m pip install -r src/device_api/dsl/requirements-cu13.txt
# For CUDA 12, use requirements-cu12.txt instead.
```

Both scripts require NCCL4Py 0.7. The loaded NCCL library version
must exactly match the NCCL version the NCCL4Py bindings were generated from,
as reported by `nccl.core.show_versions()`. Separately, the library must meet
the benchmark feature minimum: NCCL 2.30.7 for GIN put and NCCL 2.32.3 for
LSACopy. At startup, either a version mismatch or an unmet feature minimum
causes every rank to exit with an error before GPU setup or JIT compilation.
A newer NCCL library does not satisfy the matching-version requirement.

### Testing a source-built NCCL

Replace the packaged NCCL4Py installation with NCCL4Py built from the same
NCCL source revision as the installation under test. Point `NCCL_HOME` at
that revision's NCCL installation, containing both the host library and
`libnccl_device.bc` built for the selected CUDA major version. Building
NCCL4Py requires a C++ compiler and the CUDA Toolkit selected by `CUDA_HOME`:

```shell
export CUDA_HOME=/path/to/cuda
export NCCL_HOME=/path/to/nccl/install
python -m pip install -e '/path/to/matching/nccl/bindings/nccl4py[cu13]'
export LD_LIBRARY_PATH="$NCCL_HOME/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$NCCL_HOME/lib/libnccl.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
python -c 'import nccl.core as nccl; nccl.show_versions()'
```

Use the matching `cu12` extra for CUDA 12. The NCCL installation must include
matching device bitcode. Preloading the host library keeps it paired with
that bitcode when a packaged NCCL runtime is also installed. The startup
check compares the host library and bindings versions; it cannot verify the
bitcode's source revision. Version equality alone does not establish that
separate source builds used the same revision, so build all three components
from the same checkout.

## GIN put

Run exactly two MPI ranks on different nodes with one GPU per rank and a
supported GIN transport:

```shell
mpirun -n 2 -N 1 python src/device_api/dsl/gin_put.py ping
mpirun -n 2 -N 1 python src/device_api/dsl/gin_put.py pingpong
mpirun -n 2 -N 1 python src/device_api/dsl/gin_put.py bandwidth -b 1M -e 1G
```

Within a two-node GPU allocation, replace `mpirun -n 2 -N 1` with
`srun --mpi=pmix -N 2 -n 2 --ntasks-per-node=1`. Request GPUs using the site's
allocation options; add `--gpus-per-task=1` if supported.

GIN selects a GPU by node-local MPI rank among the devices visible through
`CUDA_VISIBLE_DEVICES`. When each rank sees only one GPU, it uses visible
device 0 on that rank.

Ping measures put completion time. Ping-pong uses a GIN signal on each put to
order the peer's reply. `-g/--gin-backend` requests `proxy`, `gdaki`, `gpi`,
or `efa-gda` when creating the device communicator. By default, NCCL chooses
the transport. If GIN or the requested backend is unavailable, rank 0 prints
a `SKIP` message and all ranks exit with status 0 before JIT compilation or
benchmark buffer allocation.

Metrics differ from the C++ GIN P2P tests:

- DSL ping reports the full put+flush completion time; C++ divides it by two.
- DSL bandwidth treats size as one total payload split across CTAs; C++ issues
  a full-size operation per active thread.
- DSL ping-pong orders turns with GIN signals; C++ polls the transferred
  payload.

Results are therefore not directly comparable.

GPI context capacity is platform-dependent. On systems with limited NIC
device memory (observed on B300 with SPCX), bandwidth runs with more than two
CTAs can fail during device communicator creation. If this occurs, use `-c 2`;
larger GPI CTA counts are currently unsupported on those systems.

## LSACopy

Run exactly two MPI ranks on distinct GPUs on the same node and in the same
LSA team. For example, select a pair of GPUs connected by NVLink:

```shell
export CUDA_VISIBLE_DEVICES=0,1
mpirun -n 2 python src/device_api/dsl/lsa_copy.py ping
mpirun -n 2 python src/device_api/dsl/lsa_copy.py pingpong
mpirun -n 2 python src/device_api/dsl/lsa_copy.py bandwidth -b 1M -e 1G
```

Within a one-node allocation with two GPUs, use `srun --mpi=pmix -N 1 -n 2`.
Both GPUs must be visible to each rank, or the launcher must assign a distinct
GPU to each rank (for example, with `--gpus-per-task=1` where supported).

LSACopy targets only the remote GPU. Ping includes a system fence per copy;
ping-pong includes LSA barriers. Ping requires one CTA. Ping-pong and bandwidth
can split the payload across multiple CTAs.

## Options

Both scripts share message-size, iteration, and launch options. Pass `--help`
for the complete CLI. Message sizes accept unsigned integer bytes or
case-insensitive `K`, `M`, and `G` suffixes using powers of 1024. Sizes must
be positive, and the maximum must be at least the minimum.

| Option | Ping / ping-pong default | Bandwidth default |
| --- | --- | --- |
| `-b/--min-bytes` | 8 | 1 MiB |
| `-e/--max-bytes` | 32 KiB | 1 GiB |
| `-f/--factor` | 2 | 2 |
| `-n/--iters` | 500 | 200 |
| `-w/--warmup-runs` | 5 | 5 |
| `-r/--repeats` | 5 | 5 |
| `-c/--ctas` | 1 | 16 |

`-s/--coop-scope` selects `thread`, `warp`, or `cta`. GIN defaults to `thread`
for every mode; LSACopy defaults to `thread` for latency and `cta` for
bandwidth. `-t/--threads` defaults to 1, 32, or 256 for these scopes,
respectively. Every message size must be divisible by the CTA count.

For a short smoke test, append `-b 8 -e 4096 -f 8 -n 10 -w 1 -r 2 -c 1`
to any command above. The default bandwidth sweep allocates up to 1 GiB for
GIN and 2 GiB for LSACopy per rank.
