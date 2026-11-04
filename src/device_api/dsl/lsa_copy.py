#!/usr/bin/env python3
###########################################################################
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE.txt for license information
###########################################################################
"""Measure NCCL LSACopy ping latency, ping-pong latency, or bandwidth from CuTe.

Run exactly two MPI ranks on distinct GPUs in the same LSA team.
"""

import argparse
import statistics
import sys

from cuda.core import Device
from mpi4py import MPI
from packaging.version import Version
import cupy as cp

import cutlass
import cutlass.cute as cute
import nccl.core as nccl
import nccl.core.device.cute as nccl_cute

import benchmark_args


# Oldest public NCCL release that provides every device API this benchmark uses.
FEATURE_MIN_NCCL_VERSION = Version("2.32.3")

# CuTe specializes these values when the benchmark kernel is compiled.
ITERS = 1
CTAS = 1
THREADS = 1

COOP_SCOPE_THREAD = 0
COOP_SCOPE_WARP = 1
COOP_SCOPE_CTA = 2
COOP_SCOPES = {
    "thread": COOP_SCOPE_THREAD,
    "warp": COOP_SCOPE_WARP,
    "cta": COOP_SCOPE_CTA,
}


@cute.kernel
def lsa_copy_kernel(
    dev_comm: nccl_cute.DevComm,
    window: nccl_cute.Window,
    size_bytes: cutlass.Int64,
    coop_scope: cutlass.Constexpr[int],
    benchmark: cutlass.Constexpr[str],
):
    cta, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    if cutlass.const_expr(coop_scope == COOP_SCOPE_THREAD):
        coop = nccl_cute.thread()
    elif cutlass.const_expr(coop_scope == COOP_SCOPE_WARP):
        coop = nccl_cute.warp()
    else:
        coop = nccl_cute.cta()
    peer_lsa = 1 - dev_comm.lsa_rank
    # Destination index 0 maps to lsa_rank - peer_team.rank == peer_lsa.
    peer_team = nccl_cute.Team(
        nRanks=cutlass.Int32(1),
        rank=dev_comm.lsa_rank - peer_lsa,
        stride=cutlass.Int32(1),
    )
    bytes_per_cta = size_bytes // CTAS
    payload = cute.make_layout(bytes_per_cta)
    offset = cta * bytes_per_cta
    src = window.tensor(cutlass.Uint8, payload, offset=offset)
    dst = window.tensor(cutlass.Uint8, payload, offset=size_bytes + offset)

    if cutlass.const_expr(benchmark == "pingpong"):
        if tidx < coop.size:
            # Rank 0 writes the peer's second region and rank 1 writes the
            # third, so validation can check each direction separately.
            peer_dst = window.tensor(
                cutlass.Uint8,
                payload,
                offset=(1 + dev_comm.rank) * size_bytes + offset,
            )
            barrier = nccl_cute.lsa_default(coop, dev_comm, index=cta)
            barrier.sync(coop, nccl_cute.MemoryOrder.ACQ_REL)
            for _ in range(ITERS):
                for turn in range(2):
                    if dev_comm.rank == turn:
                        nccl_cute.lsa_copy(
                            coop, src, window, peer_dst, bytes_per_cta, team=peer_team
                        )
                    # The receiver waits here for the sender's copy, so ping
                    # and pong never overlap.
                    barrier.sync(coop, nccl_cute.MemoryOrder.ACQ_REL)
            # Preserve the barrier epoch across validation, warmup and timing.
            barrier.destroy()
    elif cutlass.const_expr(benchmark == "ping"):
        if tidx < coop.size:
            for _ in range(ITERS):
                nccl_cute.lsa_copy(
                    coop, src, window, dst, bytes_per_cta, team=peer_team
                )
                # Fence on every issuing thread before the next copy.
                cute.arch.fence_acq_rel_sys()
                coop.sync()
    elif cutlass.const_expr(benchmark == "bandwidth"):
        if tidx < coop.size:
            for _ in range(ITERS):
                nccl_cute.lsa_copy(
                    coop, src, window, dst, bytes_per_cta, team=peer_team
                )
            cute.arch.fence_acq_rel_sys()
            coop.sync()
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


@cute.jit
def launch(
    dev_comm: nccl_cute.DevComm,
    window: nccl_cute.Window,
    size_bytes: cutlass.Int64,
    coop_scope: cutlass.Constexpr[int],
    benchmark: cutlass.Constexpr[str],
):
    lsa_copy_kernel(dev_comm, window, size_bytes, coop_scope, benchmark).launch(
        grid=[CTAS, 1, 1],
        block=[THREADS, 1, 1],
        cooperative=(benchmark == "pingpong"),
    )


def parse_args() -> argparse.Namespace:
    parser = benchmark_args.create_parser(__doc__, ("ping", "pingpong", "bandwidth"))
    args = benchmark_args.parse_args(parser)
    benchmark_args.validate_args(parser, args)
    if args.benchmark == "ping" and args.ctas != 1:
        parser.error("ping requires exactly one CTA")
    for size_bytes in message_sizes(args.min_bytes, args.max_bytes, args.factor):
        if size_bytes % args.ctas != 0:
            parser.error(f"{size_bytes} bytes is not divisible by {args.ctas} CTAs")
    return args


def message_sizes(min_bytes: int, max_bytes: int, factor: int):
    size = min_bytes
    while size <= max_bytes:
        yield size
        size *= factor


def compile_kernel(args):
    global ITERS, CTAS, THREADS

    ITERS = args.iters
    CTAS = args.ctas
    THREADS = args.threads
    return cute.compile(
        launch,
        nccl_cute.runtime.make_fake_dev_comm(),
        nccl_cute.runtime.make_fake_window(),
        cutlass.Int64(0),
        COOP_SCOPES[args.coop_scope],
        args.benchmark,
    )


def check_transfer(buffer, rank: int, size_bytes: int, args) -> bool:
    # Check every byte, including the untouched source and local destinations.
    # Chunk comparisons to bound temporary memory for the bandwidth sweep.
    regions = ((0, rank + 1), (size_bytes, 1 if rank == 1 else 0))
    if args.benchmark == "pingpong":
        regions += ((2 * size_bytes, 2 if rank == 0 else 0),)
    for offset, expected in regions:
        for begin in range(0, size_bytes, 1024 * 1024):
            end = min(begin + 1024 * 1024, size_bytes)
            if not bool(cp.all(buffer[offset + begin:offset + end] == expected).item()):
                return False
    return True


def run_point(compiled, dev_comm, window, buffer, device, comm, args, size_bytes):
    rank = comm.Get_rank()
    size_bytes_arg = cutlass.Int64(size_bytes)
    participates = args.benchmark == "pingpong" or rank == 0
    regions = buffer.size // args.max_bytes
    buffer[:regions * size_bytes].fill(0)
    buffer[:size_bytes].fill(rank + 1)
    device.sync()
    comm.Barrier()
    if participates:
        compiled(dev_comm, window, size_bytes_arg)
    device.sync()
    comm.Barrier()
    valid = int(check_transfer(buffer, rank, size_bytes, args))
    if not comm.allreduce(valid, op=MPI.MIN):
        raise RuntimeError(f"transfer validation failed for {size_bytes} bytes")

    for _ in range(args.warmup_runs):
        comm.Barrier()
        if participates:
            compiled(dev_comm, window, size_bytes_arg)
        device.sync()
    comm.Barrier()

    samples_us = []
    start = cp.cuda.Event() if rank == 0 else None
    stop = cp.cuda.Event() if rank == 0 else None
    for _ in range(args.repeats):
        comm.Barrier()
        if rank == 0:
            start.record()
            compiled(dev_comm, window, size_bytes_arg)
            stop.record()
            stop.synchronize()
            samples_us.append(float(cp.cuda.get_elapsed_time(start, stop)) * 1000.0)
        elif participates:
            compiled(dev_comm, window, size_bytes_arg)
            device.sync()
        comm.Barrier()

    if rank != 0:
        return None
    elapsed_us = statistics.median(samples_us)
    if args.benchmark == "bandwidth":
        return size_bytes * args.iters / elapsed_us / 1.0e3
    return elapsed_us / (args.iters * (2 if args.benchmark == "pingpong" else 1))


def main() -> int:
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()
    if nranks != 2:
        if rank == 0:
            print(f"ERROR: this benchmark requires exactly two ranks, got {nranks}")
        return 1

    versions = nccl.get_version()
    libnccl = versions.libnccl
    nccl_ok = False
    if libnccl is None:
        print(f"ERROR: rank {rank} has no loaded NCCL library")
    elif libnccl.version != versions.nccl_bindings:
        print(
            f"ERROR: rank {rank} needs NCCL {versions.nccl_bindings} to match "
            f"the NCCL4Py bindings, found {libnccl.version}. Install NCCL4Py "
            "and NCCL from the same release or source revision."
        )
    elif libnccl.version < FEATURE_MIN_NCCL_VERSION:
        print(
            f"ERROR: rank {rank} needs NCCL {FEATURE_MIN_NCCL_VERSION} or newer "
            f"for this benchmark, found {libnccl.version}"
        )
    else:
        nccl_ok = True
    if not comm.allreduce(nccl_ok, op=MPI.MIN):
        return 1

    local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    same_node = local_comm.Get_size() == nranks
    local_comm.Free()
    if not comm.allreduce(same_node, op=MPI.MIN):
        if rank == 0:
            print("ERROR: both ranks must run on the same node")
        return 1
    try:
        ndevices = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as error:
        print(f"ERROR: rank {rank} cannot query CUDA devices: {error}")
        ndevices = 0  # Fail through the allreduce below so every rank returns 1.
    if not comm.allreduce(ndevices > 0, op=MPI.MIN):
        if rank == 0:
            print("ERROR: every rank needs a visible GPU")
        return 1
    device = Device(rank % ndevices)
    device.set_current()
    # PCI identities remain comparable when Slurm masks each rank to one GPU.
    gpu_ids = comm.allgather(cp.cuda.runtime.deviceGetPCIBusId(device.device_id))
    if len(set(gpu_ids)) != nranks:
        if rank == 0:
            print("ERROR: each rank must use a distinct GPU")
        return 1

    compiled = compile_kernel(args)
    unique_id = nccl.get_unique_id() if rank == 0 else None
    unique_id = comm.bcast(unique_id, root=0)
    nccl_comm = None
    window_resource = None
    dev_comm_resource = None
    try:
        nccl_comm = nccl.Communicator.init(nranks=nranks, rank=rank, unique_id=unique_id)
        supported = nccl_comm.device_api_support and nccl_comm.team_lsa.n_ranks == nranks
        if not comm.allreduce(supported, op=MPI.MIN):
            if rank == 0:
                print("ERROR: both GPUs must support the device API and share one LSA team")
            return 1

        regions = 3 if args.benchmark == "pingpong" else 2
        buffer = nccl.cupy.empty(regions * args.max_bytes, dtype="uint8")
        window_resource = nccl_comm.register_window(buffer)
        requirements = nccl.NCCLDevCommRequirements(
            lsa_barrier_count=args.ctas if args.benchmark == "pingpong" else 0,
        )
        dev_comm_resource = nccl_comm.create_dev_comm(requirements=requirements)
        dev_comm = nccl_cute.DevComm(dev_comm_resource)
        window = nccl_cute.Window(window_resource)

        if rank == 0:
            print("# NOTE: LSACopy targets only the remote GPU using a singleton team.")
            if args.benchmark == "ping":
                print("# NOTE: ping reports copy + system fence time per iteration; no /2.")
            elif args.benchmark == "pingpong":
                print("# NOTE: pingpong reports half a round trip, including LSA barriers.")
            print(f"# ctas={args.ctas} threads={args.threads} coop_scope={args.coop_scope}")
            metric = "bandwidth_GBps" if args.benchmark == "bandwidth" else "latency_us"
            print(f"size_bytes {metric}")
        for size_bytes in message_sizes(args.min_bytes, args.max_bytes, args.factor):
            value = run_point(
                compiled, dev_comm, window, buffer, device, comm, args, size_bytes
            )
            if rank == 0:
                print(f"{size_bytes} {value:.3f}", flush=True)
        return 0
    finally:
        if dev_comm_resource is not None:
            dev_comm_resource.close()
        if window_resource is not None:
            window_resource.close()
        if nccl_comm is not None:
            nccl_comm.destroy()


if __name__ == "__main__":
    sys.exit(main())
