#!/usr/bin/env python3
###########################################################################
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE.txt for license information
###########################################################################
"""Measure NCCL GIN put ping latency, ping-pong latency, or bandwidth from CuTe.

Run with exactly two MPI ranks on different nodes. The ping and bandwidth
modes issue puts from rank 0 to rank 1. Ping-pong exchanges puts between both
ranks, uses an attached GIN signal to order each turn, and reports half of the
median round-trip time. JIT compilation happens before timing.
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
FEATURE_MIN_NCCL_VERSION = Version("2.30.7")

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
GIN_TYPES = {
    "proxy": nccl.NcclGinType.PROXY,
    "gdaki": nccl.NcclGinType.GDAKI,
    "gpi": nccl.NcclGinType.GPI,
    "efa-gda": nccl.NcclGinType.EFA_GDA,
}
SIGNAL_OP_INC = 0


@cute.kernel
def gin_put_ping_kernel(
    dev_comm: nccl_cute.DevComm,
    window: nccl_cute.Window,
    size_bytes: cutlass.Int64,
    coop_scope: cutlass.Constexpr[int],
):
    cta, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    if cutlass.const_expr(coop_scope == COOP_SCOPE_THREAD):
        coop = nccl_cute.thread()
    elif cutlass.const_expr(coop_scope == COOP_SCOPE_WARP):
        coop = nccl_cute.warp()
    else:
        coop = nccl_cute.cta()
    world = dev_comm.team_world
    context = cta % cutlass.Int32(dev_comm.gin_context_count)
    gin = dev_comm.gin(
        context,
        resource_sharing_mode=nccl_cute.GinResourceSharingMode.GPU,
    )
    bytes_per_cta = size_bytes // CTAS
    payload = cute.make_layout(bytes_per_cta)
    buffer = window.tensor(
        cutlass.Uint8, payload, offset=cta * bytes_per_cta
    )

    if tidx < coop.size:
        for _ in range(ITERS):
            gin.put(
                world,
                1,
                window,
                buffer,
                window,
                buffer,
                coop,
                is_strong=False,
                given_release=nccl_cute.ThreadScope.THREAD,
                required_release=nccl_cute.ThreadScope.THREAD,
            )
            gin.flush(coop)


@cute.kernel
def gin_put_pingpong_kernel(
    dev_comm: nccl_cute.DevComm,
    window: nccl_cute.Window,
    size_bytes: cutlass.Int64,
    coop_scope: cutlass.Constexpr[int],
):
    cta, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    if cutlass.const_expr(coop_scope == COOP_SCOPE_THREAD):
        coop = nccl_cute.thread()
    elif cutlass.const_expr(coop_scope == COOP_SCOPE_WARP):
        coop = nccl_cute.warp()
    else:
        coop = nccl_cute.cta()
    world = dev_comm.team_world
    context = cta % cutlass.Int32(dev_comm.gin_context_count)
    signal_id = cta
    gin = dev_comm.gin(
        context,
        resource_sharing_mode=nccl_cute.GinResourceSharingMode.GPU,
    )
    bytes_per_cta = size_bytes // CTAS
    payload = cute.make_layout(bytes_per_cta)
    send_offset = cta * bytes_per_cta
    send = window.tensor(cutlass.Uint8, payload, offset=send_offset)
    recv = window.tensor(
        cutlass.Uint8, payload, offset=size_bytes + send_offset
    )

    if tidx < coop.size:
        for iteration in range(ITERS):
            expected = iteration + 1
            if dev_comm.rank == 0:
                gin.put(
                    world,
                    1,
                    window,
                    recv,
                    window,
                    send,
                    coop,
                    is_strong=False,
                    is_signal=True,
                    signal_id=signal_id,
                    signal_op=SIGNAL_OP_INC,
                    signal_op_arg=1,
                    given_release=nccl_cute.ThreadScope.THREAD,
                    required_release=nccl_cute.ThreadScope.THREAD,
                )
                gin.wait_signal(coop, signal=signal_id, least=expected)
            else:
                gin.wait_signal(coop, signal=signal_id, least=expected)
                gin.put(
                    world,
                    0,
                    window,
                    recv,
                    window,
                    send,
                    coop,
                    is_strong=False,
                    is_signal=True,
                    signal_id=signal_id,
                    signal_op=SIGNAL_OP_INC,
                    signal_op_arg=1,
                    given_release=nccl_cute.ThreadScope.THREAD,
                    required_release=nccl_cute.ThreadScope.THREAD,
                )
        gin.flush(coop)
        coop.sync()
        if tidx == 0:
            gin.reset_signal(signal=signal_id)
        coop.sync()


@cute.kernel
def gin_put_bandwidth_kernel(
    dev_comm: nccl_cute.DevComm,
    window: nccl_cute.Window,
    size_bytes: cutlass.Int64,
    coop_scope: cutlass.Constexpr[int],
):
    cta, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    if cutlass.const_expr(coop_scope == COOP_SCOPE_THREAD):
        coop = nccl_cute.thread()
    elif cutlass.const_expr(coop_scope == COOP_SCOPE_WARP):
        coop = nccl_cute.warp()
    else:
        coop = nccl_cute.cta()
    world = dev_comm.team_world
    context = cta % cutlass.Int32(dev_comm.gin_context_count)
    gin = dev_comm.gin(
        context,
        resource_sharing_mode=nccl_cute.GinResourceSharingMode.GPU,
    )
    bytes_per_cta = size_bytes // CTAS
    payload = cute.make_layout(bytes_per_cta)
    buffer = window.tensor(cutlass.Uint8, payload, offset=cta * bytes_per_cta)

    if tidx < coop.size:
        for _ in range(ITERS):
            gin.put(
                world,
                1,
                window,
                buffer,
                window,
                buffer,
                coop,
                is_strong=False,
                given_release=nccl_cute.ThreadScope.THREAD,
                required_release=nccl_cute.ThreadScope.THREAD,
            )
        gin.flush(coop)


@cute.jit
def launch_ping(
    dev_comm: nccl_cute.DevComm,
    window: nccl_cute.Window,
    size_bytes: cutlass.Int64,
    coop_scope: cutlass.Constexpr[int],
):
    gin_put_ping_kernel(
        dev_comm, window, size_bytes, coop_scope
    ).launch(
        grid=[CTAS, 1, 1],
        block=[THREADS, 1, 1],
    )


@cute.jit
def launch_pingpong(
    dev_comm: nccl_cute.DevComm,
    window: nccl_cute.Window,
    size_bytes: cutlass.Int64,
    coop_scope: cutlass.Constexpr[int],
):
    gin_put_pingpong_kernel(
        dev_comm, window, size_bytes, coop_scope
    ).launch(
        grid=[CTAS, 1, 1],
        block=[THREADS, 1, 1],
        cooperative=True,
    )


@cute.jit
def launch_bandwidth(
    dev_comm: nccl_cute.DevComm,
    window: nccl_cute.Window,
    size_bytes: cutlass.Int64,
    coop_scope: cutlass.Constexpr[int],
):
    gin_put_bandwidth_kernel(
        dev_comm, window, size_bytes, coop_scope
    ).launch(
        grid=[CTAS, 1, 1],
        block=[THREADS, 1, 1],
    )


def parse_args() -> argparse.Namespace:
    parser = benchmark_args.create_parser(
        __doc__, ("ping", "pingpong", "bandwidth"), coop_scope="thread"
    )
    parser.add_argument(
        "-g",
        "--gin-backend",
        choices=tuple(GIN_TYPES),
        help="Host-side GIN transport selection (default: let NCCL choose)",
    )
    args = benchmark_args.parse_args(parser)
    benchmark_args.validate_args(parser, args)
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
    fake_comm = nccl_cute.runtime.make_fake_dev_comm()
    fake_window = nccl_cute.runtime.make_fake_window()
    if args.benchmark == "ping":
        launcher = launch_ping
    elif args.benchmark == "pingpong":
        launcher = launch_pingpong
    else:
        launcher = launch_bandwidth
    return cute.compile(
        launcher,
        fake_comm,
        fake_window,
        cutlass.Int64(0),
        COOP_SCOPES[args.coop_scope],
    )


def check_transfer(buffer, rank: int, size_bytes: int, args) -> bool:
    if args.benchmark == "pingpong":
        expected_payload = 2 if rank == 0 else 1
        bytes_per_cta = size_bytes // args.ctas
        indices = []
        for cta in range(args.ctas):
            begin = size_bytes + cta * bytes_per_cta
            indices.extend((begin, begin + bytes_per_cta - 1))
        indices = cp.asarray(indices)
        return bool(cp.all(buffer[indices] == expected_payload).item())
    if rank != 1:
        return True
    bytes_per_cta = size_bytes // args.ctas
    indices = []
    for cta in range(args.ctas):
        indices.extend((cta * bytes_per_cta, (cta + 1) * bytes_per_cta - 1))
    return bool(cp.all(buffer[cp.asarray(indices)] == 1).item())


def run_point(compiled, dev_comm, window, buffer, device, comm, args, size_bytes):
    rank = comm.Get_rank()
    size_bytes_arg = cutlass.Int64(size_bytes)
    participates = args.benchmark == "pingpong" or rank == 0

    if args.benchmark == "pingpong":
        buffer[: 2 * size_bytes].fill(0)
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
    for _ in range(args.repeats):
        comm.Barrier()
        if rank == 0:
            start = cp.cuda.Event()
            stop = cp.cuda.Event()
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
    if args.benchmark == "ping":
        return elapsed_us / args.iters
    if args.benchmark == "pingpong":
        return elapsed_us / (2 * args.iters)
    return size_bytes * args.iters / elapsed_us / 1.0e3


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
    local_rank = local_comm.Get_rank()
    local_comm.Free()
    try:
        ndevices = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as error:
        print(f"ERROR: rank {rank} cannot query CUDA devices: {error}")
        ndevices = 0  # Fail through the allreduce below so every rank returns 1.
    if not comm.allreduce(ndevices > 0, op=MPI.MIN):
        if rank == 0:
            print("ERROR: every rank needs a visible GPU")
        return 1
    device = Device(local_rank % ndevices)
    device.set_current()
    sizes = list(message_sizes(args.min_bytes, args.max_bytes, args.factor))
    unique_id = nccl.get_unique_id() if rank == 0 else None
    unique_id = comm.bcast(unique_id, root=0)

    nccl_comm = None
    window_resource = None
    dev_comm_resource = None
    try:
        nccl_comm = nccl.Communicator.init(nranks=nranks, rank=rank, unique_id=unique_id)
        props = nccl_comm.properties
        skip_reason = None
        if not props.device_api_support or props.gin_type == nccl.NcclGinType.NONE:
            skip_reason = "NCCL GIN device API is unavailable"
        elif args.gin_backend is not None:
            available = props.available_gin_types
            if available is None:
                available = {props.gin_type}
            if GIN_TYPES[args.gin_backend] not in available:
                names = ", ".join(
                    name for name, gin_type in GIN_TYPES.items()
                    if gin_type in available
                )
                skip_reason = (
                    f"requested GIN backend '{args.gin_backend}' is unavailable "
                    f"(available: {names or 'none'})"
                )
        skip_reasons = comm.allgather(skip_reason)
        if any(skip_reasons):
            if rank == 0:
                for skip_rank, reason in enumerate(skip_reasons):
                    if reason is not None:
                        print(f"SKIP: rank {skip_rank}: {reason}")
            return 0

        compiled = compile_kernel(args)
        buffer_bytes = args.max_bytes * (2 if args.benchmark == "pingpong" else 1)
        buffer = nccl.cupy.empty(buffer_bytes, dtype="uint8")
        window_resource = nccl_comm.register_window(buffer)
        requirements = nccl.NCCLDevCommRequirements(
            gin_force_enable=True,
            gin_connection_type=nccl.NcclGinConnectionType.FULL,
            gin_context_count=args.ctas,
            gin_queue_depth=1024,
            gin_signal_count=args.ctas if args.benchmark == "pingpong" else 0,
            gin_strong_signals_required=False,
            gin_va_signals_required=False,
            gin_type=GIN_TYPES.get(args.gin_backend),
        )
        dev_comm_resource = nccl_comm.create_dev_comm(requirements=requirements)
        dev_comm = nccl_cute.DevComm(dev_comm_resource)
        window = nccl_cute.Window(window_resource)

        if rank == 0:
            metric = (
                "bandwidth_GBps"
                if args.benchmark == "bandwidth"
                else "latency_us"
            )
            if args.benchmark in ("ping", "pingpong"):
                print(
                    "NOTE: PING latency is reported as-is; PINGPONG latency "
                    "is the measured round-trip latency divided by 2."
                )
            print(f"size_bytes {metric}")
        for size_bytes in sizes:
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
