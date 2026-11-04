###########################################################################
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE.txt for license information
###########################################################################
"""Shared command-line arguments for CuTe device-API benchmarks."""

import argparse
import re


def parse_size(value: str) -> int:
    """Parse bytes with optional binary K/M/G suffixes, as in the C++ tests."""
    match = re.fullmatch(r"([0-9]+)([KMGkmg]?)", value)
    if match is None:
        raise argparse.ArgumentTypeError(
            f"invalid size {value!r}: expected integer bytes or a K/M/G suffix"
        )
    multiplier = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}
    return int(match[1]) * multiplier[match[2].upper()]


def create_parser(description, benchmarks, coop_scope=None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("benchmark", choices=benchmarks)
    size_help = "size in bytes, optionally suffixed with K/M/G (powers of 1024)"
    parser.add_argument("-b", "--min-bytes", type=parse_size, help=size_help)
    parser.add_argument("-e", "--max-bytes", type=parse_size, help=size_help)
    parser.add_argument("-f", "--factor", type=int, default=2)
    parser.add_argument("-n", "--iters", type=int)
    parser.add_argument("-w", "--warmup-runs", type=int, default=5)
    parser.add_argument("-r", "--repeats", type=int, default=5)
    parser.add_argument("-c", "--ctas", type=int, help="number of CTAs")
    parser.add_argument("-t", "--threads", type=int, help="threads per CTA")
    parser.add_argument(
        "-s", "--coop-scope", choices=("thread", "warp", "cta"), default=coop_scope,
        help=f"cooperative scope (default: {coop_scope})" if coop_scope else
        "cooperative scope (default: thread for latency, cta for bandwidth)",
    )
    return parser


def parse_args(parser) -> argparse.Namespace:
    args = parser.parse_args()
    latency = args.benchmark != "bandwidth"
    if args.coop_scope is None:
        args.coop_scope = "thread" if latency else "cta"
    if args.ctas is None:
        args.ctas = 1 if latency else 16
    if args.threads is None:
        args.threads = {"thread": 1, "warp": 32, "cta": 256}[args.coop_scope]
    if args.min_bytes is None:
        args.min_bytes = 8 if latency else 1024 * 1024
    if args.max_bytes is None:
        args.max_bytes = 32 * 1024 if latency else 1024 * 1024 * 1024
    if args.iters is None:
        args.iters = 500 if latency else 200
    return args


def validate_args(parser, args) -> None:
    if args.min_bytes <= 0 or args.max_bytes < args.min_bytes:
        parser.error("message sizes must be positive and ordered")
    if args.factor < 2:
        parser.error("--factor must be at least 2")
    if args.iters <= 0 or args.warmup_runs < 0 or args.repeats <= 0:
        parser.error("iters/repeats must be positive and warmup-runs nonnegative")
    if args.ctas <= 0 or not 1 <= args.threads <= 1024:
        parser.error("invalid launch geometry")
    if args.coop_scope == "warp" and args.threads < 32:
        parser.error("warp cooperative scope requires at least 32 threads")
