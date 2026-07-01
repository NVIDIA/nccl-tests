#!/usr/bin/env python3
###########################################################################
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
###########################################################################
"""Analyze nccl-tests per-iteration JSON output (-I 1 with -J).

Usage:
    python3 analyze_perf_json.py results.json
    python3 analyze_perf_json.py results.json --sizes 8,1048576,34359738368
    python3 analyze_perf_json.py results.json --straggler
    python3 analyze_perf_json.py results.json --spikes --threshold 1.5
    python3 analyze_perf_json.py results.json --in-place --spikes
    python3 analyze_perf_json.py results.json --iter0
    python3 analyze_perf_json.py results.json --nodes
    python3 analyze_perf_json.py results.json --nodes --procs-per-node 4
    python3 analyze_perf_json.py results.json --all
"""

import json
import argparse
import statistics

PROCESS_TIMES_KEY = "per_process_max_times_us"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fmt_size(size):
    if size == 0:
        return "0 B"
    for unit, thresh in [("GB", 1<<30), ("MB", 1<<20), ("KB", 1<<10)]:
        if size >= thresh:
            val = size / thresh
            return f"{val:.0f} {unit}" if val == int(val) else f"{val:.1f} {unit}"
    return f"{size} B"


def get_process_times(per_iter):
    return per_iter.get(PROCESS_TIMES_KEY, [])


def process_max_times(process_times):
    if not process_times:
        return []
    n_iters = min(len(times) for times in process_times)
    return [max(times[i] for times in process_times) for i in range(n_iters)]


def placement_fields(in_place):
    if in_place:
        return "in_place_per_iter", "IP", "in-place"
    return "out_of_place_per_iter", "OOP", "out-of-place"


def print_overview(data, placement_key, placement_name):
    print("=" * 80)
    print("OVERVIEW")
    print("=" * 80)
    print(f"  Version:    {data.get('version')}")
    print(f"  Results:    {len(data.get('results', []))}")
    cfg = data.get('config', {})
    print(f"  Iterations: {cfg.get('iterations')}  (agg: {cfg.get('aggregated_iterations', 1)})")
    if cfg.get('per_iter_skip', 0):
        print(f"  Summary skip: {cfg['per_iter_skip']} leading iterations")
    print(f"  Placement:  {placement_name}")
    r0 = data['results'][0] if data.get('results') else {}
    per_iter = r0.get(placement_key, {})
    process_times = get_process_times(per_iter)
    print(f"  Process rows: {len(process_times)}")
    if process_times:
        print(f"  Timing key:   {PROCESS_TIMES_KEY}")
        print("  Row scope:    process-local max across local threads/GPUs")
        print("  Summary:      times_us is max across process rows per iteration")
    print()


def print_stats_table(data, placement_key, placement_label, sizes=None):
    print("=" * 80)
    print(f"PER-ITERATION STATS ({placement_label})")
    print("=" * 80)
    print(f"{'Size':>14}  {'avg (us)':>10}  {'i_min':>10}  {'i_max':>10}  {'p99':>10}  {'cv%':>7}  {'spike':>6}")
    print("-" * 80)
    for r in data['results']:
        size = r['size']
        if sizes and size not in sizes:
            continue
        per_iter = r.get(placement_key, {})
        if not per_iter:
            continue
        spike = per_iter['max_us'] / per_iter['avg_us'] if per_iter['avg_us'] > 0 else 0
        flag = " ***" if per_iter['cv_pct'] > 5 or spike > 1.5 else ""
        print(f"{size:14d}  {per_iter['avg_us']:10.1f}  {per_iter['min_us']:10.1f}  {per_iter['max_us']:10.1f}  {per_iter['p99_us']:10.1f}  {per_iter['cv_pct']:6.1f}%  {spike:5.1f}x{flag}")
    print()


def print_straggler(data, placement_key, placement_label, sizes=None):
    first = data['results'][0] if data.get('results') else {}
    nprocs = len(get_process_times(first.get(placement_key, {})))
    if nprocs == 0:
        print("No per-process data available.\n")
        return

    print("=" * 80)
    print(f"PER-PROCESS STRAGGLER ANALYSIS ({placement_label})")
    print("=" * 80)
    for r in data['results']:
        size = r['size']
        if sizes and size not in sizes:
            continue
        process_times = get_process_times(r.get(placement_key, {}))
        if not process_times:
            continue

        process_stats = []
        for pi, times in enumerate(process_times):
            avg = sum(times) / len(times)
            std = statistics.stdev(times) if len(times) > 1 else 0
            cv = (std / avg * 100) if avg > 0 else 0
            process_stats.append((pi, avg, min(times), max(times), cv))

        process_stats.sort(key=lambda x: x[1])
        fastest = process_stats[0]
        slowest = process_stats[-1]
        noisiest = max(process_stats, key=lambda x: x[4])
        spread = slowest[1] - fastest[1]

        print(f"\n{fmt_size(size):>10}:  fastest=process {fastest[0]:2d} ({fastest[1]:.1f}us)"
              f"  slowest=process {slowest[0]:2d} ({slowest[1]:.1f}us)"
              f"  spread={spread:.1f}us ({spread/fastest[1]*100:.1f}%)")
        worst = max(process_stats, key=lambda x: x[3])
        print(f"{'':>10}   noisiest=process {noisiest[0]:2d} (CV={noisiest[4]:.1f}%)"
              f"  worst i_max: process {worst[0]} ({worst[3]:.1f}us)")
    print()


def print_spikes(data, placement_key, placement_label, threshold=1.5, sizes=None):
    first = data['results'][0] if data.get('results') else {}
    nprocs = len(get_process_times(first.get(placement_key, {})))
    if nprocs == 0:
        print("No per-process data available.\n")
        return

    print("=" * 80)
    print(f"SPIKE ANALYSIS (>{threshold}x avg, {placement_label})")
    print("=" * 80)
    found = False
    for r in data['results']:
        size = r['size']
        if sizes and size not in sizes:
            continue
        per_iter = r.get(placement_key, {})
        if not per_iter or per_iter['max_us'] / per_iter['avg_us'] < threshold:
            continue
        process_times = get_process_times(per_iter)
        if not process_times:
            continue

        avg = per_iter['avg_us']
        spike_iters = {}
        for pi in range(len(process_times)):
            for i, t in enumerate(process_times[pi]):
                if t > avg * threshold:
                    if i not in spike_iters:
                        spike_iters[i] = []
                    spike_iters[i].append((pi, t))

        if spike_iters:
            found = True
            print(f"\n{fmt_size(size)} (avg={avg:.1f}, max={per_iter['max_us']:.1f}, CV={per_iter['cv_pct']:.1f}%):")
            for si in sorted(spike_iters.keys()):
                entries = spike_iters[si]
                vals = [process_times[pi][si] for pi in range(len(process_times))]
                n_high = len(entries)
                line = f"  iter {si:2d}: {n_high:2d}/{nprocs} processes spiked  max={max(vals):.1f}"
                if n_high == nprocs:
                    line += "  [SYSTEM-WIDE]"
                elif n_high == 1:
                    line += f"  [STRAGGLER: process {entries[0][0]}]"
                print(line)
                if n_high < nprocs and n_high <= 8:
                    print(f"           processes: {[e[0] for e in entries]}")

    if not found:
        print(f"\nNo iterations exceeded {threshold}x average.")
    print()


def print_iter0(data, placement_key, placement_label, sizes=None):
    print("=" * 80)
    print(f"ITERATION 0 WARMUP IMPACT (RAW, {placement_label})")
    print("=" * 80)
    print(f"{'Size':>14}  {'iter0 (us)':>10}  {'iter1+ avg':>10}  {'spike':>6}  {'CV w/ iter0':>11}  {'CV w/o':>8}")
    print("-" * 80)
    for r in data['results']:
        if sizes and r['size'] not in sizes:
            continue
        per_iter = r.get(placement_key, {})
        times = per_iter.get('times_us', [])
        if not times or len(times) < 2:
            continue

        iter0 = times[0]
        steady = times[1:]
        steady_avg = sum(steady) / len(steady)
        ratio = iter0 / steady_avg if steady_avg > 0 else 0

        all_avg = sum(times) / len(times)
        all_std = statistics.stdev(times) if len(times) > 1 else 0
        all_cv = (all_std / all_avg * 100) if all_avg > 0 else 0

        steady_std = statistics.stdev(steady) if len(steady) > 1 else 0
        steady_cv = (steady_std / steady_avg * 100) if steady_avg > 0 else 0

        flag = " ***" if ratio > 1.5 else ""
        print(f"{r['size']:14d}  {iter0:10.1f}  {steady_avg:10.1f}  {ratio:5.1f}x  {all_cv:10.1f}%  {steady_cv:7.1f}%{flag}")
    print()


def node_groups(data, nprocs, rows_per_node):
    devices = data.get('config', {}).get('devices', [])
    if len(devices) >= nprocs and all('hostname' in dev for dev in devices[:nprocs]):
        groups = []
        by_host = {}
        for proc, dev in enumerate(devices[:nprocs]):
            host = dev.get('hostname', f"node {len(groups)}")
            if host not in by_host:
                by_host[host] = []
                groups.append((host, by_host[host]))
            by_host[host].append(proc)
        return groups

    groups = []
    for start in range(0, nprocs, rows_per_node):
        end = min(start + rows_per_node, nprocs)
        groups.append((f"node {len(groups)}", list(range(start, end))))
    return groups


def print_nodes(data, placement_key, placement_label, rows_per_node=4, sizes=None):
    first = data['results'][0] if data.get('results') else {}
    nprocs = len(get_process_times(first.get(placement_key, {})))
    if nprocs == 0:
        print("No per-process data available.\n")
        return
    groups = node_groups(data, nprocs, rows_per_node)

    print("=" * 80)
    print(f"PER-NODE ANALYSIS ({len(groups)} nodes, process-max rows, {placement_label})")
    print("=" * 80)
    for r in data['results']:
        size = r['size']
        if sizes and size not in sizes:
            continue
        process_times = get_process_times(r.get(placement_key, {}))
        if not process_times:
            continue

        print(f"\n{fmt_size(size):>10}:")
        for node, (label, node_procs) in enumerate(groups):
            node_avgs = [sum(process_times[pi]) / len(process_times[pi]) for pi in node_procs]
            node_maxes = [max(process_times[pi]) for pi in node_procs]
            process_cvs = []
            for pi in node_procs:
                times = process_times[pi]
                avg = sum(times) / len(times)
                std = statistics.stdev(times) if len(times) > 1 else 0
                process_cvs.append((pi, (std / avg * 100) if avg > 0 else 0))
            worst_process = max(process_cvs, key=lambda x: x[1])
            proc_range = f"{node_procs[0]:2d}" if len(node_procs) == 1 else f"{node_procs[0]:2d}-{node_procs[-1]:2d}"
            print(f"  Node {node} ({label}, processes {proc_range}):"
                  f"  avg={sum(node_avgs)/len(node_avgs):.1f}us"
                  f"  range=[{min(node_avgs):.1f}-{max(node_avgs):.1f}]"
                  f"  max={max(node_maxes):.1f}"
                  f"  noisiest=process {worst_process[0]} CV={worst_process[1]:.1f}%")
    print()


def print_consistency(data, placement_key, placement_label):
    mismatches = 0
    total = 0
    for r in data['results']:
        per_iter = r.get(placement_key, {})
        process_times = get_process_times(per_iter)
        t = per_iter.get('times_us', [])
        if process_times and t:
            total += 1
            expected = process_max_times(process_times)
            mismatch = len(t) != len(expected) or any(abs(a - b) >= 0.001
                                                     for a, b in zip(t, expected))
            if mismatch:
                mismatches += 1
    if total == 0:
        status = "N/A (no process rows)"
    else:
        status = "PASS" if mismatches == 0 else f"FAIL ({mismatches}/{total})"
    print(f"Consistency (times_us == max over process rows, {placement_label}): {status}")
    print()


def default_sizes(data):
    all_sizes = [r['size'] for r in data.get('results', [])]
    if not all_sizes:
        return []
    targets = [0, 8, 1024, 131072, 8388608, 134217728, 1073741824, 34359738368]
    return [s for s in targets if s in all_sizes] or all_sizes


def main():
    parser = argparse.ArgumentParser(description="Analyze nccl-tests per-iteration JSON output")
    parser.add_argument("json_file", help="Path to JSON output file")
    parser.add_argument("--sizes", help="Comma-separated list of sizes to focus on (default: auto-select)")
    parser.add_argument("--straggler", action="store_true", help="Per-process straggler analysis")
    parser.add_argument("--spikes", action="store_true", help="Spike detection and correlation")
    parser.add_argument("--threshold", type=float, default=1.5, help="Spike threshold as multiple of avg (default: 1.5)")
    parser.add_argument("--in-place", action="store_true", help="Analyze in-place results instead of out-of-place")
    parser.add_argument("--iter0", action="store_true", help="Iteration 0 warmup impact analysis")
    parser.add_argument("--nodes", action="store_true", help="Per-node analysis")
    parser.add_argument("--procs-per-node", type=int, default=4,
                        help="Fallback process rows per node when hostnames are unavailable (default: 4)")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    args = parser.parse_args()

    data = load_json(args.json_file)

    sizes = None
    if args.sizes:
        sizes = set(int(s) for s in args.sizes.split(","))

    run_all = args.all or not any([args.straggler, args.spikes, args.iter0, args.nodes])
    placement_key, placement_label, placement_name = placement_fields(args.in_place)

    print_overview(data, placement_key, placement_name)
    print_consistency(data, placement_key, placement_label)

    if run_all or not any([args.straggler, args.spikes, args.iter0, args.nodes]):
        print_stats_table(data, placement_key, placement_label, sizes)

    if args.straggler or args.all:
        focus = sizes or set(default_sizes(data))
        print_straggler(data, placement_key, placement_label, focus)

    if args.spikes or args.all:
        print_spikes(data, placement_key, placement_label, args.threshold, sizes)

    if args.iter0 or args.all:
        print_iter0(data, placement_key, placement_label, sizes)

    if args.nodes or args.all:
        focus = sizes or set(default_sizes(data))
        print_nodes(data, placement_key, placement_label, args.procs_per_node, focus)


if __name__ == "__main__":
    main()
