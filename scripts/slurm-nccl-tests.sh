#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/slurm-nccl-tests.sh [-v|--verbose] [srun-options ... --] <nccl-test-binary> [test-args ...]
  srun [srun-options ...] scripts/slurm-nccl-tests.sh [-v|--verbose] <nccl-test-binary> [test-args ...]

Examples:
  scripts/slurm-nccl-tests.sh --ntasks=2 --gpus-per-task=1 -- ./build/all_reduce_perf -b 8 -e 128M -g 1
  srun --kill-on-bad-exit=1 --ntasks=2 --gpus-per-task=1 scripts/slurm-nccl-tests.sh ./build/all_reduce_perf -b 8 -e 128M -g 1
  NCCL_TESTS_COMM_PORT=45678 scripts/slurm-nccl-tests.sh --ntasks-per-node=8 -- ./build/all_reduce_perf -b 8 -e 8G -g 1

Options:
  -v, --verbose  Print NCCL_COMM_ID and SLURM nodelist diagnostics.

Environment:
  NCCL_TESTS_COMM_PORT       Bootstrap port, 1-65535 (default: job/step-derived in 49152-65535, else 29500).
  NCCL_TESTS_COMM_LOCAL_RANK Override GPU assignment; use 0 for task-isolated GPU sets.
  NCCL_TESTS_SLURM_VERBOSE   Set to 1, true, yes, or on to enable verbose output.

Launcher mode runs this wrapper inside srun with --kill-on-bad-exit=1 by default.
Inside a task, an existing NCCL_COMM_ID is preserved; otherwise the bootstrap
host comes from SLURM_STEP_NODELIST, falling back to SLURM_JOB_NODELIST.
A single visible GPU defaults to local rank 0 unless explicitly overridden.
When invoking srun yourself, pass --kill-on-bad-exit=1 to stop peers on failure.
USAGE
}

die() {
  echo "slurm-nccl-tests: $*" >&2
  exit 1
}

verbose=0
case "${NCCL_TESTS_SLURM_VERBOSE:-0}" in
  1|true|TRUE|yes|YES|on|ON) verbose=1 ;;
esac

while [[ ${1:-} == "-v" || ${1:-} == "--verbose" ]]; do
  verbose=1
  shift
done

log() {
  if [[ ${verbose} -eq 1 ]]; then
    echo "slurm-nccl-tests: $*" >&2
  fi
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

inside_srun_task=0
step_id=${SLURM_STEP_ID:-${SLURM_STEPID:-}}
if [[ -n ${SLURM_PROCID:-} && ${step_id} =~ ^[0-9]+$ ]]; then
  inside_srun_task=1
fi

if [[ ${inside_srun_task} -eq 1 ]]; then
  srun_args=()
  if [[ ${1:-} == "--" ]]; then
    shift
  fi
  cmd=("$@")
else
  srun_args=()
  cmd=()
  seen_separator=0
  for arg in "$@"; do
    if [[ ${seen_separator} -eq 0 && ${arg} == "--" ]]; then
      seen_separator=1
      continue
    fi
    if [[ ${seen_separator} -eq 0 ]]; then
      srun_args+=("${arg}")
    else
      cmd+=("${arg}")
    fi
  done

  if [[ ${seen_separator} -eq 0 ]]; then
    cmd=("${srun_args[@]}")
    srun_args=()
  fi
fi

[[ ${#cmd[@]} -gt 0 ]] || { usage >&2; exit 1; }
if [[ ${inside_srun_task} -eq 0 ]]; then
  command -v srun >/dev/null 2>&1 || die "srun was not found in PATH"
  script_path=${BASH_SOURCE[0]}
  [[ ${script_path} == /* ]] || script_path=${PWD}/${script_path}
  wrapper_args=(--)
  if [[ ${verbose} -eq 1 ]]; then
    wrapper_args=(-v --)
  fi
  # Guard the optional array expansion for Bash 3 with set -u.
  exec srun --kill-on-bad-exit=1 --export=ALL ${srun_args[@]+"${srun_args[@]}"} \
    "${script_path}" "${wrapper_args[@]}" "${cmd[@]}"
fi

if [[ -z ${NCCL_COMM_ID:-} ]]; then
  if [[ -n ${NCCL_TESTS_COMM_PORT:-} ]]; then
    port=${NCCL_TESTS_COMM_PORT}
  elif [[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]]; then
    port=$((49152 + (10#${SLURM_JOB_ID} * 31 + 10#${step_id}) % 16384))
  else
    port=29500
  fi
  [[ ${port} =~ ^[0-9]{1,5}$ ]] && ((10#${port} >= 1 && 10#${port} <= 65535)) || \
    die "NCCL_TESTS_COMM_PORT must be in 1-65535, got: ${port}"
  port=$((10#${port}))

  command -v scontrol >/dev/null 2>&1 || die "scontrol was not found in PATH"
  nodelist=${SLURM_STEP_NODELIST:-${SLURM_JOB_NODELIST:-}}
  [[ -n ${nodelist} ]] || die "SLURM_STEP_NODELIST or SLURM_JOB_NODELIST must be set"
  hosts=$(scontrol show hostnames "${nodelist}") || die "could not expand nodelist: ${nodelist}"
  master=${hosts%%$'\n'*}
  [[ -n ${master} ]] || die "could not derive a master host from nodelist: ${nodelist}"
  comm_host=${master}
  if [[ ${comm_host} == *:* && ${comm_host} != \[*\] ]]; then
    comm_host="[${comm_host}]"
  fi
  export NCCL_COMM_ID="${comm_host}:${port}"
  log "nodelist=${nodelist} master=${master}"
fi

# CUDA ordinals are task-relative when Slurm exposes just one GPU.
if [[ -z ${NCCL_TESTS_COMM_LOCAL_RANK+x} && -n ${CUDA_VISIBLE_DEVICES:-} &&
      ${CUDA_VISIBLE_DEVICES} != *,* && ${CUDA_VISIBLE_DEVICES} != -1 ]]; then
  export NCCL_TESTS_COMM_LOCAL_RANK=0
fi

log "NCCL_COMM_ID=${NCCL_COMM_ID}"
exec "${cmd[@]}"
