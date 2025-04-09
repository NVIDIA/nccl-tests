#!/bin/bash

NCCL_BUILD_DIR="/home/test/test01/zhushuang/workspace/nccl-tests/build"

LOG_FILE="/home/test/test01/zhushuang/workspace/nccl-tests/batch_test/test_nccl.log"

exec > >(tee "$LOG_FILE") 2>&1

if [ ! -d "$NCCL_BUILD_DIR" ]; then
    echo "Error: NCCL_BUILD_DIR does not exist."
    exit 1
fi

find "$NCCL_BUILD_DIR" -name "*_perf" | sort | while read -r file; do
    echo "[Processing file: $file]"
    ./build/all_gather_perf -b 8 -e 1M -f 2 -g 8
done

echo "All projects have been test." | tee -a "$LOG_FILE"