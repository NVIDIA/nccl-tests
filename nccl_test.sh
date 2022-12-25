clear

cd /home/panlichen/work2/nccl-tests
export LD_LIBRARY_PATH=/home/panlichen/work2/ofccl/build/lib
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring
# export NCCL_MAX_NCHANNELS=1
# export NCCL_MIN_NCHANNELS=1
# export NCCL_NTHREADS=64

if [ -z $BINARY ];then
    BINARY="DEBUG"
    # BINARY="MS"
    # BINARY="PERF"
fi

if [ "$BINARY" == "DEBUG" ];then
    target="./build/all_reduce_perf"
    export MY_NUM_DEV=2
    if [ $MY_NUM_DEV = 4 ]; then
        export CUDA_VISIBLE_DEVICES=0,1,4,5
    fi
    export SHOW_ALL_PREPARED_COLL=0
    export NITER=5
    export NBYTES=4K
    export WARMITER=2
    export MITER=1
    export CHECK=0
elif [ "$BINARY" == "PERF" ];then
    target="./build/all_reduce_perf"
    export MY_NUM_DEV=8
    if [ $MY_NUM_DEV = 4 ]; then
        export CUDA_VISIBLE_DEVICES=0,1,4,5
    fi
    export SHOW_ALL_PREPARED_COLL=0
    export NITER=4
    export NBYTES=8K
    export WARMITER=2
    export MITER=4
    export CHECK=0
elif [ "$BINARY" == "MS" ];then
    export MY_NUM_DEV=8
    # target="./build/ofccl_all_reduce_ms_perf"
    if [ $MY_NUM_DEV = 4 ]; then
        export CUDA_VISIBLE_DEVICES=0,1,4,5
    fi
    # export NITER=200
    # export SHOW_ALL_PREPARED_COLL=1
    # export WARMITER=0
    # export NBYTES=8K
    # export MITER=4
fi

export NSYS_FILE="nccl"
export NCU_FILE="nccl"

if [ -z $RUN_TYPE ];then
    RUN_TYPE="PURE"
    # RUN_TYPE="GDB"
    # RUN_TYPE="NSYS"
    # RUN_TYPE="NCU"
fi

if [ "$RUN_TYPE" == "PURE" ];then
    cmd="$target -b $NBYTES -e $NBYTES -f 2 -t $MY_NUM_DEV -g 1 -n $NITER -w $WARMITER -c $CHECK -m $MITER"
elif [ "$RUN_TYPE" == "GDB" ];then
    cmd="cuda-gdb $target"
    # set args -b 8M -e 8M -f 2 -t 2 -g 1 -n 1 -w 0 -c 0
elif [ "$RUN_TYPE" == "NSYS" ];then
    cmd="nsys profile -f true --trace=cuda,cudnn,cublas,osrt,nvtx -o /home/panlichen/work2/ofccl/log/nsys/$NSYS_FILE $target -b $NBYTES -e $NBYTES -f 2 -t $MY_NUM_DEV -g 1 -n $NITER -w $WARMITER -c $CHECK -m $MITER"
elif [ "$RUN_TYPE" == "NCU" ];then
    # cmd="ncu --nvtx -f -o /home/panlichen/work2/ofccl/log/nsys/$NCU_FILE $target -b $NBYTES -e $NBYTES -f 2 -t $MY_NUM_DEV -g 1 -n $NITER -w $WARMITER -c $CHECK -m $MITER"
    cmd="ncu $target -b $NBYTES -e $NBYTES -f 2 -t $MY_NUM_DEV -g 1 -n $NITER -w $WARMITER -c $CHECK -m $MITER"
fi

echo cmd=$cmd
$cmd #> /home/panlichen/work2/ofccl/log/ofccl-2ms-coll-master.log

