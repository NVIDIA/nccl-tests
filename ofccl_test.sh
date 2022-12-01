clear

cd /home/panlichen/work2/nccl-tests
export LD_LIBRARY_PATH=/home/panlichen/work2/ofccl/build/lib
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring
# export NCCL_MAX_NCHANNELS=1
# export NCCL_MIN_NCHANNELS=1
# export NCCL_NTHREADS=64

export CHECK=0

export TRAVERSE_TIMES=10
export TOLERANT_UNPROGRESSED_CNT=8000
export BASE_CTX_SWITCH_THRESHOLD=80

# export ENABLE_VQ=1
# export TOLERANT_FAIL_CHECK_SQ_CNT=5000
# export CNT_BEFORE_QUIT=5

echo TRAVERSE_TIMES=$TRAVERSE_TIMES
echo TOLERANT_UNPROGRESSED_CNT=$TOLERANT_UNPROGRESSED_CNT
echo BASE_CTX_SWITCH_THRESHOLD=$BASE_CTX_SWITCH_THRESHOLD

if [ ! -z $BINARY ];then
    echo TOLERANT_FAIL_CHECK_SQ_CNT=$TOLERANT_FAIL_CHECK_SQ_CNT
    echo CNT_BEFORE_QUIT=$CNT_BEFORE_QUIT
fi

if [ -z $BINARY ];then
    BINARY="DEBUG"
    BINARY="MS"
    # BINARY="PERF"
fi

if [ "$BINARY" == "DEBUG" ];then
    target="./build/ofccl_all_reduce_perf"
    export MY_NUM_DEV=8
    # export CUDA_VISIBLE_DEVICES=0,1,4,5
    export SHOW_ALL_PREPARED_COLL=1
    export NITER=4
    export NBYTES=8K
    export WARMITER=2
    export MITER=4
elif [ "$BINARY" == "PERF" ];then
    target="./build/ofccl_all_reduce_perf"
    export MY_NUM_DEV=8
    # export CUDA_VISIBLE_DEVICES=0,1,4,5
    export SHOW_ALL_PREPARED_COLL=0
    export NITER=4
    export NBYTES=8K
    export WARMITER=2
    export MITER=4
elif [ "$BINARY" == "MS" ];then
    target="./build/ofccl_all_reduce_ms_perf"
    export MY_NUM_DEV=8
    # export CUDA_VISIBLE_DEVICES=0,1,4,5
    export NITER=200
    export SHOW_ALL_PREPARED_COLL=1
    export WARMITER=0
    export NBYTES=8K
    export MITER=4
fi


if [ -z $RUN_TYPE ];then
    RUN_TYPE="PURE"
fi

if [ "$RUN_TYPE" == "PURE" ];then
    cmd="$target -b $NBYTES -e $NBYTES -f 2 -t $MY_NUM_DEV -g 1 -n $NITER -w $WARMITER -c $CHECK -M $MITER"
elif [ "$RUN_TYPE" == "GDB" ];then
    cmd="cuda-gdb $target"
elif [ "$RUN_TYPE" == "NSYS" ];then
    cmd="nsys profile -f true --trace=cuda,cudnn,cublas,osrt,nvtx -o /home/panlichen/work2/ofccl/log/nsys/$NSYS_FILE $target -b 64M -e 64M -f 2 -t $MY_NUM_DEV -g 1 -n 1 -w 0 -c 0"
fi

echo cmd=$cmd
$cmd #> /home/panlichen/work2/ofccl/log/ofccl.log

