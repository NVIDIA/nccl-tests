clear

cd /home/panlichen/work2/nccl-tests
export LD_LIBRARY_PATH=/home/panlichen/work2/ofccl/build/lib
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring
# export NCCL_MAX_NCHANNELS=1
# export NCCL_MIN_NCHANNELS=1
# export NCCL_NTHREADS=64
export MY_NUM_DEV=2
# export CUDA_VISIBLE_DEVICES=0,1,4,5
export SHOW_ALL_PREPARED_COLL=0
export NITER=4
export NBYTES=8K
export WARMITER=2
export MITER=4

export TRAVERSE_TIMES=10
export TOLERANT_FAIL_CHECK_SQ_CNT=500
export CNT_BEFORE_QUIT=5
export TOLERANT_UNPROGRESSED_CNT=50000
export BASE_CTX_SWITCH_THRESHOLD=100

echo TRAVERSE_TIMES=$TRAVERSE_TIMES
echo TOLERANT_FAIL_CHECK_SQ_CNT=$TOLERANT_FAIL_CHECK_SQ_CNT
echo CNT_BEFORE_QUIT=$CNT_BEFORE_QUIT
echo TOLERANT_UNPROGRESSED_CNT=$TOLERANT_UNPROGRESSED_CNT
echo BASE_CTX_SWITCH_THRESHOLD=$BASE_CTX_SWITCH_THRESHOLD

if [ -z $BINARY ];then
    BINARY="NORMAL"
    BINARY="MS"
fi

if [ "$BINARY" == "NORMAL" ];then
    target="./build/ofccl_all_reduce_perf"
elif [ "$BINARY" == "MS" ];then
    target="./build/ofccl_all_reduce_ms_perf"
    export NITER=200
    export MY_NUM_DEV=8
    export SHOW_ALL_PREPARED_COLL=1
    export WARMITER=0
fi


if [ -z $RUN_TYPE ];then
    RUN_TYPE="PURE"
fi

if [ "$RUN_TYPE" == "PURE" ];then
    cmd="$target -b $NBYTES -e $NBYTES -f 2 -t $MY_NUM_DEV -g 1 -n $NITER -w $WARMITER -c 0 -M $MITER"
elif [ "$RUN_TYPE" == "GDB" ];then
    cmd="cuda-gdb $target"
elif [ "$RUN_TYPE" == "NSYS" ];then
    cmd="nsys profile -f true --trace=cuda,cudnn,cublas,osrt,nvtx -o /home/panlichen/work2/ofccl/log/nsys/$NSYS_FILE $target -b 64M -e 64M -f 2 -t $MY_NUM_DEV -g 1 -n 1 -w 0 -c 0"
fi

echo cmd=$cmd
$cmd
