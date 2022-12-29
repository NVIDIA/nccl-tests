export LD_LIBRARY_PATH=/home/panlichen/work2/ofccl/build/lib
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring
# export NCCL_MAX_NCHANNELS=1
# export NCCL_MIN_NCHANNELS=1
# export NCCL_NTHREADS=64

export DATE=221228
export NCCL_ORDER=1

for MY_NUM_DEV in 2 4 8
do
    unset CUDA_VISIBLE_DEVICES
    if [ $MY_NUM_DEV = 4 ]; then
        export CUDA_VISIBLE_DEVICES=0,1,4,5
    fi
    export RES_DIR=result_${DATE}_${NCCL_ORDER}_${MY_NUM_DEV}cards
    if [ ! -d "$RES_DIR" ]; then 
        mkdir $RES_DIR
    fi

    for n in 5
    do
        for w in  2 
        do
            for m in 1
            do
                for iter in 1
                do
                export RES_PATH="./$RES_DIR/nccl_result_"$iter"_n"$n"_w"$w"_m"$m".txt"
                ## Time
                echo $(date +%F%n%T)>> $RES_PATH
                    for a in 64 128 256 512 1K 2K 4K 8K 16K 32K 64K 128K 256K 512K 1M 2M 4M 8M 16M 32M 64M 128M 256M 512M 1G
                    do
                    ## Test
                    /home/panlichen/work2/nccl-tests/build/all_reduce_perf -b $a -e $a -f 2 -t $MY_NUM_DEV -g 1 -n $n -w $w -c 0 -m $m >> $RES_PATH
                    done
                done 
            done
        done
    done
done
