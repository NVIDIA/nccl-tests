export LD_LIBRARY_PATH=/home/panlichen/work2/ofccl/build/lib
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring

export DATE=221221
export OF_ORDER=1

export TRAVERSE_TIMES=10
export TOLERANT_UNPROGRESSED_CNT=10000
export BASE_CTX_SWITCH_THRESHOLD=80
export BOUNS_SWITCH_4_PROCESSED_COLL=0
export DEV_TRY_ROUND=10

# export SHOW_ALL_PREPARED_COLL=1

for MY_NUM_DEV in 2 4 8
do
    unset CUDA_VISIBLE_DEVICES
    if [ $MY_NUM_DEV = 4 ]; then
        export CUDA_VISIBLE_DEVICES=0,1,4,5
    fi
    export RES_DIR=test_result_${DATE}_${OF_ORDER}_${MY_NUM_DEV}cards
    if [ ! -d "$RES_DIR" ]; then 
        mkdir $RES_DIR
    fi

    for n in 8
    do
        for w in  2 
        do
            for m in 1
            do
                for iter in 1 2 3 
                do
                export RES_PATH="./$RES_DIR/ofccl_result_"$iter"_n"$n"_w"$w"_m"$m".txt"
                ## Time
                echo $(date +%F%n%T)>> $RES_PATH
                    for a in 64 128 256 512 1K 2K 4K 8K 16K 32K 64K 128K 256K 512K 1M 2M 4M 8M 16M 32M 64M 128M 256M 512M 1G
                    do
                    ## Test
                    /home/panlichen/work2/nccl-tests/build/ofccl_all_reduce_perf -b $a -e $a -f 2 -t $MY_NUM_DEV -g 1 -n $n -w $w -c 0 -M $m >> $RES_PATH
                    done
                done 
            done
        done
    done
done
