g++ static_nccl.cpp -o static_nccl.out
g++ static_time.cpp -o static_time.out

export DATE=221221
export NCCL_ORDER=1

for cards in 2 4 8
do
  export RES_DIR="test_result_${DATE}_${NCCL_ORDER}_"$cards"cards"
  export OUTPUT_BW_PATH="./$RES_DIR/result_statics_nccl_"$cards"cards.txt" 
  export OUTPUT_TIME_PATH="./$RES_DIR/result_statics_nccl_"$cards"cards_time.txt" 
  echo  $(date +%F%n%T)>>$OUTPUT_BW_PATH
  echo  $(date +%F%n%T)>>$OUTPUT_TIME_PATH
  for n in 32
  do
    for w in  2 
    do
    for m in 1
      do
        for iter in 1 2 3    
        do
          export INPUT_PATH="./$RES_DIR/nccl_result_"$iter"_n"$n"_w"$w"_m"$m".txt"
          ./static_nccl.out $INPUT_PATH $OUTPUT_BW_PATH   $cards 
          ./static_time.out $INPUT_PATH $OUTPUT_TIME_PATH   $cards 
        done 
      done
    done
  done 
done