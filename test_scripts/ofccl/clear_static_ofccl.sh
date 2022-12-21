g++ clear_static_ofccl.cpp -o clear_static_ofccl.out
g++ clear_static_ofccl_time.cpp -o clear_static_ofccl_time.out

export DATE=221221
export OF_ORDER=1

for cards in 2 4 8
do
  export RES_DIR="test_result_${DATE}_${OF_ORDER}_"$cards"cards"
  export OUTPUT_BW_PATH="./$RES_DIR/result_statics_ofccl_"$cards"cards.txt" 
  export OUTPUT_TIME_PATH="./$RES_DIR/result_statics_ofccl_"$cards"cards_time.txt" 
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
          export INPUT_PATH="./$RES_DIR/ofccl_result_"$iter"_n"$n"_w"$w"_m"$m".txt"
          ./clear_static_ofccl.out $INPUT_PATH $OUTPUT_BW_PATH   $cards 
          ./clear_static_ofccl_time.out $INPUT_PATH $OUTPUT_TIME_PATH   $cards 
        done 
      done
    done
  done 
done