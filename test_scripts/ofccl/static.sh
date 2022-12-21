g++ statics_ofccl.cpp -o statics_ofccl.out

g++ statics_totalCtx.cpp -o statics_totalCtx.out 
export RES_DIR=test_result_221120_2cards
export OUTPUT_PATH="./$RES_DIR/result_statics_all.txt" 
echo  $(date +%F%n%T)>>$OUTPUT_PATH
for n in 4
do
  for w in 2
  do
    for M in 4
    do
      for iter in 1 2 3   
      do
        export INPUT_PATH="./$RES_DIR/ofccl_result_"$iter"_n"$n"_w"$w"_M"$M".txt"
        ./statics_ofccl.out $INPUT_PATH $OUTPUT_PATH
        ./statics_totalCtx.out $INPUT_PATH $OUTPUT_PATH        
      done 
    done
  done
done 
