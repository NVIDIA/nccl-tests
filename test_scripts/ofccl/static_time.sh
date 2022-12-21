g++ static_time.cpp -o static_time.out


export RES_DIR=test_result_221119
export OUTPUT_PATH="./$RES_DIR/result_statics_all_time.txt" 
echo  $(date +%F%n%T)>>$OUTPUT_PATH
for n in 4
do
  for w in 2
  do
    for M in 4
    do
      for iter in 1 2 3   
      do
        export INPUT_PATH="./$RES_DIR/test_result_"$iter"_n"$n"_w"$w"_M"$M".txt"
        ./static_time.out $INPUT_PATH $OUTPUT_PATH
     
      done 
    done
  done
done 
