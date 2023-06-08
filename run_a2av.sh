#!/bin/bash

cp $1 ./alltoallv_param.csv
./build/alltoallv2_perf ${@:2}
