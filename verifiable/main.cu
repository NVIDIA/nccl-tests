
#include <cuda_runtime.h>
#include <iostream>

#define NCCL_VERIFIABLE_SELF_TEST 1
#include "verifiable.h"

int main(int arg_n, char **args) {
  std::cerr<<"You are hoping to see no output beyond this line."<<std::endl;
  cudaSetDevice(0);
  ncclVerifiableLaunchSelfTest();
  cudaDeviceSynchronize();
  return 0;
}
