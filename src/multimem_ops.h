/*************************************************************************
 * Copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _MULTIMEM_OPS_H_
#define _MULTIMEM_OPS_H_

#include <cuda_runtime.h>
#include <cassert>

// Multimem operations. Since Multimem is currently only available in PTX here are C++ wrappers around it.
// First template argument is data type, second template type is vectorized data type.
// In the future, the second template type also dictates reduction accuracy

template<typename ptrT, typename valT>
__device__ __forceinline__ valT multimemLoadSum(const ptrT* addr) {
  assert(false);
  // static_assert(std::is_same<ptrT, void>::value, "multimemLoadSum can only be instantiated with implemented types");
  // static_assert(std::is_same<valT, void>::value, "multimemLoadSum can only be instantiated with implemented types");
  return valT{0};
}

#if __CUDA_ARCH__ >= 900  // Hopper and later
template<>
__device__ __forceinline__ double multimemLoadSum<double, double>(const double* addr) {
  const uintptr_t multimem_addr = reinterpret_cast<uintptr_t>(addr);
  double result;
  asm volatile("multimem.ld_reduce.global.add.f64 %0, [%1];" : "=d"(result) : "l"(multimem_addr) : "memory");
  return result;
}
#endif

#if __CUDA_ARCH__ >= 900  // Hopper and later
template<>
__device__ __forceinline__ float multimemLoadSum<float, float>(const float* addr) {
  const uintptr_t multimem_addr = reinterpret_cast<uintptr_t>(addr);
  float result;
  asm volatile("multimem.ld_reduce.global.add.f32 %0, [%1];" : "=f"(result) : "l"(multimem_addr) : "memory");
  return result;
}
#endif

#if __CUDA_ARCH__ >= 900  // Hopper and later
template<>
__device__ __forceinline__ float2 multimemLoadSum<float, float2>(const float* addr) {
  const uintptr_t multimem_addr = reinterpret_cast<uintptr_t>(addr);
  float2 result;
  asm volatile("multimem.ld_reduce.global.add.v2.f32 {%0,  %1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(multimem_addr) : "memory");
  return result;
}
#endif

#if __CUDA_ARCH__ >= 900  // Hopper and later
template<>
__device__ __forceinline__ float4 multimemLoadSum<float, float4>(const float* addr) {
  const uintptr_t multimem_addr = reinterpret_cast<uintptr_t>(addr);
  float4 result;
  asm volatile("multimem.ld_reduce.global.add.v4.f32 {%0,  %1, %2, %3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(multimem_addr) : "memory");
  return result;
}
#endif

template<typename ptrT, typename valT>
__device__ __forceinline__ void multimemStore(ptrT* addr, const valT val) {
  assert(false);
  // static_assert(std::is_same<ptrT, void>::value, "multimemStore can only be instantiated with implemented types");
  // static_assert(std::is_same<valT, void>::value, "multimemStore can only be instantiated with implemented types");
}

#if __CUDA_ARCH__ >= 900  // Hopper and later
template<>
__device__ __forceinline__ void multimemStore<double, double>(double* addr, const double val) {
  const uintptr_t multimem_addr = reinterpret_cast<uintptr_t>(addr);
  asm volatile("multimem.st.global.f64 [%0], %1;" : : "l"(multimem_addr), "d"(val) : "memory");
}
#endif

#if __CUDA_ARCH__ >= 900  // Hopper and later
template<>
__device__ __forceinline__ void multimemStore<float, float>(float* addr, const float val) {
  const uintptr_t multimem_addr = reinterpret_cast<uintptr_t>(addr);
  asm volatile("multimem.st.global.f32 [%0], %1;" : : "l"(multimem_addr), "f"(val) : "memory");
}
#endif

#if __CUDA_ARCH__ >= 900  // Hopper and later
template<>
__device__ __forceinline__ void multimemStore<float, float2>(float* addr, const float2 val) {
  const uintptr_t multimem_addr = reinterpret_cast<uintptr_t>(addr);
  asm volatile("multimem.st.global.v2.f32 [%0], {%1, %2};" : : "l"(multimem_addr), "f"(val.x), "f"(val.y) : "memory");
}
#endif

#if __CUDA_ARCH__ >= 900  // Hopper and later
template<>
__device__ __forceinline__ void multimemStore<float, float4>(float* addr, const float4 val) {
  const uintptr_t multimem_addr = reinterpret_cast<uintptr_t>(addr);
  asm volatile("multimem.st.global.v4.f32 [%0], {%1, %2, %3, %4};" : : "l"(multimem_addr), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w) : "memory");
}
#endif


#endif // _MULTIMEM_OPS_H_
