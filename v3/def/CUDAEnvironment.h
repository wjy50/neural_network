/**
 * Created by wjy50 on 18-5-18.
 */

#ifndef NEURAL_NETWORK_CUDAENVIRONMENT_H
#define NEURAL_NETWORK_CUDAENVIRONMENT_H

#define ENABLE_CUDA 1

#if ENABLE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifdef __JETBRAINS_IDE__

#define __CUDACC__ 1
#define __host__
#define __device__
#define __global__
#define __forceinline__
#define __shared__
inline void __syncthreads() {}
inline void __threadfence_block() {}
inline void __threadfence() {}
template<class T> inline T __clz(const T val) { return val; }
struct __cuda_fake_struct { int x; int y; };
extern __cuda_fake_struct blockDim;
extern __cuda_fake_struct threadIdx;
extern __cuda_fake_struct blockIdx;

#define CU_MAX std::max

#define CU_MIN std::min

#else

#define CU_MAX max

#define CU_MIN min

#endif

#endif //ENABLE_CUDA

#endif //NEURAL_NETWORK_CUDAENVIRONMENT_H
