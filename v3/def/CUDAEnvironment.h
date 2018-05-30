/**
 * Created by wjy50 on 18-5-18.
 */

#ifndef NEURAL_NETWORK_CUDAENVIRONMENT_H
#define NEURAL_NETWORK_CUDAENVIRONMENT_H

#define ENABLE_CUDA 0

#if ENABLE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifdef __JETBRAINS_IDE__

#define __host__
#define __device__
#define __shared__
#define __constant__
#define __global__

#endif

#endif //ENABLE_CUDA

#endif //NEURAL_NETWORK_CUDAENVIRONMENT_H
