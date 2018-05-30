/**
 * Created by wjy50 on 18-5-18
 */

#ifndef NEURAL_NETWORK_CUDATOOLS_CUH
#define NEURAL_NETWORK_CUDATOOLS_CUH

#include "CUDAHelper.h"

#if ENABLE_CUDA

#include <cuda.h>

__global__ void cuConv(
        const FloatType *in, int iW, int iH, int iChannel,
        const FloatType *k, int kW, int kH, int kCount,
        FloatType *out, int oW, int oH,
        int xS, int yS,
        int xP, int yP
);

__global__ void cuMaxPooling_O(
        const FloatType *in, int iW, int iH,
        FloatType *out, int oW, int oH,
        int *xOffset, int *yOffset,
        int pW, int pH,
        int xS, int yS,
        int xP, int yP,
        int channel
);

__global__ void cuMaxPooling(
        const FloatType *in, int iW, int iH,
        FloatType *out, int oW, int oH,
        int pW, int pH,
        int xS, int yS,
        int xP, int yP,
        int channel
);

__global__ void cuMeanPooling(
        const FloatType *in, int iW, int iH,
        FloatType *out, int oW, int oH,
        int pW, int pH,
        int xS, int yS,
        int xP, int yP,
        int channel
);

__global__ void cuMaxBP(
        FloatType *in, int iW, int iH,
        const FloatType *out, int oW, int oH,
        const int *xOffset, const int *yOffset,
        int pW, int pH,
        int xS, int yS,
        int xP, int yP,
        int channel
);

__global__ void cuMeanBP(
        FloatType *in, int iW, int iH,
        const FloatType *out, int oW, int oH,
        int pW, int pH,
        int xS, int yS,
        int xP, int yP,
        int channel
);

__global__ void cuSigmoid(FloatType *a, const FloatType *z, int len);

__global__ void cu_dSigmoid_dx(FloatType *r, const FloatType *x, int len);

__global__ void cuRE_LU(FloatType *a, const FloatType *z, int len);

__global__ void cu_dRE_LU_dx(FloatType *r, const FloatType *x, int len);

__global__ void cuL_RE_LU(FloatType *a, const FloatType *z, int len);

__global__ void cu_dL_RE_LU_dx(FloatType *r, const FloatType *x, int len);

__global__ void cuSoftMax(FloatType *a, const FloatType *z, int len);

#endif //ENABLE_CUDA

#endif //NEURAL_NETWORK_CUDATOOLS_CUH