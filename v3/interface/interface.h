/**
 * Created by wjy50 on 18-5-18.
 */

#ifndef NEURAL_NETWORK_INTERFACE_H
#define NEURAL_NETWORK_INTERFACE_H

#include "../def/type.h"
#include "../def/CUDAEnvironment.h"

#if ENABLE_CUDA

#include <cuda_runtime.h>

#define multiplyMVTo(r, m, v, x, y) M_CUBLAS_GEMV(M_CUBLAS_HANDLE, CUBLAS_OP_T, (y), (x), &DEF_ALPHA, m, (y), (v), 1, &DEF_BETA, (r), 1)

template<typename T>
T *allocArray(int count)
{
    T *p;
    cudaMalloc(&p, count * sizeof(T));
    return p;
}

template<typename T>
void clearArray(T *arr, int len)
{
    cudaMemset(arr, 0, len * sizeof(T));
}

#else

#include <cstring>

template<typename T>
T *allocArray(int count)
{
    return new T[count];
}

template<typename T>
void clearArray(T *arr, int len)
{
    memset(arr, 0, len * sizeof(T));
}

void multiplyMVTo(FloatType *r, const FloatType *m, const FloatType *v, int x, int y);

void multiplyMTmTo(FloatType *r, const FloatType *lhs, const FloatType *rhs, int x, int y, int z);

void multiplyMMTo(FloatType *r, const FloatType *lhs, const FloatType *rhs, int x, int y, int z);

void multiplyTmMTo(FloatType *r, const FloatType *lhs, const FloatType *rhs, int x, int y, int z);

void multiplyNVTo(FloatType *r, FloatType n, const FloatType *v, int dim);

void averageVTo(FloatType *r, const FloatType *v, int dim, int count);

#endif

void freeArray(void *p);

void leakyReLU(FloatType *out, const FloatType *in, int len, FloatType l);

void leakyReLU_bp(FloatType *out, const FloatType *x, const FloatType *delta, int len, FloatType l);

void reLU(FloatType *out, const FloatType *in, int len);

void reLU_bp(FloatType *out, const FloatType *x, const FloatType *delta, int len);

void sigmoidOutput(FloatType *out, const FloatType *in, int len);

void subtractVTo(FloatType *r, const FloatType *a, const FloatType *b, int len);

void softMaxOutput(FloatType *out, const FloatType *in, int len, int count);

void sgd(FloatType *params, const FloatType *gradients, FloatType eta, int len);

void l2SGD(FloatType *params, const FloatType *gradients, FloatType eta, FloatType reg, int len);

void adamFirstMomentEstimate(FloatType *m, FloatType beta, FloatType oneMBeta, const FloatType *g, int len);

void adamSecondMomentEstimate(FloatType *v, FloatType beta, FloatType oneMBeta, const FloatType *g, int len);

void adamUpdate(FloatType *params, const FloatType *m, const FloatType *v, int len, FloatType alpha,
                FloatType oneMBeta1T, FloatType oneMBeta2T
);

/*EWIN: exponentially weighted infinity norm*/
void adaMaxEWIN(FloatType *u, FloatType beta, const FloatType *g, int len);

void adaMaxUpdate(FloatType *params, const FloatType *m, const FloatType *u, int len,
                  FloatType learningRate, FloatType betaTMOne
);

void getMNISTBatch(FloatType *data, FloatType *labels,
                   const unsigned char *dataBuffer, const unsigned char *labelBuffer,
                   const int *indices, int count
);

void getCIFAR10Batch(FloatType *data, FloatType *labels,
                     const unsigned char *buffer,
                     const int *indices, int count
);

void meanPooling(const FloatType *in, int inputWidth, int inputHeight,
                 FloatType *out, int outputWidth, int outputHeight,
                 int windowWidth, int windowHeight,
                 int xStride, int yStride,
                 int channelCount
);

void meanPoolingBP(const FloatType *in, int inputWidth, int inputHeight,
                   FloatType *out, int outputWidth, int outputHeight,
                   int windowWidth, int windowHeight,
                   int xStride, int yStride,
                   int channelCount
);

void maxPooling(const FloatType *in, int inputWidth, int inputHeight,
                FloatType *out, int outputWidth, int outputHeight,
                int *xOffset, int *yOffset,
                int windowWidth, int windowHeight,
                int xStride, int yStride,
                int channelCount
);

void maxPoolingBP(const FloatType *in, int inputWidth, int inputHeight,
                  FloatType *out, int outputWidth, int outputHeight,
                  const int *xOffset, const int *yOffset,
                  int windowWidth, int windowHeight,
                  int xStride, int yStride,
                  int channelCount
);

void conv(const FloatType *in, int inputWidth, int inputHeight, int channelCount,
          FloatType *out, int outputWidth, int outputHeight,
          const FloatType *kernel, int kernelWidth, int kernelHeight, int kernelCount,
          int xStride, int yStride, int xPadding, int yPadding,
          int batchSize
);

void convBP(const FloatType *in, int inputWidth, int inputHeight,
            FloatType *out, int outputWidth, int outputHeight, int channelCount,
            const FloatType *kernel, int kernelWidth, int kernelHeight, int kernelCount,
            int xStride, int yStride, int xPadding, int yPadding,
            int batchSize
);

void convGradients(FloatType *kernel, int kernelWidth, int kernelHeight,
                   FloatType *biases, int kernelCount,
                   const FloatType *delta, int outputWidth, int outputHeight,
                   const FloatType *input, int inputWidth, int inputHeight, int channelCount,
                   int xStride, int yStride, int xPadding, int yPadding,
                   int batchSize
);

void linearLayerBias(FloatType *vs, int dim, const FloatType *biases, int batchSize);

void convLayerBias(FloatType *m, int r, int c, int channel, const FloatType *biases, int batchSize);

void initConvKernel(FloatType *k, int size, int inputDim);

void initLinearWeights(FloatType *w, int outputDim, int inputDim);

void linearDropout(FloatType *v, int dim, const int *ids, int dropoutCount, int batchSize);

#endif //NEURAL_NETWORK_INTERFACE_H
