/**
 * Created by wjy50 on 18-5-18.
 */

#include <random>
#include "interface.h"
#include "../utils/UniquePointerExt.h"

#if ENABLE_CUDA

cublasHandle_t M_CUBLAS_HANDLE;

FloatType DEF_ALPHA = 1, DEF_BETA = 0;

#define DEF_THREADS_PER_BLOCK 512

void initializeCUDA()
{
    cublasCreate_v2(&M_CUBLAS_HANDLE);
}

void destroyCUDA()
{
    cublasDestroy_v2(M_CUBLAS_HANDLE);
    cudaThreadExit();
}

__global__ void cuLeakyReLU(FloatType *out, const FloatType *in, int len, FloatType l)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        out[index] = in[index] >= 0 ? in[index] : (l * in[index]);
    }
}

void leakyReLU(FloatType *out, const FloatType *in, int len, FloatType l)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    cuLeakyReLU<<<blocks, threadsPerBlock>>>(out, in, len, l);
}

__global__ void cuLeakyReLU_bp(FloatType *out, const FloatType *x, const FloatType *delta, int len, FloatType l)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        out[index] = x[index] >= 0 ? delta[index] : (l * delta[index]);
    }
}

void leakyReLU_bp(FloatType *out, const FloatType *x, const FloatType *delta, int len, FloatType l)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    cuLeakyReLU_bp<<<blocks, threadsPerBlock>>>(out, x, delta, len, l);
}

__global__ void cuReLU(FloatType *out, const FloatType *in, int len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        out[index] = in[index] >= 0 ? in[index] : 0;
    }
}

void reLU(FloatType *out, const FloatType *in, int len)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    cuReLU<<<blocks, threadsPerBlock>>>(out, in, len);
}

__global__ void cuReLU_bp(FloatType *out, const FloatType *x, const FloatType *delta, int len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        out[index] = x[index] >= 0 ? delta[index] : 0;
    }
}

void reLU_bp(FloatType *out, const FloatType *x, const FloatType *delta, int len)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    cuReLU_bp<<<blocks, threadsPerBlock>>>(out, x, delta, len);
}

__global__ void cuSigmoidOutput(FloatType *out, const FloatType *in, int len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        out[index] = 1 / (1 + CU_EXP(-in[index]));
    }
}

void sigmoidOutput(FloatType *out, const FloatType *in, int len)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    cuSigmoidOutput<<<blocks, threadsPerBlock>>>(out, in, len);
}

__global__ void cuAddVTo(FloatType *r, const FloatType *a, const FloatType *b, int len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        r[index] = a[index] + b[index];
    }
}

void addVTo(FloatType *r, const FloatType *a, const FloatType *b, int len)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    cuAddVTo<<<blocks, threadsPerBlock>>>(r, a, b, len);
}

__global__ void cuSubtractVTo(FloatType *r, const FloatType *a, const FloatType *b, int len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        r[index] = a[index] - b[index];
    }
}

void subtractVTo(FloatType *r, const FloatType *a, const FloatType *b, int len)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    cuSubtractVTo<<<blocks, threadsPerBlock>>>(r, a, b, len);
}

__global__ void cuExpArr(FloatType *out, const FloatType *in, int len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        out[index] = CU_EXP(in[index]);
    }
}

__global__ void cuSoftMax(FloatType *out, int len, int count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        int offset = index * len;
        FloatType sum = 0;
        for (int i = 0; i < len; ++i) {
            sum += out[offset + i];
        }
        for (int i = 0; i < len; ++i) {
            out[offset + i] /= sum;
        }
    }
}

void softMaxOutput(FloatType *out, const FloatType *in, int len, int count)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = ((len * count) + threadsPerBlock - 1) / threadsPerBlock;
    cuExpArr<<<blocks, threadsPerBlock>>>(out, in, len * count);
    int smThreadsPerBlock = 256;
    int smBlocks = (count + smThreadsPerBlock - 1) / smThreadsPerBlock;
    cuSoftMax<<<smBlocks, smThreadsPerBlock>>>(out, len, count);
}

__global__ void cuLinearLayerBias(FloatType *vs, int dim, const FloatType *biases, int len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        int bIndex = index % dim;
        vs[index] += biases[bIndex];
    }
}

void linearLayerBias(FloatType *vs, int dim, const FloatType *biases, int batchSize)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = ((dim * batchSize) + threadsPerBlock - 1) / threadsPerBlock;
    cuLinearLayerBias<<<blocks, threadsPerBlock>>>(vs, dim, biases, dim * batchSize);
}

__global__ void cuConvLayerBias(FloatType *m, int imgSize, int channel, const FloatType *biases, int len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        int bIndex = (index / imgSize) % channel;
        m[index] += biases[bIndex];
    }
}

void convLayerBias(FloatType *m, int r, int c, int channel, const FloatType *biases, int batchSize)
{
    int len = r * c * channel * batchSize;
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    cuConvLayerBias<<<blocks, threadsPerBlock>>>(m, r * c, channel, biases, len);
}

void initConvKernel(FloatType *k, int size, int inputDim)
{
    std::random_device rd;
    std::normal_distribution<FloatType> distribution(0, std::sqrt(static_cast<FloatType>(2) / inputDim));
    std::unique_ptr<FloatType[]> tmp = make_unique_array<FloatType[]>(static_cast<size_t>(size));
    for (int i = 0; i < size; ++i) {
        tmp[i] = distribution(rd);
    }
    cudaMemcpy(k, tmp.get(), size * sizeof(FloatType), cudaMemcpyHostToDevice);
}

void initLinearWeights(FloatType *w, int outputDim, int inputDim)
{
    std::random_device rd;
    std::normal_distribution<FloatType> distribution(0, std::sqrt(static_cast<FloatType>(2) / inputDim));
    int weightCount = outputDim * inputDim;
    std::unique_ptr<FloatType[]> tmp = make_unique_array<FloatType[]>(static_cast<size_t>(weightCount));
    for (int i = 0; i < weightCount; ++i) {
        tmp[i] = distribution(rd);
    }
    cudaMemcpy(w, tmp.get(), weightCount * sizeof(FloatType), cudaMemcpyHostToDevice);
}

__global__ void cuSGD(FloatType *params, const FloatType *gradients, FloatType eta, int len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        params[index] += eta * gradients[index];
    }
}

void sgd(FloatType *params, const FloatType *gradients, FloatType eta, int len)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    cuSGD<<<blocks, threadsPerBlock>>>(params, gradients, eta, len);
}

__global__ void cuL2SGD(FloatType *params, const FloatType *gradients, FloatType eta, FloatType reg, int len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        params[index] = params[index] * reg + eta * gradients[index];
    }
}

void l2SGD(FloatType *params, const FloatType *gradients, FloatType eta, FloatType reg, int len)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
}

#endif //ENABLE_CUDA