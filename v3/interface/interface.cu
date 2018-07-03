/**
 * Created by wjy50 on 18-5-18.
 */

#include <random>
#include <curand_kernel.h>
#include "interface.h"
#include "../utils/UniquePointerExt.h"
#include "../utils/permutation.h"
#include "../nn/AutoEncoder.h"
#include "../nn/layer/activation/LReLULayer.h"
#include "../nn/layer/activation/SigmoidOutputLayer.h"
#include "../nn/optimizer/AdamOptimizer.h"
#include "../nn/layer/LinearLayer.h"
#include "../data/mnist/MNISTDataSet.h"
#include "../nn/layer/ConvLayer.h"

#if ENABLE_CUDA

cublasHandle_t M_CUBLAS_HANDLE;

FloatType DEF_ALPHA = 1, DEF_BETA = 0;

FloatType *sharedDeviceFloatArray = nullptr;
int sharedDeviceFloatArrayLen = 0;

void initializeCUDA()
{
    cublasCreate_v2(&M_CUBLAS_HANDLE);
    sharedDeviceFloatArray = nullptr;
    sharedDeviceFloatArrayLen = 0;
}

void destroyCUDA()
{
    freeArray(sharedDeviceFloatArray);
    sharedDeviceFloatArray = nullptr;
    sharedDeviceFloatArrayLen = 0;
    cublasDestroy_v2(M_CUBLAS_HANDLE);
    cudaThreadExit();
}

void ensureSharedDeviceFloatArraySize(int size)
{
    if (sharedDeviceFloatArrayLen < size) {
        if (!sharedDeviceFloatArray) sharedDeviceFloatArrayLen = 1;
        while (sharedDeviceFloatArrayLen < size) sharedDeviceFloatArrayLen <<= 1;
        freeArray(sharedDeviceFloatArray);
        sharedDeviceFloatArray = allocArray<FloatType>(sharedDeviceFloatArrayLen);
    }
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

void alphaXPlusY(FloatType alpha, const FloatType *x, FloatType *y, int len)
{
    M_CUBLAS_AXPY(M_CUBLAS_HANDLE, len, &alpha, x, 1, y, 1);
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

__global__ void cuSoftMax(FloatType *out, const FloatType *in, int blockLen, int totalLen, int groupLen, int maxP2)
{
    int offset = blockIdx.x * blockLen;
    int index = offset + threadIdx.x;
    blockLen = CU_MIN(totalLen - offset, blockLen);
    extern __shared__ FloatType sharedM[];
    if (threadIdx.x < blockLen) {
        out[index] = CU_EXP(in[index]);
        sharedM[threadIdx.x] = out[index];
    }
    int groupId = threadIdx.x / groupLen;
    int groupOffset = groupId * groupLen;
    int indexInGroup = threadIdx.x % groupLen;
    __syncthreads();
    for (; maxP2 > 0; maxP2 >>= 1) {
        if (threadIdx.x < blockLen && indexInGroup < maxP2 && indexInGroup + maxP2 < groupLen) {
            sharedM[threadIdx.x] += sharedM[threadIdx.x + maxP2];
        }
        __syncthreads();
    }
    if (threadIdx.x < blockLen) {
        out[index] /= sharedM[groupOffset];
    }
}

__global__ void cuSoftMaxExp(FloatType *out, const FloatType *in, int totalLen)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < totalLen) {
        out[index] = CU_EXP(in[index]);
    }
}

__global__ void cuGroupSum(FloatType *sum, const FloatType *in, int groupLen, int blocksPerGroup, FloatType div = 1)
{
    int groupId = blockIdx.x / blocksPerGroup;
    int blockIdInGroup = blockIdx.x % blocksPerGroup;
    int groupOffset = groupId * groupLen;
    int offsetInGroup = blockIdInGroup * blockDim.x;
    int offset = groupOffset + offsetInGroup;
    int additionLen = CU_MIN(static_cast<int>(blockDim.x), groupLen - offsetInGroup);
    int maxP2 = 1;
    while (maxP2 < additionLen) maxP2 <<= 1;
    maxP2 >>= 1;
    extern __shared__ FloatType sharedM[];
    if (offsetInGroup + threadIdx.x < groupLen) {
        sharedM[threadIdx.x] = in[offset + threadIdx.x];
    }
    __syncthreads();
    for (; maxP2 > 0; maxP2 >>= 1) {
        if (offsetInGroup + threadIdx.x < groupLen && threadIdx.x < maxP2 && threadIdx.x + maxP2 < additionLen) {
            sharedM[threadIdx.x] += sharedM[threadIdx.x + maxP2];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        sum[blockIdx.x] = sharedM[threadIdx.x] / div;
    }
}

__global__ void cuSoftMaxNormalize(FloatType *out, const FloatType *sum, int groupLen, int totalLen)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < totalLen) {
        out[index] /= sum[index / groupLen];
    }
}

void softMaxOutput(FloatType *out, const FloatType *in, int len, int count)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int totalLen = len * count;
    if (len <= threadsPerBlock) {
        int blockLen = threadsPerBlock / len * len;
        int blocks = (totalLen + blockLen - 1) / blockLen;
        int maxP2 = 1;
        while (maxP2 < len) maxP2 <<= 1;
        maxP2 >>= 1;
        cuSoftMax<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(FloatType)>>>(out, in, blockLen, totalLen, len, maxP2);
    } else {
        int expBlocks = (totalLen + threadsPerBlock - 1) / threadsPerBlock;
        cuSoftMaxExp<<<expBlocks, threadsPerBlock>>>(out, in, totalLen);

        int blocksPerGroup = (len + threadsPerBlock - 1) / threadsPerBlock;
        int blocks = blocksPerGroup * count;
        ensureSharedDeviceFloatArraySize(blocks);
        cuGroupSum<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(FloatType)>>>(sharedDeviceFloatArray, out, len, blocksPerGroup);
        int nLen = blocksPerGroup;
        blocksPerGroup = (nLen + threadsPerBlock - 1) / threadsPerBlock;
        for (; nLen > 1; ) {
            cuGroupSum<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(FloatType)>>>(sharedDeviceFloatArray, sharedDeviceFloatArray, nLen, blocksPerGroup);
            nLen = blocksPerGroup;
            blocksPerGroup = (nLen + threadsPerBlock - 1) / threadsPerBlock;
        }

        cuSoftMaxNormalize<<<blocks, threadsPerBlock>>>(out, sharedDeviceFloatArray, len, totalLen);
    }
}

__global__ void cuLinearLayerBias(FloatType *vs, int dim, const FloatType *biases, int len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        vs[index] += biases[index % dim];
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
        m[index] += biases[(index / imgSize) % channel];
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
    std::normal_distribution<FloatType> distribution(0, std::sqrt(static_cast<FloatType>(1) / inputDim));
    std::unique_ptr<FloatType[]> tmp = make_unique_array<FloatType[]>(static_cast<size_t>(size));
    for (int i = 0; i < size; ++i) {
        tmp[i] = distribution(rd);
    }
    cudaMemcpy(k, tmp.get(), size * sizeof(FloatType), cudaMemcpyHostToDevice);
}

void initLinearWeights(FloatType *w, int outputDim, int inputDim)
{
    std::random_device rd;
    std::normal_distribution<FloatType> distribution(0, std::sqrt(static_cast<FloatType>(1) / inputDim));
    int weightCount = outputDim * inputDim;
    std::unique_ptr<FloatType[]> tmp = make_unique_array<FloatType[]>(static_cast<size_t>(weightCount));
    for (int i = 0; i < weightCount; ++i) {
        tmp[i] = distribution(rd);
    }
    cudaMemcpy(w, tmp.get(), weightCount * sizeof(FloatType), cudaMemcpyHostToDevice);
}

void sgd(FloatType *params, const FloatType *gradients, FloatType eta, int len)
{
    M_CUBLAS_AXPY(M_CUBLAS_HANDLE, len, &eta, gradients, 1, params, 1);
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
    cuL2SGD<<<blocks, threadsPerBlock>>>(params, gradients, eta, reg, len);
}

__global__ void cuAdamEstimate(FloatType *m, FloatType *v, FloatType beta1, FloatType oneMBeta1, FloatType beta2, FloatType oneMBeta2, const FloatType *g, int len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        m[index] = m[index] * beta1 + oneMBeta1 * g[index];
        v[index] = v[index] * beta2 + oneMBeta2 * g[index] * g[index];
    }
}

void adamEstimate(FloatType *m, FloatType *v, FloatType beta1, FloatType oneMBeta1, FloatType beta2, FloatType oneMBeta2, const FloatType *g, int len)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    cuAdamEstimate<<<blocks, threadsPerBlock>>>(m, v, beta1, oneMBeta1, beta2, oneMBeta2, g, len);
}

__global__ void cuAdamUpdate(FloatType *params, const FloatType *m, const FloatType *v, int len,
                             FloatType alpha, FloatType oneMBeta1T, FloatType oneMBeta2T,
                             FloatType weightDecay, int decayRange
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        if (index < decayRange)
            params[index] = params[index] * weightDecay + alpha * m[index] / (oneMBeta1T * CU_SQRT(v[index] / oneMBeta2T) + static_cast<FloatType>(1e-6));
        else
            params[index] += alpha * m[index] / (oneMBeta1T * CU_SQRT(v[index] / oneMBeta2T) + static_cast<FloatType>(1e-6));
    }
}

void adamUpdate(FloatType *params, const FloatType *m, const FloatType *v, int len, FloatType alpha,
                FloatType oneMBeta1T, FloatType oneMBeta2T, FloatType weightDecay, int decayRange
)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    cuAdamUpdate<<<blocks, threadsPerBlock>>>(params, m, v, len, alpha, oneMBeta1T, oneMBeta2T, weightDecay, decayRange);
}

__global__ void cuAdaMaxEstimate(FloatType *m, FloatType *u, FloatType beta1, FloatType oneMBeta1, FloatType beta2, const FloatType *g, int len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        m[index] = m[index] * beta1 + oneMBeta1 * g[index];
        u[index] = CU_FMAX(beta2 * u[index], CU_FABS(g[index]));
    }
}

void adaMaxEstimate(FloatType *m, FloatType *u, FloatType beta1, FloatType oneMBeta1, FloatType beta2, const FloatType *g, int len)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    cuAdaMaxEstimate<<<blocks, threadsPerBlock>>>(m, u, beta1, oneMBeta1, beta2, g, len);
}

__global__ void cuAdaMaxUpdate(FloatType *params, const FloatType *m, const FloatType *u, int len,
                               FloatType learningRate, FloatType betaTMOne, FloatType weightDecay, int decayRange
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        if (index < decayRange)
            params[index] = params[index] * weightDecay + learningRate * m[index] / (betaTMOne * u[index] + static_cast<FloatType>(1e-6));
        else
            params[index] += learningRate * m[index] / (betaTMOne * u[index] + static_cast<FloatType>(1e-6));
    }
}

void adaMaxUpdate(FloatType *params, const FloatType *m, const FloatType *u, int len,
                  FloatType learningRate, FloatType betaTMOne, FloatType weightDecay, int decayRange
)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    cuAdaMaxUpdate<<<blocks, threadsPerBlock>>>(params, m, u, len, learningRate, betaTMOne, weightDecay, decayRange);
}

__global__ void cuLinearDropout(FloatType *v, const int *ids, int dropoutCount, int dim, int len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len && ids[index % dim] < dropoutCount) {
        v[index] = 0;
    }
}

void linearDropout(FloatType *v, int dim, const int *ids, int dropoutCount, int batchSize)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = ((dim * batchSize) + threadsPerBlock - 1) / threadsPerBlock;
    cuLinearDropout<<<blocks, threadsPerBlock>>>(v, ids, dropoutCount, dim, dim * batchSize);
}

__global__ void cuAverageVTo(FloatType *r, const FloatType *v, int dim, int count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dim) {
        FloatType sum = 0;
        const FloatType *cv = v + index;
        for (int i = 0; i < count; ++i) {
            sum += *cv;
            cv += dim;
        }
        r[index] = sum / count;
    }
}

void averageVTo(FloatType *r, const FloatType *v, int dim, int count)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = (dim + threadsPerBlock - 1) / threadsPerBlock;
    cuAverageVTo<<<blocks, threadsPerBlock>>>(r, v, dim, count);
}

__global__ void cuGetMNISTData(
        FloatType *data,
        const unsigned char *dataBuffer,
        const int *indices, int len
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        int i = indices[index / (28 * 28)];
        data[index] = static_cast<FloatType>(dataBuffer[i * 28 * 28 + index % (28 * 28)]) / 0xff;
    }
}

__global__ void cuGetMNISTLabel(
        FloatType *label,
        const unsigned char *labelBuffer,
        const int *indices, int len
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        int i = indices[index / 10];
        label[index] = labelBuffer[i] == (index % 10) ? 1 : 0;
    }
}

void getMNISTBatch(FloatType *data, FloatType *labels,
                   const unsigned char *dataBuffer, const unsigned char *labelBuffer,
                   const int *indices, int count
)
{
    int threadsPerBlocks = DEF_THREADS_PER_BLOCK;
    int len = 28 * 28 * count;
    int blocks = (len + threadsPerBlocks - 1) / threadsPerBlocks;
    cuGetMNISTData<<<blocks, threadsPerBlocks>>>(data, dataBuffer, indices, len);

    if (labels) {
        len = 10 * count;
        blocks = (len + threadsPerBlocks - 1) / threadsPerBlocks;
        cuGetMNISTLabel<<<blocks, threadsPerBlocks>>>(labels, labelBuffer, indices, len);
    }
}

__global__ void cuConv(
        const FloatType *input, int inputWidth, int inputHeight, int inputSize, int channelCount,
        const FloatType *kernel, int kernelWidth, int kernelHeight, int kernelSize, int kernelCount,
        FloatType *output, int outputWidth, int outputSize,
        int xStride, int yStride, int nXPadding, int nYPadding,
        int totalLen, int inputJump, int kernelJump
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < totalLen) {
        int outputLayer = index / outputSize;
        int kernelId = outputLayer % kernelCount;
        int batchId = outputLayer / kernelCount;
        int indexInOutput = index % outputSize;
        int y = indexInOutput / outputWidth;
        int x = indexInOutput % outputWidth;
        int inputY = y * yStride + nYPadding;
        int inputX = x * xStride + nXPadding;
        int kernelY = CU_MAX(-inputY, 0);
        int kernelX = CU_MAX(-inputX, 0);
        int yLim = CU_MIN(kernelHeight, inputHeight - inputY);
        int xLim = CU_MIN(kernelWidth, inputWidth - inputX);
        FloatType sum = 0;
        const FloatType *inputBegin = input + batchId * inputJump + (inputY + kernelY) * inputWidth + inputX;
        const FloatType *kernelBegin = kernel + kernelId * kernelJump + kernelY * kernelWidth;
        for (int i = 0; i < channelCount; ++i) {
            const FloatType *mInput = inputBegin;
            const FloatType *mKernel = kernelBegin;
            for (int j = kernelY; j < yLim; ++j) {
                for (int k = kernelX; k < xLim; ++k) {
                    sum += mKernel[k] * mInput[k];
                }
                mInput += inputWidth;
                mKernel += kernelWidth;
            }
            inputBegin += inputSize;
            kernelBegin += kernelSize;
        }
        output[index] = sum;
    }
}

void conv(
        const FloatType *in, int inputWidth, int inputHeight, int channelCount,
        FloatType *out, int outputWidth, int outputHeight,
        const FloatType *kernel, int kernelWidth, int kernelHeight, int kernelCount,
        int xStride, int yStride, int xPadding, int yPadding,
        int batchSize
)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int outputSize = outputWidth * outputHeight;
    int len = outputSize * kernelCount * batchSize;
    int blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    int inputSize = inputWidth * inputHeight;
    int kernelSize = kernelWidth * kernelHeight;
    cuConv<<<blocks, threadsPerBlock>>>(
            in, inputWidth, inputHeight, inputSize, channelCount,
            kernel, kernelWidth, kernelHeight, kernelSize, kernelCount,
            out, outputWidth, outputSize,
            xStride, yStride, -xPadding, -yPadding,
            len, inputSize * channelCount, kernelSize * channelCount
    );
}

__global__ void im2col(
        FloatType *m, int outputWidth, int outputSize,
        const FloatType *input, int inputWidth, int inputHeight, int inputSize,
        int kernelWidth, int kernelSize,
        int xStride, int yStride, int nXPadding, int nYPadding,
        int len
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        int r = index / outputSize;
        int c = index % outputSize;
        int channel = r / kernelSize;
        //int outputX = c % outputWidth;
        //int outputY = c / outputWidth;
        int kernelIndex = r % kernelSize;
        int inputX = (c % outputWidth) * xStride + nXPadding + (kernelIndex % kernelWidth);
        int inputY = (c / outputWidth) * yStride + nYPadding + kernelIndex / kernelWidth;
        if (inputX < 0 || inputY < 0 || inputX >= inputWidth || inputY >= inputHeight) m[index] = 0;
        else m[index] = input[channel * inputSize + inputY * inputWidth + inputX];
    }
}

void conv2(
        const FloatType *in, int inputWidth, int inputHeight, int channelCount,
        FloatType *out, int outputWidth, int outputHeight,
        const FloatType *kernel, int kernelWidth, int kernelHeight, int kernelCount,
        int xStride, int yStride, int xPadding, int yPadding,
        int batchSize
)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int outputSize = outputWidth * outputHeight;
    int kernelSize = kernelWidth * kernelHeight;
    int k = kernelSize * channelCount;
    int len = outputSize * k;
    ensureSharedDeviceFloatArraySize(len);
    int inputSize = inputWidth * inputHeight;
    int nXPadding = -xPadding, nYPadding = -yPadding;
    int inputJump = inputSize * channelCount;
    int outputJump = outputSize * kernelCount;
    int blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    for (int i = 0; i < batchSize; ++i) {
        im2col<<<blocks, threadsPerBlock>>>(
                sharedDeviceFloatArray, outputWidth, outputSize,
                in + i * inputJump, inputWidth, inputHeight, inputSize,
                kernelWidth, kernelSize,
                xStride, yStride, nXPadding, nYPadding,
                len
        );
        multiplyMMTo(out + i * outputJump, kernel, sharedDeviceFloatArray, kernelCount, k, outputSize);
    }
}

__global__ void cuConvBP(
        const FloatType *in, int inputWidth, int inputSize,
        const FloatType *kernel, int kernelWidth, int kernelHeight, int kernelSize, int kernelCount, int kernelJump,
        FloatType *out, int outputWidth, int outputHeight, int outputSize, int channelCount,
        int xStride, int yStride, int xPadding, int yPadding,
        int totalLen
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < totalLen) {
        int outputId = index / outputSize;
        int batchId = outputId / channelCount;
        int kernelLayer = outputId % channelCount;
        int indexInOutput = index % outputSize;
        int outputY = indexInOutput / outputWidth;
        int outputX = indexInOutput % outputWidth;
        FloatType sum = 0;
        int iy = (outputY < kernelHeight - yPadding) ? 0 : ((outputY + yPadding - kernelHeight) / yStride + 1);
        int ix = (outputX < kernelWidth - xPadding) ? 0 : ((outputX + xPadding - kernelWidth) / xStride + 1);
        int yBegin = iy * yStride - yPadding;
        int xBegin = ix * xStride - xPadding;
        int yLim = CU_MIN(outputY, outputHeight + yPadding - kernelHeight);
        int xLim = CU_MIN(outputX, outputWidth + xPadding - kernelWidth);
        for (int i = 0; i < kernelCount; ++i) {
            const FloatType *mKernel = kernel + kernelLayer * kernelSize + i * kernelJump;
            const FloatType *mInput = in + batchId * inputSize * kernelCount + inputSize * i;
            int inY = iy;
            for (int j = yBegin; j <= yLim; j += yStride, ++inY) {
                int kernelY = outputY - j;
                int inX = ix;
                for (int k = xBegin; k <= xLim; k += xStride, ++inX) {
                    int kernelX = outputX - k;
                    sum += mKernel[kernelY * kernelWidth + kernelX] * mInput[inY * inputWidth + inX];
                }
            }
        }
        out[index] = sum;
    }
}

void convBP(const FloatType *delta, int outputWidth, int outputHeight,
            FloatType *deltaOut, int inputWidth, int inputHeight, int channelCount,
            const FloatType *kernel, int kernelWidth, int kernelHeight, int kernelCount,
            int xStride, int yStride, int xPadding, int yPadding,
            int batchSize
)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int kernelSize = kernelWidth * kernelHeight;
    int kernelJump = kernelSize * channelCount;
    int outputSize = inputWidth * inputHeight;
    int totalLen = outputSize * channelCount * batchSize;
    int blocks = (totalLen + threadsPerBlock - 1) / threadsPerBlock;
    cuConvBP<<<blocks, threadsPerBlock>>> (delta, outputWidth, outputWidth * outputHeight,
            kernel, kernelWidth, kernelHeight, kernelSize, kernelCount, kernelJump,
            deltaOut, inputWidth, inputHeight, outputSize, channelCount,
            xStride, yStride, xPadding, yPadding,
            totalLen
    );
}

__global__ void im2colBP(
        FloatType *m,
        int inputWidth, int inputSize,
        int kernelWidth,int kernelSize,
        const FloatType *delta, int outputWidth, int outputSize,
        int bpWidth, int bpHeight, int bpXUnit, int bpYUnit, int bpXPadding, int bpYPadding,
        int len
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        int c = index % inputSize;
        int r = index / inputSize;
        int channel = r / kernelSize;
        int kernelIndex = r % kernelSize;
        int bpX = (c % inputWidth) - bpXPadding + (kernelIndex % kernelWidth);
        int bpY = (c / inputWidth) - bpYPadding + kernelIndex / kernelWidth;
        if (bpX < 0 || bpY < 0 || bpX >= bpWidth || bpY >= bpHeight || (bpX % bpXUnit) != 0 || (bpY % bpYUnit) != 0)
            m[index] = 0;
        else
            m[index] = delta[channel * outputSize + (bpY / bpYUnit) * outputWidth + bpX / bpXUnit];
    }
}

__global__ void rearrangeKernel(
        FloatType *o, const FloatType *kernel,
        int kernelSize, int kernelJump, int iKernelJump,
        int len
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        int iIndexInKernel = kernelSize - 1 - (index % kernelSize);
        int iChannel = index / kernelJump;
        int iKernel = (index % kernelJump) / kernelSize;
        o[index] = kernel[iKernel * iKernelJump + iChannel * kernelSize + iIndexInKernel];
    }
}

void convBP2(const FloatType *delta, int outputWidth, int outputHeight,
            FloatType *deltaOut, int inputWidth, int inputHeight, int channelCount,
            const FloatType *kernel, int kernelWidth, int kernelHeight, int kernelCount,
            int xStride, int yStride, int xPadding, int yPadding,
            int batchSize
)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int kernelSize = kernelWidth * kernelHeight;
    int kernelJump = kernelSize * kernelCount;
    int len1 = kernelJump * channelCount;
    int inputSize = inputWidth * inputHeight;
    int len2 = inputSize * kernelJump;
    int outputSize = outputWidth * outputHeight;
    int deltaJump = outputSize * kernelCount;
    int inputJump = inputSize * channelCount;
    ensureSharedDeviceFloatArraySize(len1 + len2);
    FloatType *m = sharedDeviceFloatArray + len1;
    rearrangeKernel<<<(len1 + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
            sharedDeviceFloatArray, kernel,
            kernelSize, kernelJump, kernelSize * channelCount,
            len1);
    int bpWidth = xStride * (outputWidth - 1) + 1;
    int bpHeight = yStride * (outputHeight - 1) + 1;
    int bpXPadding = kernelWidth - 1 - xPadding;
    int bpYPadding = kernelHeight - 1 - yPadding;
    int blocks = (len2 + threadsPerBlock - 1) / threadsPerBlock;
    for (int i = 0; i < batchSize; ++i) {
        im2colBP<<<blocks, threadsPerBlock>>>(
                m,
                inputWidth, inputSize,
                kernelWidth, kernelSize,
                delta + i * deltaJump, outputWidth, outputSize,
                bpWidth, bpHeight, xStride, yStride, bpXPadding, bpYPadding,
                len2
        );
        multiplyMMTo(deltaOut + i * inputJump, sharedDeviceFloatArray, m, channelCount, kernelJump, inputSize);
    }
}

__global__ void cuConvKernelGrad(
        const FloatType *input, int inputWidth, int inputHeight, int inputSize, int inputJump, int channelCount,
        const FloatType *delta, int deltaWidth, int deltaHeight, int deltaSize, int deltaJump,
        FloatType *kernel, int kernelWidth, int kernelSize,
        int xStride, int yStride, int nXPadding, int nYPadding,
        int len, int batchSize
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int kernelLayerId = index / kernelSize;
    int indexInKernel = index % kernelSize;
    int y = indexInKernel / kernelWidth;
    int x = indexInKernel % kernelWidth;
    int inputY = y * yStride + nYPadding;
    int inputX = x * xStride + nXPadding;
    int deltaY = CU_MAX(-inputY, 0);
    int deltaX = CU_MAX(-inputX, 0);
    int yLim = CU_MIN(deltaHeight, inputHeight - inputY);
    int xLim = CU_MIN(deltaWidth, inputWidth - inputX);
    if (index < len) {
        FloatType sum = 0;
        const FloatType *deltaBegin = delta + (kernelLayerId / channelCount) * deltaSize + deltaY * deltaWidth;
        const FloatType *inputBegin = input + (kernelLayerId % channelCount) * inputSize + (inputY + deltaY) * inputWidth + inputX;
        for (int i = 0; i < batchSize; ++i) {
            const FloatType *mDelta = deltaBegin;
            const FloatType *mInput = inputBegin;
            for (int j = deltaY; j < yLim; ++j) {
                for (int k = deltaX; k < xLim; ++k) {
                    sum += mDelta[k] * mInput[k];
                }
                mDelta += deltaWidth;
                mInput += inputWidth;
            }
            deltaBegin += deltaJump;
            inputBegin += inputJump;
        }
        kernel[index] = sum / batchSize;
    }
}

__global__ void cuConvBiasGrad(
        FloatType *biases,
        const FloatType *delta, int deltaSize, int deltaJump,
        int len, int batchSize
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        FloatType sum = 0;
        const FloatType *mDelta = delta + index * deltaSize;
        for (int i = 0; i < batchSize; ++i) {
            for (int j = 0; j < deltaSize; ++j) {
                sum += mDelta[j];
            }
            mDelta += deltaJump;
        }
        biases[index] = sum / batchSize;
    }
}

void convGradients(
        FloatType *kernel, int kernelWidth, int kernelHeight,
        FloatType *biases, int kernelCount,
        const FloatType *delta, int outputWidth, int outputHeight,
        const FloatType *input, int inputWidth, int inputHeight, int channelCount,
        int xStride, int yStride, int xPadding, int yPadding,
        int batchSize
)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int kernelSize = kernelWidth * kernelHeight;
    int len = kernelSize * channelCount * kernelCount;
    int blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    int inputSize = inputWidth * inputHeight;
    int deltaSize = outputWidth * outputHeight;
    int deltaJump = deltaSize * kernelCount;
    cuConvKernelGrad<<<blocks, threadsPerBlock>>>(
            input, inputWidth, inputHeight, inputSize, inputSize * channelCount, channelCount,
            delta, outputWidth, outputHeight, deltaSize, deltaJump,
            kernel, kernelWidth, kernelSize,
            xStride, yStride, -xPadding, -yPadding,
            len, batchSize
    );

    if (biases) {
        blocks = (kernelCount + threadsPerBlock - 1) / threadsPerBlock;
        cuConvBiasGrad<<<blocks, threadsPerBlock>>>(biases, delta, deltaSize, deltaJump, kernelCount, batchSize);
    }
}

__global__ void im2colGrad(
        FloatType *m, int kernelWidth, int kernelSize, int kernelJump,
        const FloatType *input, int inputWidth, int inputHeight, int inputSize,
        int deltaWidth,
        int xStride, int yStride, int nXPadding, int nYPadding,
        int len
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        int c = index % kernelJump;
        int r = index / kernelJump;
        int channel = c / kernelSize;
        int kernelIndex = c % kernelSize;
        int ix = (kernelIndex % kernelWidth) * xStride + nXPadding + (r % deltaWidth);
        int iy = (kernelIndex / kernelWidth) * yStride + nYPadding + r / deltaWidth;
        if (ix < 0 || iy < 0 || ix >= inputWidth || iy >= inputHeight) m[index] = 0;
        else m[index] = input[channel * inputSize + iy * inputWidth + ix];
    }
}

void convGradients2(
        FloatType *kernel, int kernelWidth, int kernelHeight,
        FloatType *biases, int kernelCount,
        const FloatType *delta, int outputWidth, int outputHeight,
        const FloatType *input, int inputWidth, int inputHeight, int channelCount,
        int xStride, int yStride, int xPadding, int yPadding,
        int batchSize
)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int deltaSize = outputWidth * outputHeight;
    int kernelSize = kernelWidth * kernelHeight;
    int kernelJump = kernelSize * channelCount;
    int inputSize = inputWidth * inputHeight;
    int inputJump = inputSize * channelCount;
    int deltaJump = deltaSize * kernelCount;
    int nXPadding = -xPadding, nYPadding = -yPadding;
    int len = kernelJump * deltaSize;
    ensureSharedDeviceFloatArraySize(len);
    int blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    im2colGrad<<<blocks, threadsPerBlock>>>(
            sharedDeviceFloatArray, kernelWidth, kernelSize, kernelJump,
            input, inputWidth, inputHeight, inputSize,
            outputWidth,
            xStride, yStride, nXPadding, nYPadding,
            len
    );
    multiplyMMTo(kernel, delta, sharedDeviceFloatArray, kernelCount, deltaSize, kernelJump);
    for (int i = 1; i < batchSize - 1; ++i) {
        im2colGrad<<<blocks, threadsPerBlock>>>(
                sharedDeviceFloatArray, kernelWidth, kernelSize, kernelJump,
                input + i * inputJump, inputWidth, inputHeight, inputSize,
                outputWidth,
                xStride, yStride, nXPadding, nYPadding,
                len
        );
        addMultiplyMMTo(kernel, delta + i * deltaJump, sharedDeviceFloatArray, kernelCount, deltaSize, kernelJump);
    }
    int i = batchSize - 1;
    im2colGrad<<<blocks, threadsPerBlock>>>(
            sharedDeviceFloatArray, kernelWidth, kernelSize, kernelJump,
                    input + i * inputJump, inputWidth, inputHeight, inputSize,
                    outputWidth,
                    xStride, yStride, nXPadding, nYPadding,
                    len
    );
    DEF_ALPHA = static_cast<FloatType>(1) / batchSize;
    addMultiplyMMTo(kernel, delta + i * deltaJump, sharedDeviceFloatArray, kernelCount, deltaSize, kernelJump);
    DEF_ALPHA = 1;

    if (biases) {
        blocks = (kernelCount + threadsPerBlock - 1) / threadsPerBlock;
        cuConvBiasGrad<<<blocks, threadsPerBlock>>>(biases, delta, deltaSize, deltaJump, kernelCount, batchSize);
    }
}

__global__ void cuMeanPooling(
        const FloatType *in, int inputWidth, int inputSize,
        FloatType *out, int outputWidth, int outputSize,
        int windowWidth, int windowHeight, int windowSize, int xStride, int yStride,
        int len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        int channelId = index / outputSize;
        int indexInChannel = index % outputSize;
        int y = indexInChannel / outputWidth;
        int x = indexInChannel % outputWidth;
        int yBegin = y * yStride;
        int xBegin = x * xStride;
        const FloatType *mInput = in + channelId * inputSize + yBegin * inputWidth;
        FloatType sum = 0;
        for (int i = yBegin, yLim = i + windowHeight; i < yLim; ++i) {
            for (int j = xBegin, xLim = j + windowWidth; j < xLim; ++j) {
                sum += mInput[j];
            }
            mInput += inputWidth;
        }
        out[index] = sum / windowSize;
    }
}

void meanPooling(const FloatType *in, int inputWidth, int inputHeight,
                 FloatType *out, int outputWidth, int outputHeight,
                 int windowWidth, int windowHeight,
                 int xStride, int yStride,
                 int channelCount
)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int len = outputWidth * outputHeight * channelCount;
    int blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    cuMeanPooling<<<blocks, threadsPerBlock>>>(
            in, inputWidth, inputWidth * inputHeight,
                    out, outputWidth, outputWidth * outputHeight,
                    windowWidth, windowHeight, windowWidth * windowHeight, xStride, yStride, len
    );
}

__global__ void cuMeanPoolingBP(
        const FloatType *in, int inputWidth, int inputSize,
        FloatType *out, int outputWidth, int outputHeight, int outputSize,
        int windowWidth, int windowHeight, int windowSize, int xStride, int yStride,
        int len
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        int channelId = index / outputSize;
        int indexInChannel = index % outputSize;
        int y = indexInChannel / outputWidth;
        int x = indexInChannel % outputWidth;
        int iy = y < windowHeight ? 0 : ((y - windowHeight) / yStride + 1);
        int ix = x < windowWidth ? 0 : ((x - windowWidth) / xStride + 1);
        int yBegin = iy * yStride;
        int xBegin = ix * xStride;
        int yLim = CU_MIN(y, outputHeight - windowHeight);
        int xLim = CU_MIN(x, outputWidth - windowWidth);
        int inY = iy;
        const FloatType *mInput = in + channelId * inputSize + iy * inputWidth;
        FloatType sum = 0;
        for (int i = yBegin; i <= yLim; i += yStride, ++inY) {
            int inX = ix;
            for (int j = xBegin; j <= xLim; j += xStride, ++inX) {
                sum += mInput[inX];
            }
            mInput += inputWidth;
        }
        out[index] = sum / windowSize;
    }
}

void meanPoolingBP(const FloatType *in, int inputWidth, int inputHeight,
                   FloatType *out, int outputWidth, int outputHeight,
                   int windowWidth, int windowHeight,
                   int xStride, int yStride,
                   int channelCount
)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int len = outputWidth * outputHeight * channelCount;
    int blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    cuMeanPoolingBP<<<blocks, threadsPerBlock>>>(
            in, inputWidth, inputWidth * inputHeight,
            out, outputWidth, outputHeight, outputWidth * outputHeight,
            windowWidth, windowHeight, windowWidth * windowHeight,
            xStride, yStride, len
    );
}

__global__ void cuMaxPooling(
        const FloatType *in, int inputWidth, int inputSize,
        FloatType *out, int outputWidth, int outputSize,
        int *xOffset, int *yOffset,
        int windowWidth, int windowHeight, int xStride, int yStride,
        int len
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        int channelId = index / outputSize;
        int indexInChannel = index % outputSize;
        int y = indexInChannel / outputWidth;
        int x = indexInChannel % outputWidth;
        int yBegin = y * yStride;
        int xBegin = x * xStride;
        FloatType max = 0;
        int xo = -1;
        int yo = -1;
        const FloatType *mInput = in + channelId * inputSize + yBegin * inputWidth;
        for (int i = yBegin, yLim = i + windowHeight; i < yLim; ++i) {
            for (int j = xBegin, xLim = j + windowWidth; j < xLim; ++j) {
                FloatType c = mInput[j];
                if (xo < 0 || c > max) {
                    max = c;
                    yo = windowHeight + i - yLim;
                    xo = windowWidth + j - xLim;
                }
            }
            mInput += inputWidth;
        }
        out[index] = max;
        xOffset[index] = xo;
        yOffset[index] = yo;
    }
}

__global__ void cuMaxPoolingWithoutOffset(
        const FloatType *in, int inputWidth, int inputSize,
        FloatType *out, int outputWidth, int outputSize,
        int windowWidth, int windowHeight, int xStride, int yStride,
        int len
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        int channelId = index / outputSize;
        int indexInChannel = index % outputSize;
        int y = indexInChannel / outputWidth;
        int x = indexInChannel % outputWidth;
        int yBegin = y * yStride;
        int xBegin = x * xStride;
        FloatType max = 0;
        int b = 0;
        const FloatType *mInput = in + channelId * inputSize + yBegin * inputWidth;
        for (int i = yBegin, yLim = i + windowHeight; i < yLim; ++i) {
            for (int j = xBegin, xLim = j + windowWidth; j < xLim; ++j) {
                FloatType c = mInput[j];
                if (b == 0 || c > max) {
                    max = c;
                    b = 1;
                }
            }
            mInput += inputWidth;
        }
        out[index] = max;
    }
}

void maxPooling(const FloatType *in, int inputWidth, int inputHeight,
                FloatType *out, int outputWidth, int outputHeight,
                int *xOffset, int *yOffset,
                int windowWidth, int windowHeight,
                int xStride, int yStride,
                int channelCount
)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int len = outputWidth * outputHeight * channelCount;
    int blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    if (xOffset) {
        cuMaxPooling<<<blocks, threadsPerBlock>>>(
                in, inputWidth, inputWidth * inputHeight,
                        out, outputWidth, outputWidth * outputHeight,
                        xOffset, yOffset, windowWidth, windowHeight,
                        xStride, yStride, len
        );
    } else {
        cuMaxPoolingWithoutOffset<<<blocks, threadsPerBlock>>>(
                in, inputWidth, inputWidth * inputHeight,
                        out, outputWidth, outputWidth * outputHeight,
                        windowWidth, windowHeight,
                        xStride, yStride, len
        );
    }
}

__global__ void cuMaxPoolingBP(
        const FloatType *in, int inputWidth, int inputSize,
        FloatType *out, int outputWidth, int outputHeight, int outputSize,
        const int *xOffset, const int *yOffset,
        int windowWidth, int windowHeight, int xStride, int yStride,
        int len
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        int channelId = index / outputSize;
        int indexInChannel = index % outputSize;
        int y = indexInChannel / outputWidth;
        int x = indexInChannel % outputWidth;
        int iy = y < windowHeight ? 0 : ((y - windowHeight) / yStride + 1);
        int ix = x < windowWidth ? 0 : ((x - windowWidth) / xStride + 1);
        int yBegin = iy * yStride;
        int xBegin = ix * xStride;
        int yLim = CU_MIN(y, outputHeight - windowHeight);
        int xLim = CU_MIN(x, outputWidth - windowWidth);
        int inY = iy;
        int jump = channelId * inputSize + iy * inputWidth;
        const FloatType *mInput = in + jump;
        const int *mXOffset = xOffset + jump;
        const int *mYOffset = yOffset + jump;
        FloatType sum = 0;
        for (int i = yBegin; i <= yLim; i += yStride, ++inY) {
            int inX = ix;
            int cy = y - i;
            for (int j = xBegin; j <= xLim; j += xStride, ++inX) {
                int cx = x - j;
                if (mYOffset[inX] == cy && mXOffset[inX] == cx)
                    sum += mInput[inX];
            }
            mYOffset += inputWidth;
            mXOffset += inputWidth;
            mInput += inputWidth;
        }
        out[index] = sum;
    }
}

void maxPoolingBP(const FloatType *in, int inputWidth, int inputHeight,
                  FloatType *out, int outputWidth, int outputHeight,
                  const int *xOffset, const int *yOffset,
                  int windowWidth, int windowHeight,
                  int xStride, int yStride,
                  int channelCount
)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int len = outputWidth * outputHeight * channelCount;
    int blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    cuMaxPoolingBP<<<blocks, threadsPerBlock>>>(
            in, inputWidth, inputWidth * inputHeight,
                    out, outputWidth, outputHeight, outputWidth * outputHeight,
                    xOffset, yOffset, windowWidth, windowHeight,
                    xStride, yStride, len
    );
}

__global__ void cuGetCIFAR10Data(FloatType *data, const unsigned char *buffer, const int *indices, int len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        int i = indices[index / (32 * 32 * 3)];
        data[index] = static_cast<FloatType>(buffer[i * 3073 + 1 + index % (32 * 32 * 3)]) / 0xff;
    }
}

__global__ void cuGetCIFAR10Label(FloatType *label, const unsigned char *buffer, const int *indices, int len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        int i = indices[index / 10];
        label[index] = buffer[i * 3073] == (index % 10) ? 1 : 0;
    }
}

void getCIFAR10Batch(FloatType *data, FloatType *labels,
                     const unsigned char *buffer,
                     const int *indices, int count
)
{
    int threadsPerBlocks = DEF_THREADS_PER_BLOCK;
    int len = 32 * 32 * 3 * count;
    int blocks = (len + threadsPerBlocks - 1) / threadsPerBlocks;
    cuGetCIFAR10Data<<<blocks, threadsPerBlocks>>>(data, buffer, indices, len);

    if (labels) {
        len = 10 * count;
        blocks = (len + threadsPerBlocks - 1) / threadsPerBlocks;
        cuGetCIFAR10Label<<<blocks, threadsPerBlocks>>>(labels, buffer, indices, len);
    }
}

__global__ void cuFillN(FloatType *arr, int len, FloatType v)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        arr[index] = v;
    }
}

void m_fill_n(FloatType *arr, int len, FloatType v)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    cuFillN<<<blocks, threadsPerBlock>>>(arr, len, v);
}

__global__ void cuIncArray(int *arr, int len, int bias)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        arr[index] = index + bias;
    }
}

void incArray(int *arr, int len, int bias)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    cuIncArray<<<blocks, threadsPerBlock>>>(arr, len, bias);
}

__global__ void cuShuffle(int *arr, int len, unsigned long seed)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index == 0) {
        curandState state;
        curand_init(seed, static_cast<unsigned long long int>(threadIdx.x), 0, &state);
        for (int i = len - 1; i > 0; --i) {
            auto d = static_cast<int>(curand_uniform(&state) * (i + 1));
            if (d != i) {
                int tmp = arr[d];
                arr[d] = arr[i];
                arr[i] = tmp;
            }
        }
    }
}

void randomPermutation(int *arr, int len, int bias)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    cuIncArray<<<blocks, threadsPerBlock>>>(arr, len, bias);
    cuShuffle<<<1, 32>>>(arr, len, static_cast<unsigned long>(clock()));
}

//以下尚未测试

void scaleV(FloatType *v, FloatType n, int dim)
{
    M_CUBLAS_SCALE_V(M_CUBLAS_HANDLE, dim, &n, v, 1);
}

__global__ void cuBatchNormalize(FloatType *out, const FloatType *x, const FloatType *avg, const FloatType *oneDivDev, int dim, int len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        int lIndex = index % dim;
        out[index] = (x[index] - avg[lIndex]) * oneDivDev[lIndex];
    }
}

void batchNormalize(FloatType *out, const FloatType *x, const FloatType *avg, const FloatType *oneDivDev, int dim, int batchSize)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = ((dim * batchSize) + threadsPerBlock - 1) / threadsPerBlock;
    cuBatchNormalize<<<blocks, threadsPerBlock>>>(out, x, avg, oneDivDev, dim, dim * batchSize);
}

__global__ void cuBnForward(FloatType *out, FloatType *normOut, const FloatType *x, const FloatType *avg, const FloatType *oneDivDev, const FloatType *gamma, const FloatType *beta, int dim, int len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        int lIndex = index % dim;
        FloatType norm = (x[index] - avg[lIndex]) * oneDivDev[lIndex];
        normOut[index] = norm;
        out[index] = norm * gamma[lIndex] + beta[lIndex];
    }
}

void bnForward(FloatType *out, FloatType *normOut, const FloatType *x, const FloatType *avg, const FloatType *oneDivDev, const FloatType *gamma, const FloatType *beta, int dim, int batchSize)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = ((dim * batchSize) + threadsPerBlock - 1) / threadsPerBlock;
    cuBnForward<<<blocks, threadsPerBlock>>>(out, normOut, x, avg, oneDivDev, gamma, beta, dim, dim * batchSize);
}

__global__ void cuBnAvg(FloatType *avg, FloatType *xSubAvg, const FloatType *x, int dim, int batchSize)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dim) {
        FloatType sum = 0;
        const FloatType *cx = x + index;
        for (int i = 0; i < batchSize; ++i) {
            sum += *cx;
            cx += dim;
        }
        avg[index] = sum /= batchSize;
        sum *= -1;
        FloatType *cv = xSubAvg + index;
        cx = x + index;
        for (int i = 0; i < batchSize; ++i) {
            *cv = *cx + sum;
            cv += dim;
            cx += dim;
        }
    }
}

void bnAvg(FloatType *avg, FloatType *xSubAvg, const FloatType *x, int dim, int batchSize)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = (dim + threadsPerBlock - 1) / threadsPerBlock;
    cuBnAvg<<<blocks, threadsPerBlock>>>(avg, xSubAvg, x, dim, batchSize);
}

__global__ void cuBnOneDivDev(FloatType *var, FloatType *oneDivDev, const FloatType *xSubAvg, int dim, int batchSize)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dim) {
        FloatType sum = 0;
        const FloatType *curXSubAvg = xSubAvg + index;
        for (int i = 0; i < batchSize; ++i) {
            FloatType a = *curXSubAvg;
            sum += a * a;
            curXSubAvg += dim;
        }
        var[index] = sum /= batchSize;

        oneDivDev[index] = static_cast<FloatType>(1) / CU_SQRT(sum + static_cast<FloatType>(1e-4));
    }
}

void bnOneDivDev(FloatType *var, FloatType *oneDivDev, const FloatType *xSubAvg, int dim, int batchSize)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = (dim + threadsPerBlock - 1) / threadsPerBlock;
    cuBnOneDivDev<<<blocks, threadsPerBlock>>>(var, oneDivDev, xSubAvg, dim, batchSize);
}

__global__ void cuBnDeltaMulCenter(FloatType *out, const FloatType *delta, const FloatType *xSubAvg, int dim, int batchSize)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dim) {
        FloatType sum = 0;
        for (int i = 0; i < batchSize; ++i) {
            sum += delta[i * dim + index] * xSubAvg[i * dim + index];
        }
        out[index] = sum;
    }
}

void bnDeltaMulCenter(FloatType *out, const FloatType *delta, const FloatType *xSubAvg, int dim, int batchSize)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = (dim + threadsPerBlock - 1) / threadsPerBlock;
    cuBnDeltaMulCenter<<<blocks, threadsPerBlock>>>(out, delta, xSubAvg, dim, batchSize);
}

__global__ void cuBnBackProp(FloatType *out, const FloatType *gamma, const FloatType *normDelta, const FloatType *normOut, const FloatType *var, const FloatType * deltaMulCenter, int dim, int batchSize, int len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        int lIndex = index % dim;
        out[index] = gamma[lIndex] * (normDelta[index] - normOut[index] * deltaMulCenter[lIndex] / ((var[lIndex] + static_cast<FloatType>(1e-4)) * batchSize));
    }
}

void bnBackProp(FloatType *out, const FloatType *gamma, const FloatType *normDelta, const FloatType *normOut, const FloatType *var, const FloatType *deltaMulCenter, int dim, int batchSize)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = ((dim * batchSize) + threadsPerBlock - 1) / threadsPerBlock;
    cuBnBackProp<<<blocks, threadsPerBlock>>>(out, gamma, normDelta, normOut, var, deltaMulCenter, dim, batchSize, dim * batchSize);
}

__global__ void cuBnGammaGradients(FloatType *gamma, const FloatType *delta, const FloatType *normOut, int dim, int batchSize)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dim) {
        FloatType sum = 0;
        for (int i = 0; i < batchSize; ++i) {
            sum += delta[i * dim + index] * normOut[i * dim + index];
        }
        gamma[index] = sum;
    }
}

__global__ void cuBnBetaGradients(FloatType *beta, const FloatType *delta, int dim, int batchSize)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dim) {
        FloatType sum = 0;
        for (int i = 0; i < batchSize; ++i) {
            sum += delta[i * dim + index];
        }
        beta[index] = sum;
    }
}

void bnGradients(FloatType *gamma, FloatType *beta, const FloatType *delta, const FloatType *normOut, int dim, int batchSize)
{
    int threadsPerBlock = std::min((dim + 31) / 32 * 32, DEF_THREADS_PER_BLOCK);
    int blocks = (dim + threadsPerBlock - 1) / threadsPerBlock;
    cuBnGammaGradients<<<blocks, threadsPerBlock>>>(gamma, delta, normOut, dim, batchSize);
    cuBnBetaGradients<<<blocks, threadsPerBlock>>>(beta, delta, dim, batchSize);
}

__global__ void cuBnGlobalValues(FloatType *globalAvg, FloatType *globalVar, const FloatType *avg, const FloatType *var, int dim)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dim) {
        globalAvg[index] = globalAvg[index] * static_cast<FloatType>(0.8) + avg[index] * static_cast<FloatType>(0.2);
        globalVar[index] = globalVar[index] * static_cast<FloatType>(0.8) + var[index] * static_cast<FloatType>(0.2);
    }
}

void bnGlobalValues(FloatType *globalAvg, FloatType *globalVar, const FloatType *avg, const FloatType *var, int dim)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = (dim + threadsPerBlock - 1) / threadsPerBlock;
    cuBnGlobalValues<<<blocks, threadsPerBlock>>>(globalAvg, globalVar, avg, var, dim);
}

__global__ void cuBnGlobalOneDivDev(FloatType *globalOneDivDev, const FloatType *globalVar, int dim)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dim) {
        globalOneDivDev[index] = static_cast<FloatType>(1) / CU_SQRT(globalVar[index] + static_cast<FloatType>(1e-4));
    }
}

void bnGlobalOneDivDev(FloatType *globalOneDivDev, const FloatType *globalVar, int dim)
{
    int threadsPerBlock = DEF_THREADS_PER_BLOCK;
    int blocks = (dim + threadsPerBlock - 1) / threadsPerBlock;
    cuBnGlobalOneDivDev<<<blocks, threadsPerBlock>>>(globalOneDivDev, globalVar, dim);
}

#endif //ENABLE_CUDA