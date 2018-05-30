/**
 * Created by wjy50 on 18-4-24.
 */

#include <cstring>
#include <iostream>
#include "PoolingLayer.h"

using namespace ffw;
using namespace std;

PoolingLayer::PoolingLayer(int inputWidth, int inputHeight,
                           int windowWidth, int windowHeight,
                           int xStride, int yStride,
                           int channel,
                           ffw::PoolingMethod poolingMethod
) : AbsLayer(((inputWidth - windowWidth) / xStride + 1) * ((inputHeight - windowHeight) / yStride + 1) * channel,
             inputWidth * inputHeight * channel)
{
    this->inputWidth = inputWidth;
    this->inputHeight = inputHeight;
    inputSize = inputWidth * inputHeight;
    this->windowWidth = windowWidth;
    this->windowHeight = windowHeight;
    windowSize = windowWidth * windowHeight;
    this->xStride = xStride;
    this->yStride = yStride;
    outputWidth = (inputWidth - windowWidth) / xStride + 1;
    outputHeight = (inputHeight - windowHeight) / yStride + 1;
    outputSize = outputWidth * outputHeight;
    this->channel = channel;

    this->poolingMethod = poolingMethod;

    output = nullptr;
    xOffset = nullptr;
    yOffset = nullptr;
}

#if !ENABLE_CUDA

void PoolingLayer::maxPooling(const FloatType *x, FloatType *output, int *xOffset, int *yOffset)
{
    //long st = clock();
    if (xOffset && yOffset) {
        for (int i = 0; i < channel; ++i) {
            const FloatType *in = x + i * inputSize;
            FloatType *out = output + i * outputSize;
            int *xo = xOffset + i * outputSize;
            int *yo = yOffset + i * outputSize;
            int oy = 0;
            for (int j = 0; j <= inputHeight - windowHeight; j += yStride, ++oy) {
                int ox = 0;
                for (int k = 0; k <= inputWidth - windowWidth; k += xStride, ++ox) {
                    int maxR = 0;
                    int maxC = indexOfMax(in + j * inputWidth + k, windowWidth);
                    FloatType max = in[j * inputWidth + k + maxC];
                    for (int l = 1; l < windowHeight; ++l) {
                        int c = indexOfMax(in + (j + l) * inputWidth + k, windowWidth);
                        if (in[(j + l) * inputWidth + k + c] > max) {
                            max = in[(j + l) * inputWidth + k + c];
                            maxR = l;
                            maxC = c;
                        }
                    }
                    const int o = oy * outputWidth + ox;
                    out[o] = max;
                    xo[o] = maxC;
                    yo[o] = maxR;
                }
            }
        }
    } else {
        for (int i = 0; i < channel; ++i) {
            const FloatType *in = x + i * inputSize;
            FloatType *out = output + i * outputSize;
            int oy = 0;
            for (int j = 0; j <= inputHeight - windowHeight; j += yStride, ++oy) {
                int ox = 0;
                for (int k = 0; k <= inputWidth - windowWidth; k += xStride, ++ox) {
                    int c = indexOfMax(in + j * inputWidth + k, windowWidth);
                    FloatType max = in[j * inputWidth + k + c];
                    for (int l = 1; l < windowHeight; ++l) {
                        c = indexOfMax(in + (j + l) * inputWidth + k, windowWidth);
                        if (in[(j + l) * inputWidth + k + c] > max) {
                            max = in[(j + l) * inputWidth + k + c];
                        }
                    }
                    out[oy * outputWidth + ox] = max;
                }
            }
        }
    }
    /*long t1 = clock() - st;

    st = clock();
    const FloatType *in = x;
    int *xo = xOffset;
    int *yo = yOffset;
    FloatType *out = output;
    for (int i = 0; i < channel; ++i, in += inputSize) {
        int inRowOffset = 0;
        int outRowOffset = 0;
        for (int j = 0, oy = 0; j < inputHeight; j += yStride, ++oy) {
            for (int k = 0, ox = 0; k < inputWidth; k += xStride, ++ox) {
                int maxC = indexOfMax(in + inRowOffset + k, xStride);
                int outOffset = outRowOffset + ox;
                out[outOffset] = in[k + inRowOffset + maxC];
                xo[outOffset] = maxC;
                yo[outOffset] = 0;
            }
            for (int k = 1; k < yStride; ++k) {
                inRowOffset += inputWidth;
                for (int l = 0, ox = 0; l < inputWidth; l += xStride, ++ox) {
                    int maxC = indexOfMax(in + inRowOffset + l, xStride);
                    FloatType m = in[l + inRowOffset + maxC];
                    int outOffset = outRowOffset + ox;
                    if (out[outOffset] < m) {
                        out[outOffset] = m;
                        xo[outOffset] = maxC;
                        yo[outOffset] = k;
                    }
                }
            }
            outRowOffset += outputWidth;
        }

        xo += outputSize;
        yo += outputSize;
        out += outputSize;
    }

    cout << clock() - st - t1 << endl;*/
}

void PoolingLayer::meanPooling(const FloatType *x, FloatType *output)
{
    for (int i = 0; i < channel; ++i) {
        const FloatType *in = x + i * inputSize;
        FloatType *out = output + i * outputSize;
        int oy = 0;
        for (int j = 0; j <= inputHeight - windowHeight; j += yStride, ++oy) {
            int ox = 0;
            for (int k = 0; k <= inputWidth - windowWidth; k += xStride, ++ox) {
                FloatType sum = 0;
                for (int l = 0; l < windowHeight; ++l) {
                    for (int m = 0; m < windowWidth; ++m) {
                        sum += in[(j + l) * inputWidth + k + m];
                    }
                }
                out[oy * outputWidth + ox] = sum / windowSize;
            }
        }
    }
}

void PoolingLayer::maxBackProp(FloatType *backPropDelta, const FloatType *delta, int *xOffset, int *yOffset)
{
    for (int i = 0; i < channel; ++i) {
        FloatType *od = backPropDelta + i * inputSize;
        const FloatType *d = delta + i * outputSize;
        int *xo = xOffset + i * outputSize;
        int *yo = yOffset + i * outputSize;
        int oy = 0;
        for (int j = 0; j <= inputHeight - windowHeight; j += yStride, ++oy) {
            int ox = 0;
            for (int k = 0; k <= inputWidth - windowWidth; k += xStride, ++ox) {
                const int o = oy * outputWidth + ox;
                od[(j + yo[o]) * inputWidth + k + xo[o]] += d[o];
            }
        }
    }
}

void PoolingLayer::meanBackProp(FloatType *backPropDelta, const FloatType *delta)
{
    for (int i = 0; i < channel; ++i) {
        FloatType *od = backPropDelta + i * inputSize;
        const FloatType *d = delta + i * outputSize;
        int oy = 0;
        for (int j = 0; j <= inputHeight - windowHeight; j += yStride, ++oy) {
            int ox = 0;
            for (int k = 0; k <= inputWidth - windowWidth; k += xStride, ++ox) {
                const FloatType di = d[oy * outputWidth + ox] / windowSize;
                for (int l = 0; l < windowHeight; ++l) {
                    for (int m = 0; m < windowWidth; ++m) {
                        od[(j + l) * inputWidth + k + m] += di;
                    }
                }
            }
        }
    }
}

int PoolingLayer::indexOfMax(const FloatType *arr, int n)
{
    int maxI = 0;
    FloatType max = arr[0];
    for (int i = 1; i < n; ++i) {
        if (arr[i] > max) {
            maxI = i;
            max = arr[i];
        }
    }
    return maxI;
}

#endif //ENABLE_CUDA

void PoolingLayer::feedForward(const FloatType *x)
{
#if ENABLE_CUDA
    //TODO cuda
#else

    switch (poolingMethod) {
        case MAX_POOLING:
            maxPooling(x, output, nullptr, nullptr);
            break;
        case MEAN_POOLING:
            meanPooling(x, output);
            break;
    }

#endif
}

void PoolingLayer::feedForwardForOptimization(const FloatType *x)
{
#if ENABLE_CUDA
    //TODO cuda
#else

    switch (poolingMethod) {
        case MAX_POOLING:
            for (int m = 0; m < miniBatchSize; ++m) {
                maxPooling(x + inputDim * m, output + neuronCount * m, xOffset + neuronCount * m, yOffset + neuronCount * m);
            }
            break;
        case MEAN_POOLING:
            for (int m = 0; m < miniBatchSize; ++m) {
                meanPooling(x + inputDim * m, output + neuronCount * m);
            }
            break;
    }

#endif
}

void PoolingLayer::computeBackPropDelta(FloatType *backPropDelta)
{
#if ENABLE_CUDA
    //TODO cuda
#else

    memset(backPropDelta, 0, inputDim * miniBatchSize * sizeof(FloatType));
    switch (poolingMethod) {
        case MAX_POOLING:
            for (int m = 0; m < miniBatchSize; ++m) {
                maxBackProp(backPropDelta + inputDim * m, delta + neuronCount * m, xOffset + neuronCount * m, yOffset + neuronCount * m);
            }
            break;
        case MEAN_POOLING:
            for (int m = 0; m < miniBatchSize; ++m) {
                meanBackProp(backPropDelta + inputDim * m, delta + neuronCount * m);
            }
            break;
    }

#endif
}

void PoolingLayer::backPropagateDelta()
{
    /*池化层不需要更新参数，什么都不用做*/
}

int PoolingLayer::getOutputWidth()
{
    return outputWidth;
}

int PoolingLayer::getOutputHeight()
{
    return outputHeight;
}

int PoolingLayer::getChannelCount()
{
    return channel;
}

void PoolingLayer::initialize(int miniBatchSize)
{
    this->miniBatchSize = miniBatchSize;
#if ENABLE_CUDA

    cudaMalloc(&output, outputSize * channel * miniBatchSize * sizeof(FloatType));
    if (poolingMethod != MEAN_POOLING) {
        cudaMalloc(&xOffset, outputSize * channel * miniBatchSize * sizeof(int));
        cudaMalloc(&yOffset, outputSize * channel * miniBatchSize * sizeof(int));
    }
    cudaMalloc(&delta, outputSize * channel * miniBatchSize * sizeof(FloatType));

#else

    output = new FloatType[outputSize * channel * miniBatchSize];
    if (poolingMethod != MEAN_POOLING) {
        xOffset = new int[outputSize * channel * miniBatchSize];
        yOffset = new int[outputSize * channel * miniBatchSize];
    }
    delta = new FloatType[outputSize * channel * miniBatchSize];

#endif
}

const FloatType *PoolingLayer::getWeightedOutput()
{
    return output;
}

const FloatType *PoolingLayer::getActivationOutput()
{
    return output;
}

FloatType *PoolingLayer::getDelta()
{
    return delta;
}

void PoolingLayer::computeGradient(const FloatType *prevActivation)
{
    //没有梯度，什么都不用做
}

void PoolingLayer::updateParameters()
{
    //没有参数，什么都不用做
}

void PoolingLayer::computeOutputDelta(const FloatType *y)
{
    //池化层不能作为输出层，什么都不做
}

int PoolingLayer::getWeightCount()
{
    return 0;
}

int PoolingLayer::getBiasCount()
{
    return 0;
}

PoolingLayer::~PoolingLayer()
{
#if ENABLE_CUDA
    cudaFree(output);
    cudaFree(xOffset);
    cudaFree(yOffset);
    cudaFree(delta);
#else
    delete[] output;
    delete[] xOffset;
    delete[] yOffset;
    delete[] delta;
#endif
}