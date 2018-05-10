/**
 * Created by wjy50 on 18-4-24.
 */

#include <cstring>
#include <random>
#include <cassert>
#include <iostream>

#include "ConvLayer.h"

using namespace ffw;
using namespace std;

ConvLayer::ConvLayer(int inputWidth, int inputHeight, int inputChannel, int kernelWidth,
                     int kernelHeight,
                     int kernelCount, int xStride, int yStride, int xPadding, int yPadding,
                     Activator activator)
        : AbsLayer(((xPadding * 2 + inputWidth - kernelWidth) / xStride + 1) *
                   ((yPadding * 2 + inputHeight - kernelHeight) / yStride + 1) * kernelCount,
                   inputChannel * inputWidth * inputHeight)
{
    this->inputWidth = inputWidth;
    this->inputHeight = inputHeight;
    this->inputChannel = inputChannel;
    inputSize = inputWidth * inputHeight;

    this->xStride = xStride;
    this->yStride = yStride;
    this->xPadding = xPadding;
    this->yPadding = yPadding;

    this->kernelWidth = kernelWidth;
    this->kernelHeight = kernelHeight;
    this->kernelCount = kernelCount;
    kernelSize = kernelWidth * kernelHeight;

    outputWidth = (xPadding * 2 + inputWidth - kernelWidth) / xStride + 1;
    outputHeight = (yPadding * 2 + inputHeight - kernelHeight) / yStride + 1;
    outputSize = outputWidth * outputHeight;

    weightCount = kernelSize * inputChannel * kernelCount;
    kernels = new FloatType[weightCount];
    biases = new FloatType[kernelCount]();

    weightGradient = new FloatType[weightCount];
    biasGradient = new FloatType[kernelCount];

    z = nullptr;
    a = nullptr;

    delta = nullptr;

    assert(activator != OUTPUT_ACTIVATOR);/*（目前）卷积层不能作为输出层*/
    this->activator = activator;
    activation = ACTIVATION_FUNCTIONS[activator];
    dActivation_dx = D_ACTIVATION_FUNCTIONS[activator];

    right = inputWidth + xPadding - kernelWidth;
    bottom = inputHeight + yPadding - kernelHeight;
}

void ConvLayer::initialize(int miniBatchSize)
{
    memset(biases, 0, kernelCount * sizeof(FloatType));

    random_device rd;
    normal_distribution<FloatType> distribution(0, sqrt(static_cast<FloatType>(2) / inputDim));
    int l = kernelSize * inputChannel * kernelCount;
    for (int i = 0; i < l; ++i) {
        kernels[i] = distribution(rd);
    }

    this->miniBatchSize = miniBatchSize;
    z = new FloatType[outputSize * kernelCount * miniBatchSize];
    a = new FloatType[outputSize * kernelCount * miniBatchSize];
    delta = new FloatType[outputSize * kernelCount * miniBatchSize];
}

#define NEW_CONV

void ConvLayer::convolution(const FloatType *input, const FloatType *kernel, FloatType *out)
{
    for (int i = 0; i < kernelHeight; ++i) {
        int jLim = min(inputHeight - 1, bottom + i);
        int kerRowOffset = i * kernelWidth;
        int oy = 0, j = i - yPadding;
        while (j < 0) {
            oy++;
            j += yStride;
        }
        for (; j <= jLim; j += yStride, oy++) {
            int inRowOffset = j * inputWidth;
            int outRowOffset = oy * outputWidth;
            for (int l = -xPadding, ox = 0; l <= right; l += xStride, ox++) {
                FloatType sum = 0;
                for (int m = max(0, -l); m < kernelWidth; ++m) {
                    sum += kernel[kerRowOffset + m] * input[inRowOffset + l + m];
                }
                out[outRowOffset + ox] += sum;
            }
        }
    }
}

void ConvLayer::feedForward(const FloatType *x)
{
    FloatType *out = a;
    for (int i = 0; i < kernelCount; ++i, out += outputSize) {
        const FloatType b = biases[i];
        for (int j = 0; j < outputSize; ++j) {
            out[j] = b;
        }
    }
#ifdef NEW_CONV
    const FloatType *ker = kernels;
    out = a;
    for (int i = 0; i < kernelCount; ++i, out += outputSize) {
        for (int ch = 0; ch < inputChannel; ++ch, ker += kernelSize) {
            convolution(x + ch * inputSize, ker, out);
        }
    }
#else
    const FloatType *ker = kernels;
    for (int i = 0; i < kernelCount; ++i) {
        FloatType *out = z + i * outputSize;
        for (int j = 0; j < inputChannel; ++j) {
            const FloatType *in = x + j * inputSize;
            int oy = 0;
            for (int k = -yPadding; k <= bottom; k += yStride, oy++) {
                int ox = 0;
                for (int l = -xPadding; l <= right; l += xStride, ox++) {
                    int inputX = l < 0 ? 0 : l;
                    int kerX = inputX - l;
                    int inputY = k < 0 ? 0 : k;
                    int kerY = inputY - k;
                    const int w = min(l + kernelWidth, inputWidth) - inputX;
                    const int h = min(k + kernelHeight, inputHeight) - inputY;
                    FloatType sum = 0;
                    for (int m = 0; m < h; ++m) {
                        for (int n = 0; n < w; ++n) {
                            sum += ker[(kerY+m)*kernelWidth+kerX+n] * in[(inputY+m)*inputWidth+inputX+n];
                        }
                    }
                    out[oy*outputWidth+ox] += sum;
                }
            }
            ker += kernelSize;
        }
    }
#endif
    for (int i = 0; i < neuronCount; ++i) {
        a[i] = activation(a[i]);
    }
}

void ConvLayer::feedForwardForOptimization(const FloatType *x)
{
    FloatType *out = z;
    for (int m = 0; m < miniBatchSize; ++m) {
        for (int i = 0; i < kernelCount; ++i, out += outputSize) {
            const FloatType b = biases[i];
            for (int j = 0; j < outputSize; ++j) {
                out[j] = b;
            }
        }
    }

    out = z;
    const FloatType *curX = x;
    for (int m = 0; m < miniBatchSize; ++m) {
        const FloatType *ker = kernels;
        for (int i = 0; i < kernelCount; ++i, out += outputSize) {
            for (int ch = 0; ch < inputChannel; ++ch, ker += kernelSize) {
                convolution(curX + ch * inputSize, ker, out);
            }
        }
        curX += inputDim;
    }

    for (int i = 0; i < neuronCount * miniBatchSize; ++i) {
        a[i] = activation(z[i]);
    }
}

void ConvLayer::backPropagateDelta()
{
    for (int i = 0; i < neuronCount * miniBatchSize; ++i) {
        delta[i] *= dActivation_dx(z[i]);
    }
}

void ConvLayer::computeBackPropDelta(FloatType *backPropDelta)
{
    memset(backPropDelta, 0, inputDim * miniBatchSize * sizeof(FloatType));
#ifdef NEW_CONV
    const FloatType *curDelta = delta;
    FloatType *curBPD = backPropDelta;
    for (int b = 0; b < miniBatchSize; ++b) {
        const FloatType *ker = kernels;
        for (int i = 0; i < kernelCount; ++i, curDelta += outputSize) {
            for (int ch = 0; ch < inputChannel; ++ch, ker += kernelSize) {
                FloatType *back = curBPD + ch * inputSize;
                for (int j = 0; j < kernelHeight; ++j) { /*kernel的第j行与输入第k行*/
                    int kLim = min(inputHeight - 1, bottom + j);
                    int kerRowOffset = j * kernelWidth;
                    int oy = 0, k = j - yPadding;
                    while (k < 0) {
                        oy++;
                        k += yStride;
                    }
                    for (; k <= kLim; k += yStride, oy++) {
                        int inRowOffset = k * inputWidth;
                        int fromRowOffset = oy * outputWidth;
                        for (int l = -xPadding, ox = 0; l <= right; l += xStride, ox++) {
                            FloatType di = curDelta[fromRowOffset + ox];
                            for (int m = max(0, -l); m < kernelWidth; ++m) {
                                back[inRowOffset + l + m] += ker[kerRowOffset + m] * di;
                            }
                        }
                    }
                }
            }
        }
        curBPD += inputDim;
    }
#else
    const FloatType *ker = kernels;
    for (int i = 0; i < kernelCount; ++i) {
        const FloatType *from = delta + i * outputSize;
        for (int j = 0; j < inputChannel; ++j) {
            FloatType *back = backPropDelta + j * inputSize;
            int oy = 0;
            for (int k = -yPadding; k <= bottom; k += yStride, oy++) {
                int ox = 0;
                for (int l = -xPadding; l <= right; l += xStride, ox++) {
                    int inputX = l < 0 ? 0 : l;
                    int kerX = inputX - l;
                    int inputY = k < 0 ? 0 : k;
                    int kerY = inputY - k;
                    const int w = min(l+kernelWidth, inputWidth) - inputX;
                    const int h = min(k+kernelHeight, inputHeight) - inputY;
                    FloatType di = from[oy*outputWidth+ox];
                    for (int m = 0; m < h; ++m) {
                        for (int n = 0; n < w; ++n) {
                            back[(inputY+m)*inputWidth+inputX+n] += ker[(kerY+m)*kernelWidth+kerX+n] * di;
                        }
                    }
                }
            }
            ker += kernelSize;
        }
    }
#endif
}

void ConvLayer::computeGradient(const FloatType *prevActivation)
{
    memset(biasGradient, 0, kernelCount * sizeof(FloatType));
    memset(weightGradient, 0, kernelSize * inputChannel * kernelCount * sizeof(FloatType));
    const FloatType *curDelta = delta;
    for (int m = 0; m < miniBatchSize; ++m) {
        for (int i = 0; i < kernelCount; ++i, curDelta += outputSize) {
            FloatType sum = 0;
            for (int j = 0; j < outputSize; ++j) {
                sum += curDelta[j];
            }
            biasGradient[i] += sum;
        }
    }
    for (int i = 0; i < kernelCount; ++i) {
        biasGradient[i] /= miniBatchSize;
    }
#ifdef NEW_CONV
    curDelta = delta;
    const FloatType *curPA = prevActivation;
    for (int b = 0; b < miniBatchSize; ++b) {
        FloatType *kerG = weightGradient;
        for (int i = 0; i < kernelCount; ++i, curDelta += outputSize) {
            for (int ch = 0; ch < inputChannel; ++ch, kerG += kernelSize) {
                const FloatType *pa = curPA + ch * inputSize;
                for (int j = 0; j < kernelHeight; ++j) { /*kernel的第j行与输入第k行*/
                    int kLim = min(inputHeight - 1, bottom + j);
                    int kerRowOffset = j * kernelWidth;
                    int oy = 0, k = j - yPadding;
                    while (k < 0) {
                        oy++;
                        k += yStride;
                    }
                    for (; k <= kLim; k += yStride, oy++) {
                        int inRowOffset = k * inputWidth;
                        int fromRowOffset = oy * outputWidth;
                        for (int l = -xPadding, ox = 0; l <= right; l += xStride, ox++) {
                            FloatType di = curDelta[fromRowOffset + ox];
                            for (int m = max(0, -l); m < kernelWidth; ++m) {
                                kerG[kerRowOffset + m] += di * pa[inRowOffset + l + m];
                            }
                        }
                    }
                }
            }
        }
        curPA += inputDim;
    }
#else
    FloatType *kerG = weightGradient;
    for (int i = 0; i < kernelCount; ++i) {
        const FloatType *from = delta + i * outputSize;
        for (int j = 0; j < inputChannel; ++j) {
            const FloatType *pa = prevActivation + j * inputSize;
            int oy = 0;
            for (int k = -yPadding; k <= bottom; k += yStride, oy++) {
                int ox = 0;
                for (int l = -xPadding; l <= right; l += xStride, ox++) {
                    int inputX = l < 0 ? 0 : l;
                    int kerX = inputX - l;
                    int inputY = k < 0 ? 0 : k;
                    int kerY = inputY - k;
                    const int w = min(l+kernelWidth, inputWidth) - inputX;
                    const int h = min(k+kernelHeight, inputHeight) - inputY;
                    FloatType di = from[oy*outputWidth+ox];
                    for (int m = 0; m < h; ++m) {
                        for (int n = 0; n < w; ++n) {
                            kerG[(kerY+m)*kernelWidth+kerX+n] += di * pa[(inputY+m)*inputWidth+inputX+n];
                        }
                    }
                }
            }
            kerG += kernelSize;
        }
    }
#endif
    for (int i = 0; i < weightCount; ++i) {
        weightGradient[i] /= miniBatchSize;
    }
}

void ConvLayer::updateParameters()
{
    optimizer->update(kernels, weightGradient, biases, biasGradient);
}

const FloatType *ConvLayer::getWeightedOutput()
{
    return z;
}

const FloatType *ConvLayer::getActivationOutput()
{
    return a;
}

int ConvLayer::getOutputWidth()
{
    return outputWidth;
}

int ConvLayer::getOutputHeight()
{
    return outputHeight;
}

int ConvLayer::getKernelCount()
{
    return kernelCount;
}

FloatType *ConvLayer::getDelta()
{
    return delta;
}

int ConvLayer::getWeightCount()
{
    return weightCount;
}

int ConvLayer::getBiasCount()
{
    return kernelCount;
}

void ConvLayer::computeOutputDelta(const FloatType *y)
{
    //（目前）卷积层不能作为输出层，什么都不做
    assert(false);
}

ConvLayer::~ConvLayer()
{
    delete[] kernels;
    delete[] biases;
    delete[] weightGradient;
    delete[] biasGradient;
    delete[] z;
    delete[] a;
}