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

ConvLayer::ConvLayer(int inputWidth, int inputHeight, int inputChannel, int kernelWidth, int kernelHeight,
                     int kernelCount, int xStride, int yStride, int xPadding, int yPadding, Activator activator)
        : AbsLayer(((xPadding * 2 + inputWidth - kernelWidth) / xStride + 1) * ((yPadding * 2 + inputHeight - kernelHeight) / yStride + 1) * kernelCount, inputChannel * inputWidth * inputHeight)
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

    outputWidth = (xPadding*2 + inputWidth - kernelWidth) / xStride + 1;
    outputHeight = (yPadding*2 + inputHeight - kernelHeight) / yStride + 1;
    outputSize = outputWidth * outputHeight;

    kernels = new double[kernelSize * inputChannel * kernelCount];
    biases = new double[kernelCount];

    weightGradient = new double[kernelSize * inputChannel * kernelCount];
    biasGradient = new double[kernelCount];

    z = new double[outputSize * kernelCount];
    a = new double[outputSize * kernelCount];

    delta = new double[outputSize * kernelCount];

    assert(activator != OUTPUT_ACTIVATOR);/*（目前）卷积层不能作为输出层*/
    this->activator = activator;
    activation = ACTIVATION_FUNCTIONS[activator];
    dActivation_dx = D_ACTIVATION_FUNCTIONS[activator];

    regParam = 0;
}

void ConvLayer::initialize()
{
    memset(biases, 0, kernelCount * sizeof(double));

    random_device rd;
    normal_distribution<double> distribution(0, ((double)2/sqrt(inputDim)));
    for (int i = 0; i < kernelSize * inputChannel * kernelCount; ++i) {
        kernels[i] = distribution(rd);
    }
}

void ConvLayer::feedForward(const double *x)
{
    for (int i = 0; i < kernelCount; ++i) {
        double *out = z + i * outputSize;
        for (int j = 0; j < outputSize; ++j) {
            out[j] = biases[i];
        }
    }
    int right = inputWidth + xPadding - kernelWidth;
    int bottom = inputHeight + yPadding - kernelHeight;
    const double *ker = kernels;
    for (int i = 0; i < kernelCount; ++i) {
        double *out = z + i * outputSize;
        for (int j = 0; j < inputChannel; ++j) {
            const double *in = x + j * inputSize;
            int oy = 0;
            for (int k = -yPadding; k <= bottom; k += yStride, oy++) {
                int ox = 0;
                for (int l = -xPadding; l <= right; l += xStride, ox++) {
                    int inputX = max(l, 0);
                    int inputXLim = min(l+kernelWidth, inputWidth);
                    int inputY = max(k, 0);
                    int inputYLim = min(k+kernelHeight, inputHeight);
                    int kerX = inputX - l;
                    int kerY = inputY - k;
                    int w = inputXLim - inputX;
                    int h = inputYLim - inputY;
                    double sum = 0;
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
    for (int i = 0; i < neuronCount; ++i) {
        a[i] = activation(z[i]);
    }
    /*for (int i = 0; i < outputHeight; ++i) {
        for (int j = 0; j < outputWidth; ++j) {
            cout << (int)(a[i*outputWidth+j]*10) << " ";
        }
        cout << endl;
    }
    cout << endl;
    for (int i = 0; i < outputHeight; ++i) {
        for (int j = 0; j < outputWidth; ++j) {
            cout << ((int)(a[i*outputWidth+j]*10) == 0 ? ' ' : '0') << " ";
        }
        cout << endl;
    }
    cout << endl;*/
}

void ConvLayer::backPropagateDelta()
{
    for (int i = 0; i < neuronCount; ++i) {
        delta[i] = delta[i] * dActivation_dx(z[i]);
    }
}

void ConvLayer::computeBackPropDelta(double *backPropDelta)
{
    memset(backPropDelta, 0, inputDim * sizeof(double));
    int right = inputWidth + xPadding - kernelWidth;
    int bottom = inputHeight + yPadding - kernelHeight;
    const double *ker = kernels;
    for (int i = 0; i < kernelCount; ++i) {
        const double *from = delta + i * outputSize;
        for (int j = 0; j < inputChannel; ++j) {
            double *back = backPropDelta + j * inputSize;
            int oy = 0;
            for (int k = -yPadding; k <= bottom; k += yStride, oy++) {
                int ox = 0;
                for (int l = -xPadding; l <= right; l += xStride, ox++) {
                    int inputX = max(l, 0);
                    int inputXLim = min(l+kernelWidth, inputWidth);
                    int inputY = max(k, 0);
                    int inputYLim = min(k+kernelHeight, inputHeight);
                    int kerX = inputX - l;
                    int kerY = inputY - k;
                    int w = inputXLim - inputX;
                    int h = inputYLim - inputY;
                    double di = from[oy*outputWidth+ox];
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
}

void ConvLayer::clearGradient()
{
    memset(biasGradient, 0, kernelCount * sizeof(double));
    memset(weightGradient, 0, kernelSize * inputChannel * kernelCount * sizeof(double));
}

void ConvLayer::accumulateGradient(const double *prevActivation)
{
    for (int i = 0; i < kernelCount; ++i) {
        const double *from = delta + i * outputSize;
        for (int j = 0; j < outputSize; ++j) {
            biasGradient[i] += from[j];
        }
    }
    int right = inputWidth + xPadding - kernelWidth;
    int bottom = inputHeight + yPadding - kernelHeight;
    double *kerG = weightGradient;
    for (int i = 0; i < kernelCount; ++i) {
        const double *from = delta + i * outputSize;
        for (int j = 0; j < inputChannel; ++j) {
            const double *pa = prevActivation + j * inputSize;
            int oy = 0;
            for (int k = -yPadding; k <= bottom; k += yStride, oy++) {
                int ox = 0;
                for (int l = -xPadding; l <= right; l += xStride, ox++) {
                    int inputX = max(l, 0);
                    int inputXLim = min(l+kernelWidth, inputWidth);
                    int inputY = max(k, 0);
                    int inputYLim = min(k+kernelHeight, inputHeight);
                    int kerX = inputX - l;
                    int kerY = inputY - k;
                    int w = inputXLim - inputX;
                    int h = inputYLim - inputY;
                    double di = from[oy*outputWidth+ox];
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
}

void ConvLayer::updateParameters(size_t batchSize, size_t trainSetSize)
{
    double eta = -learningRate;
    for (int i = 0; i < kernelCount; ++i) {
        biases[i] += eta * biasGradient[i] / batchSize;
    }
    double ws = (1 + eta * regParam / trainSetSize);
    for (int i = 0; i < kernelSize * inputChannel * kernelCount; ++i) {
        kernels[i] = kernels[i] * ws + eta * weightGradient[i] / batchSize;
    }
}

const double * ConvLayer::getWeightedOutput()
{
    return z;
}

const double * ConvLayer::getActivationOutput()
{
    return a;
}

void ConvLayer::setRegParam(double regParam)
{
    this->regParam = regParam;
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

double* ConvLayer::getDelta()
{
    return delta;
}

void ConvLayer::computeOutputDelta(const double *y)
{
    //（目前）卷积层不能作为输出层，什么都不做
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