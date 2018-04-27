/**
 * Created by wjy50 on 18-4-24.
 */

#include <cstring>
#include <iostream>
#include "PoolingLayer.h"

using namespace ffw;
using namespace std;

PoolingLayer::PoolingLayer(int inputWidth, int inputHeight, int xStride, int yStride, int channel, ffw::PoolingMethod poolingMethod) : AbsLayer((inputWidth/xStride)*(inputHeight/yStride)*channel, inputWidth*inputHeight*channel)
{
    this->inputWidth = inputWidth;
    this->inputHeight = inputHeight;
    inputSize = inputWidth * inputHeight;
    this->xStride = xStride;
    this->yStride = yStride;
    outputWidth = inputWidth / xStride;
    outputHeight = inputHeight / yStride;
    outputSize = outputWidth * outputHeight;
    this->channel = channel;

    this->poolingMethod = poolingMethod;

    output = new double[outputSize * channel];
    if (poolingMethod != MEAN_POOLING) {
        xOffset = new int[outputSize * channel];
        yOffset = new int[outputSize * channel];
    } else {
        xOffset = nullptr;
        yOffset = nullptr;
    }
    delta = new double[outputSize * channel];
}

void PoolingLayer::maxPooling(const double *x)
{
    for (int i = 0; i < channel; ++i) {
        const double *in = x+i*inputSize;
        double *out = output+i*outputSize;
        int *xo = xOffset+i*outputSize;
        int *yo = yOffset+i*outputSize;
        int oy = 0;
        for (int j = 0; j < inputHeight; j += yStride, ++oy) {
            int ox = 0;
            for (int k = 0; k < inputWidth; k += xStride, ++ox) {
                int maxR = 0;
                int maxC = indexOfMax(in + j*inputWidth + k, xStride);
                double max = in[j*inputWidth + k + maxC];
                for (int l = 1; l < yStride; ++l) {
                    int c = indexOfMax(in + (j+l)*inputWidth + k, xStride);
                    if (in[(j+l)*inputWidth + k + c] > max) {
                        max = in[(j+l)*inputWidth + k + c];
                        maxR = l;
                        maxC = c;
                    }
                }
                out[oy*outputWidth + ox] = max;
                xo[oy*outputWidth + ox] = maxC;
                yo[oy*outputWidth + ox] = maxR;
            }
        }
    }
    /*for (int i = 0; i < outputHeight; ++i) {
        for (int j = 0; j < outputWidth; ++j) {
            cout << (int)(output[i*outputWidth+j]*10) << " ";
        }
        cout << endl;
    }
    cout << endl;*/
}

void PoolingLayer::meanPooling(const double *x)
{
    for (int i = 0; i < channel; ++i) {
        const double *in = x+i*inputSize;
        double *out = output+i*outputSize;
        int oy = 0;
        for (int j = 0; j < inputHeight; j += yStride, ++oy) {
            int ox = 0;
            for (int k = 0; k < inputWidth; k += xStride, ++ox) {
                double sum = 0;
                for (int l = 0; l < yStride; ++l) {
                    for (int m = 0; m < xStride; ++m) {
                        sum += in[(j+l)*inputWidth + k + m];
                    }
                }
                out[oy*outputWidth + ox] = sum / (xStride * yStride);
            }
        }
    }
}

void PoolingLayer::feedForward(const double *x)
{
    switch (poolingMethod) {
        case MAX_POOLING:
            maxPooling(x);
            break;
        case MEAN_POOLING:
            meanPooling(x);
            break;
    }
}

void PoolingLayer::computeBackPropDelta(double *backPropDelta)
{
    switch (poolingMethod) {
        case MAX_POOLING:
            memset(backPropDelta, 0, channel * inputSize * sizeof(double));
            maxBackProp(backPropDelta);
            break;
        case MEAN_POOLING:
            meanBackProp(backPropDelta);
            break;
    }
}

void PoolingLayer::maxBackProp(double *backPropDelta)
{
    for (int i = 0; i < channel; ++i) {
        double *od = backPropDelta+i*inputSize;
        const double *d = delta+i*outputSize;
        int *xo = xOffset+i*outputSize;
        int *yo = yOffset+i*outputSize;
        int oy = 0;
        for (int j = 0; j < inputHeight; j += yStride, ++oy) {
            int ox = 0;
            for (int k = 0; k < inputWidth; k += xStride, ++ox) {
                od[(j+yo[oy*outputWidth+ox])*inputWidth + k + xo[oy*outputWidth+ox]] = d[oy*outputWidth+ox];
            }
        }
    }
}

void PoolingLayer::meanBackProp(double *backPropDelta)
{
    for (int i = 0; i < channel; ++i) {
        double *od = backPropDelta+i*inputSize;
        const double *d = delta+i*outputSize;
        int oy = 0;
        for (int j = 0; j < inputHeight; j += yStride, ++oy) {
            int ox = 0;
            for (int k = 0; k < inputWidth; k += xStride, ++ox) {
                for (int l = 0; l < yStride; ++l) {
                    for (int m = 0; m < xStride; ++m) {
                        od[(j+l)*inputWidth + k + m] = d[oy*outputWidth+ox] / (xStride * yStride);
                    }
                }
            }
        }
    }
}

void PoolingLayer::backPropagateDelta()
{
    /*池化层不需要更新参数，什么都不用做*/
}

int PoolingLayer::indexOfMax(const double *arr, int n)
{
    int max = 0;
    for (int i = 1; i < n; ++i) {
        if (arr[i] > arr[max]) max = i;
    }
    return max;
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

void PoolingLayer::initialize()
{
    //什么都不用做
}

const double * PoolingLayer::getWeightedOutput()
{
    return output;
}

const double * PoolingLayer::getActivationOutput()
{
    return output;
}

double* PoolingLayer::getDelta()
{
    return delta;
}

void PoolingLayer::clearGradient()
{
    //没有梯度，什么都不用做
}

void PoolingLayer::accumulateGradient(const double *prevActivation)
{
    //没有梯度，什么都不用做
}

void PoolingLayer::updateParameters(size_t batchSize, size_t trainSetSize)
{
    //没有参数，什么都不用做
}

void PoolingLayer::computeOutputDelta(const double *y)
{
    //池化层不能作为输出层，什么都不做
}

PoolingLayer::~PoolingLayer()
{
    delete[] output;
    delete[] xOffset;
    delete[] yOffset;
    delete[] delta;
}