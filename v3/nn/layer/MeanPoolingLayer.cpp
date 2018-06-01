/**
 * Created by wjy50 on 18-5-18.
 */

#include <cassert>
#include "MeanPoolingLayer.h"
#include "../../interface/interface.h"

MeanPoolingLayer::MeanPoolingLayer(
        int inputWidth, int inputHeight, int inputChannel,
        int windowWidth, int windowHeight,
        int xStride, int yStride
) : LayerBase(
        inputWidth * inputHeight * inputChannel,
        ((inputWidth - windowWidth) / (xStride > 0 ? xStride : windowWidth) + 1) * ((inputHeight - windowHeight) / (yStride > 0 ? yStride : windowHeight) + 1) * inputChannel
)
{
    this->xStride = xStride > 0 ? xStride : windowWidth;
    this->yStride = yStride > 0 ? yStride : windowHeight;

    assert((inputWidth - windowWidth) % this->xStride == 0 && (inputHeight - windowHeight) % this->yStride == 0);

    this->inputWidth = inputWidth;
    this->inputHeight = inputHeight;

    this->inputChannel = inputChannel;

    this->windowWidth = windowWidth;
    this->windowHeight = windowHeight;

    outputWidth = (inputWidth - windowWidth) / this->xStride + 1;
    outputHeight = (inputHeight - windowHeight) / this->yStride + 1;

    delta = nullptr;
}

const FloatType* MeanPoolingLayer::feedForward(const FloatType *x)
{
    meanPooling(x, inputWidth, inputHeight, output, outputWidth, outputHeight, windowWidth, windowHeight, xStride, yStride, inputChannel);
    return output;
}

const FloatType* MeanPoolingLayer::feedForwardForOptimization(const FloatType *x)
{
    meanPooling(x, inputWidth, inputHeight, output, outputWidth, outputHeight, windowWidth, windowHeight, xStride, yStride, inputChannel * miniBatchSize);
    return output;
}

void MeanPoolingLayer::backPropagate(const FloatType *y)
{
    meanPoolingBP(y, outputWidth, outputHeight, deltaOutput, inputWidth, inputHeight, windowWidth, windowHeight, xStride, yStride, inputChannel * miniBatchSize);
}

FloatType* MeanPoolingLayer::getDelta()
{
    return delta;
}

void MeanPoolingLayer::computeGradients() {}

int MeanPoolingLayer::getOutputWidth()
{
    return outputWidth;
}

int MeanPoolingLayer::getOutputHeight()
{
    return outputHeight;
}

int MeanPoolingLayer::getChannelCount()
{
    return inputChannel;
}

void MeanPoolingLayer::onInitialized()
{
    delta = allocArray<FloatType>(outputDim * miniBatchSize);
}

MeanPoolingLayer::~MeanPoolingLayer()
{
    freeArray(delta);
}