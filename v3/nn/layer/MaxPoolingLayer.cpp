/**
 * Created by wjy50 on 18-5-18.
 */

#include <cassert>
#include "MaxPoolingLayer.h"
#include "../../interface/interface.h"

MaxPoolingLayer::MaxPoolingLayer(
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
}

const FloatType* MaxPoolingLayer::feedForward(const FloatType *x)
{
    maxPooling(x, inputWidth, inputHeight, output, outputWidth, outputHeight, nullptr, nullptr, windowWidth, windowHeight, xStride, yStride, inputChannel);
    return output;
}

const FloatType* MaxPoolingLayer::feedForwardForOptimization(const FloatType *x)
{
    maxPooling(x, inputWidth, inputHeight, output, outputWidth, outputHeight, xOffset, yOffset, windowWidth, windowHeight, xStride, yStride, inputChannel * miniBatchSize);
    return output;
}

void MaxPoolingLayer::backPropagate(const FloatType *y)
{
    maxPoolingBP(y, outputWidth, outputHeight, deltaOutput, inputWidth, inputHeight, xOffset, yOffset, windowWidth, windowHeight, xStride, yStride, inputChannel * miniBatchSize);
}

void MaxPoolingLayer::onInitialized()
{
    xOffset = allocArray<int>(outputDim * miniBatchSize);
    yOffset = allocArray<int>(outputDim * miniBatchSize);
}

FloatType* MaxPoolingLayer::getDelta()
{
    return deltaOutput;
}

void MaxPoolingLayer::computeGradients() {}

int MaxPoolingLayer::getOutputWidth()
{
    return outputWidth;
}

int MaxPoolingLayer::getOutputHeight()
{
    return outputHeight;
}

int MaxPoolingLayer::getChannelCount()
{
    return inputChannel;
}

MaxPoolingLayer::~MaxPoolingLayer()
{
    freeArray(xOffset);
    freeArray(yOffset);
}