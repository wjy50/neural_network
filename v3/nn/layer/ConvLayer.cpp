/**
 * Created by wjy50 on 18-5-18.
 */

#include <cassert>
#include <random>
#include "ConvLayer.h"
#include "../../interface/interface.h"

ConvLayer::ConvLayer(
        int inputWidth, int inputHeight, int inputChannel,
        int kernelWidth, int kernelHeight, int kernelCount,
        int xStride, int yStride, int xPadding, int yPadding,
        bool noBias
) : LayerBase(
        inputWidth * inputHeight * inputChannel,
        ((inputWidth - kernelWidth + xPadding * 2) / xStride + 1) * ((inputHeight - kernelHeight + yPadding * 2) / yStride + 1) * kernelCount
)
{
    assert((inputWidth - kernelWidth + xPadding * 2) % xStride == 0 && (inputHeight - kernelHeight + yPadding * 2) % yStride == 0);

    int weightCount = kernelWidth * kernelHeight * inputChannel * kernelCount;
    if (noBias) {
        allocParamsAndGradients(weightCount);/*biasCount == 0*/
        biases = nullptr;
        biasGradients = nullptr;
    } else {
        allocParamsAndGradients(weightCount + kernelCount);/*biasCount == kernelCount*/
        biases = params + weightCount;
        biasGradients = gradients + weightCount;
    }
    kernels = params;
    kernelGradients = gradients;

    this->inputWidth = inputWidth;
    this->inputHeight = inputHeight;
    this->inputChannel = inputChannel;

    this->kernelWidth = kernelWidth;
    this->kernelHeight = kernelHeight;
    this->kernelCount = kernelCount;

    outputWidth = (inputWidth - kernelWidth + xPadding * 2) / xStride + 1;
    outputHeight = (inputHeight - kernelHeight + yPadding * 2) / yStride + 1;

    this->xStride = xStride;
    this->yStride = yStride;

    this->xPadding = xPadding;
    this->yPadding = yPadding;

    delta = nullptr;

    in = nullptr;
}

const FloatType* ConvLayer::feedForward(const FloatType *x, int count)
{
    conv2(x, inputWidth, inputHeight, inputChannel, output, outputWidth, outputHeight, kernels, kernelWidth, kernelHeight, kernelCount, xStride, yStride, xPadding, yPadding, count);
    if (biases) convLayerBias(output, outputHeight, outputWidth, kernelCount, biases, count);
    return output;
}

const FloatType* ConvLayer::feedForwardForOptimization(const FloatType *x)
{
    in = x;
    conv2(x, inputWidth, inputHeight, inputChannel, output, outputWidth, outputHeight, kernels, kernelWidth, kernelHeight, kernelCount, xStride, yStride, xPadding, yPadding, miniBatchSize);
    if (biases) convLayerBias(output, outputHeight, outputWidth, kernelCount, biases, miniBatchSize);
    return output;
}

void ConvLayer::backPropagate(const FloatType *y)
{
    convBP2(y, outputWidth, outputHeight, deltaOutput, inputWidth, inputHeight, inputChannel, kernels, kernelWidth, kernelHeight, kernelCount, xStride, yStride, xPadding, yPadding, miniBatchSize);
}

void ConvLayer::computeGradients()
{
    convGradients2(kernelGradients, kernelWidth, kernelHeight, biasGradients, kernelCount, delta, outputWidth, outputHeight, in, inputWidth, inputHeight, inputChannel, xStride, yStride, xPadding, yPadding, miniBatchSize);
}

void ConvLayer::onInitialized()
{
    delta = allocArray<FloatType>(outputDim * miniBatchSize);

    initConvKernel(kernels, kernelWidth * kernelHeight * inputChannel * kernelCount, inputDim);
}

FloatType* ConvLayer::getDelta()
{
    return delta;
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

int ConvLayer::getKernelParamCount()
{
    return kernelWidth * kernelHeight * inputChannel * kernelCount;
}

ConvLayer::~ConvLayer()
{
    freeArray(delta);
}