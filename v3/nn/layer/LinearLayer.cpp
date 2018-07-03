/**
 * Created by wjy50 on 18-5-18.
 */

#include "LinearLayer.h"
#include "../../interface/interface.h"

LinearLayer::LinearLayer(int inputDim, int neuronCount, bool noBias) : LayerBase(inputDim, neuronCount)
{
    int weightCount = inputDim * neuronCount;
    if (noBias) {
        allocParamsAndGradients(weightCount);/*biasCount == 0*/
        biases = nullptr;
        biasGradients = nullptr;
    } else {
        allocParamsAndGradients(weightCount + neuronCount);/*biasCount == neuronCount*/
        biases = params + weightCount;
        biasGradients = gradients + weightCount;
    }
    weights = params;
    weightGradients = gradients;

    delta = nullptr;

    in = nullptr;
}

const FloatType* LinearLayer::feedForward(const FloatType *x, int count)
{
    multiplyMTmTo(output, x, weights, count, inputDim, outputDim);
    if (biases) linearLayerBias(output, outputDim, biases, count);
    return output;
}

const FloatType* LinearLayer::feedForwardForOptimization(const FloatType *x)
{
    in = x;
    /*T(weights * T(x)) == x * T(weights)*/
    multiplyMTmTo(output, x, weights, miniBatchSize, inputDim, outputDim);
    if (biases) linearLayerBias(output, outputDim, biases, miniBatchSize);
    return output;
}

void LinearLayer::backPropagate(const FloatType *y)
{
    /*T(T(weights) * T(y)) == y * weights*/
    multiplyMMTo(deltaOutput, y, weights, miniBatchSize, outputDim, inputDim);
}

void LinearLayer::computeGradients()
{
    multiplyTmMTo(weightGradients, delta, in, miniBatchSize, outputDim, inputDim);
    scaleV(weightGradients, static_cast<FloatType>(1) / miniBatchSize, inputDim * outputDim);
    if (biases) averageVTo(biasGradients, delta, outputDim, miniBatchSize);
}

void LinearLayer::onInitialized()
{
    delta = allocArray<FloatType>(outputDim * miniBatchSize);

    initLinearWeights(weights, outputDim, inputDim);
}

FloatType* LinearLayer::getDelta()
{
    return delta;
}

LinearLayer::~LinearLayer()
{
    freeArray(delta);
}