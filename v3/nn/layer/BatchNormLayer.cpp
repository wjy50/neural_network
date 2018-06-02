/**
 * Created by wjy50 on 18-6-1.
 */

#include <cmath>
#include <algorithm>
#include "BatchNormLayer.h"
#include "../../interface/interface.h"

BatchNormLayer::BatchNormLayer(int dim) : LayerBase(dim, dim)
{
    delta = nullptr;
    normOut = nullptr;
    xSubAvg = nullptr;
    deltaAvg = nullptr;
    normDelta = nullptr;
    allocParamsAndGradients(2 * dim);
    gamma = params;
    std::fill_n(gamma, dim, 1);
    beta = params + dim;
    gammaGradient = gradients;
    betaGradient = gradients + dim;
    avg = allocArray<FloatType>(dim);
    var = allocArray<FloatType>(dim);
    oneDivDev = allocArray<FloatType>(dim);
    deltaMulCenter = allocArray<FloatType>(dim);
    globalAvg = allocArray<FloatType>(dim);
    globalVar = allocArray<FloatType>(dim);
    globalOneDivDev = allocArray<FloatType>(dim);
    clearArray<FloatType>(globalAvg, dim);
    clearArray<FloatType>(globalVar, dim);
    clearArray<FloatType>(globalOneDivDev, dim);

    miniBatchCount = 0;
}

const FloatType* BatchNormLayer::feedForward(const FloatType *x)
{
    batchNormalize(normOut, x, globalAvg, globalOneDivDev, outputDim, 1);
    bnTransform(output, normOut, gamma, beta, outputDim, 1);
    return output;
}

const FloatType* BatchNormLayer::feedForwardForOptimization(const FloatType *x)
{
    averageVTo(avg, x, outputDim, miniBatchSize);
    bnXSubAvg(xSubAvg, x, avg, outputDim, miniBatchSize);
    bnVariance(var, xSubAvg, outputDim, miniBatchSize);
    bnOneDivDev(oneDivDev, var, outputDim);
    batchNormalize(normOut, x, avg, oneDivDev, outputDim, miniBatchSize);
    bnTransform(output, normOut, gamma, beta, outputDim, miniBatchSize);
    bnGlobalValues(globalAvg, globalVar, globalOneDivDev, avg, var, outputDim, miniBatchSize, ++miniBatchCount);
    return output;
}

void BatchNormLayer::backPropagate(const FloatType *y)
{
    averageVTo(deltaAvg, delta, outputDim, miniBatchSize);
    batchNormalize(normDelta, delta, deltaAvg, oneDivDev, outputDim, miniBatchSize);
    bnDeltaMulCenter(deltaMulCenter, delta, xSubAvg, outputDim, miniBatchSize);
    bnBackProp(deltaOutput, gamma, normDelta, normOut, var, deltaMulCenter, outputDim, miniBatchSize);
}

void BatchNormLayer::computeGradients()
{
    bnGradients(gammaGradient, betaGradient, delta, normOut, outputDim, miniBatchSize);
}

FloatType* BatchNormLayer::getDelta()
{
    return delta;
}

void BatchNormLayer::onInitialized()
{
    delta = allocArray<FloatType>(outputDim * miniBatchSize);
    normOut = allocArray<FloatType>(outputDim * miniBatchSize);
    xSubAvg = allocArray<FloatType>(outputDim * miniBatchSize);
    deltaAvg = allocArray<FloatType>(outputDim * miniBatchSize);
    normDelta = allocArray<FloatType>(outputDim * miniBatchSize);
}

BatchNormLayer::~BatchNormLayer()
{
    freeArray(delta);
    freeArray(normOut);
    freeArray(avg);
    freeArray(var);
    freeArray(oneDivDev);
    freeArray(xSubAvg);
    freeArray(deltaAvg);
    freeArray(globalVar);
    freeArray(globalAvg);
    freeArray(globalOneDivDev);
    freeArray(deltaMulCenter);
}