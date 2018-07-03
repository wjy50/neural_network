/**
 * Created by wjy50 on 18-6-1.
 */

#include <cmath>
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
    m_fill_n(gamma, dim, 1);
    beta = params + dim;
    gammaGradient = gradients;
    betaGradient = gradients + dim;
    avg = allocArray<FloatType>(dim);
    var = allocArray<FloatType>(dim);
    oneDivDev = allocArray<FloatType>(dim);
    deltaMulCenter = allocArray<FloatType>(dim);
    globalAvg = allocArray<FloatType>(dim);
    globalVar = allocArray<FloatType>(dim);
    clearArray<FloatType>(globalAvg, dim);
    clearArray<FloatType>(globalVar, dim);
    newBatch = true;
}

const FloatType* BatchNormLayer::feedForward(const FloatType *x, int count)
{
    if (newBatch) {
        bnGlobalOneDivDev(oneDivDev, globalVar, outputDim);
        newBatch = false;
    }
    bnForward(output, normOut, x, globalAvg, oneDivDev, gamma, beta, outputDim, count);
    return output;
}

const FloatType* BatchNormLayer::feedForwardForOptimization(const FloatType *x)
{
    bnAvg(avg, xSubAvg, x, outputDim, miniBatchSize);
    bnOneDivDev(var, oneDivDev, xSubAvg, outputDim, miniBatchSize);
    bnGlobalValues(globalAvg, globalVar, avg, var, outputDim);
    bnForward(output, normOut, x, avg, oneDivDev, gamma, beta, outputDim, miniBatchSize);
    newBatch = true;
    return output;
}

void BatchNormLayer::backPropagate(const FloatType *y) /*y == delta*/
{
    averageVTo(deltaAvg, y, outputDim, miniBatchSize);
    batchNormalize(normDelta, y, deltaAvg, oneDivDev, outputDim, miniBatchSize);
    bnDeltaMulCenter(deltaMulCenter, y, xSubAvg, outputDim, miniBatchSize);
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
    freeArray(deltaMulCenter);
}