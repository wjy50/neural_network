/**
 * Created by wjy50 on 18-6-1.
 */

#include <cmath>
#include <algorithm>
#include <iostream>
#include "BatchNormLayer.h"
#include "../../interface/interface.h"

BatchNormLayer::BatchNormLayer(int dim) : LayerBase(dim, dim)
{
    delta = nullptr;
    normOut = nullptr;
    xSubAvg = nullptr;
    dC_dNormOut = nullptr;
    midComp = nullptr;
    allocParamsAndGradients(2 * dim);
    gamma = params;
    std::fill_n(gamma, dim, 1);
    beta = params + dim;
    gammaGradient = gradients;
    betaGradient = gradients + dim;
    avg = allocArray<FloatType>(dim);
    var = allocArray<FloatType>(dim);
    oneDivDev = allocArray<FloatType>(dim);
    dC_dVar = allocArray<FloatType>(dim);
    dC_dAvg = allocArray<FloatType>(dim);
    avgSum = allocArray<FloatType>(dim);
    varSum = allocArray<FloatType>(dim);
    globalAvg = allocArray<FloatType>(dim);
    globalOneDivDev = allocArray<FloatType>(dim);

    miniBatchCount = 0;
}

const FloatType* BatchNormLayer::feedForward(const FloatType *x)
{
    batchNormalize(normOut, x, globalAvg, globalOneDivDev, outputDim, 1);
    bnTransform(output, normOut, gamma, beta, outputDim, 1);
    /*for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            std::cout << gamma[i * 28 + j] << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;*/
    return output;
}

const FloatType* BatchNormLayer::feedForwardForOptimization(const FloatType *x)
{
    averageVTo(avg, x, outputDim, miniBatchSize);
    varianceVTo(var, x, avg, outputDim, miniBatchSize);
    bnOneDivDev(oneDivDev, var, outputDim);
    batchNormalize(normOut, x, avg, oneDivDev, outputDim, miniBatchSize);
    bnTransform(output, normOut, gamma, beta, outputDim, miniBatchSize);
    bnXSubAvg(xSubAvg, x, avg, outputDim, miniBatchSize);
    addVTo(avgSum, avgSum, avg, outputDim);
    addVTo(varSum, varSum, var, outputDim);
    bnGlobalValues(globalAvg, globalOneDivDev, avgSum, varSum, outputDim, miniBatchSize, ++miniBatchCount);
    /*for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            std::cout << (static_cast<int>(output[i * 28 + j] * 10) > 0 ? 'O' : ' ') << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;*/
    return output;
}

void BatchNormLayer::backPropagate(const FloatType *y)
{
    bnDC_dNormOut(dC_dNormOut, delta, gamma, outputDim, miniBatchSize);
    bnDC_dVar(dC_dVar, oneDivDev, dC_dNormOut, xSubAvg, outputDim, miniBatchSize);
    bnMidComp(midComp, dC_dNormOut, oneDivDev, dC_dVar, xSubAvg, outputDim, miniBatchSize);
    bnDC_dAvg(dC_dAvg, midComp, outputDim, miniBatchSize);
    bnBackProp(deltaOutput, midComp, dC_dAvg, outputDim, miniBatchSize);
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
    dC_dNormOut = allocArray<FloatType>(outputDim * miniBatchSize);
    midComp = allocArray<FloatType>(outputDim * miniBatchSize);
}

BatchNormLayer::~BatchNormLayer()
{
    freeArray(delta);
    freeArray(normOut);
    freeArray(avg);
    freeArray(var);
    freeArray(oneDivDev);
    freeArray(dC_dNormOut);
    freeArray(dC_dVar);
    freeArray(dC_dAvg);
    freeArray(xSubAvg);
    freeArray(midComp);
    freeArray(avgSum);
    freeArray(varSum);
    freeArray(globalAvg);
    freeArray(globalOneDivDev);
}