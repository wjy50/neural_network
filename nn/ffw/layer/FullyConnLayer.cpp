/**
 * Created by wjy50 on 18-4-24.
 */

#include <random>
#include <cstring>
#include <cassert>
#include <iostream>

#include "FullyConnLayer.h"
#include "../../../math/Matrix.h"
#include "../../../utils/UniquePointerExt.h"
#include "../../../math/permutation.h"

using namespace ffw;
using namespace std;

FullyConnLayer::FullyConnLayer(int neuronCount, int inputDim, Activator activator) : AbsLayer(neuronCount, inputDim)
{
    this->activator = activator;
    if (activator != OUTPUT_ACTIVATOR) {
        activation = ACTIVATION_FUNCTIONS[activator];
        dActivation_dx = D_ACTIVATION_FUNCTIONS[activator];
    } else {
        activation = nullptr;
        dActivation_dx = nullptr;
    }

    weightCount = neuronCount * inputDim;
    weights = new FloatType[weightCount];
    weightGradient = new FloatType[weightCount];

    /*bias全部置0*/
    biases = new FloatType[neuronCount]();
    biasGradient = new FloatType[neuronCount];

    z = nullptr;
    a = nullptr;

    delta = nullptr;

    dropoutCount = 0;
    neuronIds = nullptr;
}

void FullyConnLayer::initialize(int miniBatchSize)
{
    random_device rd;
    normal_distribution<FloatType> distribution(0, sqrt(static_cast<FloatType>(2)/inputDim));
    /*随机生成weight*/
    for (int i = 0; i < neuronCount * inputDim; ++i) {
        weights[i] = distribution(rd);
    }

    this->miniBatchSize = miniBatchSize;
    z = new FloatType[neuronCount * miniBatchSize];
    a = new FloatType[neuronCount * miniBatchSize];

    delta = new FloatType[neuronCount * miniBatchSize];
}

void FullyConnLayer::feedForward(const FloatType *x)
{
    //此时不能有dropout
    multiplyMVTo(a, weights, x, neuronCount, inputDim);

    if (activator == OUTPUT_ACTIVATOR) {
        if (neuronCount > 1) {
            for (int i = 0; i < neuronCount; ++i) {
                a[i] += biases[i];
            }
            softMax(a, neuronCount);
        }
        else {
            a[0] = sigmoid(a[0] + biases[0]);
        }
    } else {
        for (int i = 0; i < neuronCount; ++i) {
            a[i] = activation(a[i] + biases[i]);
        }
    }
}

void FullyConnLayer::feedForwardForOptimization(const FloatType *x)
{
    if (dropoutCount == 0) {
        FloatType *curZ = z;
        const FloatType *curX = x;
        for (int m = 0; m < miniBatchSize; ++m) {
            multiplyMVTo(curZ, weights, curX, neuronCount, inputDim);
            curX += inputDim;
            curZ += neuronCount;
        }

        if (activator == OUTPUT_ACTIVATOR) {
            if (neuronCount > 1) {
                FloatType *curA = a;
                curZ = z;
                for (int m = 0; m < miniBatchSize; ++m) {
                    for (int i = 0; i < neuronCount; ++i) {
                        curZ[i] += biases[i];
                    }
                    softMaxInto(curA, curZ, neuronCount);
                    curA += neuronCount;
                    curZ += neuronCount;
                }
            }
            else {
                for (int i = 0; i < miniBatchSize; ++i) {
                    a[i] = sigmoid(z[i] += biases[0]);
                }
            }
        } else {
            FloatType *curA = a;
            curZ = z;
            for (int m = 0; m < miniBatchSize; ++m) {
                for (int i = 0; i < neuronCount; ++i) {
                    curA[i] = activation(curZ[i] += biases[i]);
                }
                curA += neuronCount;
                curZ += neuronCount;
            }
        }
    } else {
        if (!neuronIds) neuronIds = new int[neuronCount];
        randomPermutation<int>(neuronIds, neuronCount);

        int begin = 0;
        FloatType *curZ = z;
        const FloatType *curX = x;
        for (int m = 0; m < miniBatchSize; ++m) {
            for (int i = 0; i < neuronCount; ++i) {
                if (neuronIds[i] < dropoutCount) {
                    if (i > begin) {
                        multiplyMVTo(curZ + begin, weights + begin * inputDim, curX, i - begin, inputDim);
                    }
                    begin = i + 1;
                }
            }
            if (begin + 1 < neuronCount) {
                multiplyMVTo(curZ + begin, weights + begin * inputDim, curX, neuronCount - 1 - begin, inputDim);
            }
            curZ += neuronCount;
            curX += inputDim;
        }

        for (int i = 0; i < neuronCount * miniBatchSize; ++i) {
            a[i] = neuronIds[i % neuronCount] < dropoutCount ? 0 : activation(z[i] += biases[i % neuronCount]);
        }
    }
}

void FullyConnLayer::computeOutputDelta(const FloatType *y)
{
    if (activator == OUTPUT_ACTIVATOR) {
        subMMTo(delta, a, y, neuronCount * miniBatchSize, 1);
    }
}

void FullyConnLayer::computeBackPropDelta(FloatType *backPropDelta)
{
    FloatType *curBPD = backPropDelta;
    const FloatType *curDelta = delta;
    for (int m = 0; m < miniBatchSize; ++m) {
        multiplyTransposedMVTo(curBPD, weights, curDelta, neuronCount, inputDim);
        curBPD += inputDim;
        curDelta += neuronCount;
    }
}

void FullyConnLayer::backPropagateDelta()
{
    if (dropoutCount == 0) {
        for (int i = 0; i < neuronCount * miniBatchSize; ++i) {
            delta[i] *= dActivation_dx(z[i]);
        }
    } else {
        for (int i = 0; i < neuronCount * miniBatchSize; ++i) {
            if (neuronIds[i % neuronCount] < dropoutCount) delta[i] = 0;
            else delta[i] *= dActivation_dx(z[i]);
        }
    }
}

void FullyConnLayer::computeGradient(const FloatType *prevActivation)
{
    memset(weightGradient, 0, neuronCount * inputDim * sizeof(FloatType));
    memset(biasGradient, 0, neuronCount * sizeof(FloatType));
    const FloatType *curDelta = delta;
    for (int m = 0; m < miniBatchSize; ++m) {
        addMMTo(biasGradient, biasGradient, curDelta, neuronCount, 1);
        curDelta += neuronCount;
    }
    for (int i = 0; i < neuronCount; ++i) {
        biasGradient[i] /= miniBatchSize;
    }

    curDelta = delta;
    const FloatType *curPA = prevActivation;
    for (int m = 0; m < miniBatchSize; ++m) {
        for (int i = 0; i < neuronCount; ++i) {
            const FloatType di = curDelta[i];
            if (di != 0) {
                FloatType *wi = weightGradient + i * inputDim;
                for (int j = 0; j < inputDim; ++j) {
                    wi[j] += curPA[j] * di;
                }
            }
        }
        curDelta += neuronCount;
        curPA += inputDim;
    }
    for (int i = 0; i < weightCount; ++i) {
        weightGradient[i] /= miniBatchSize;
    }
}

void FullyConnLayer::updateParameters()
{
    optimizer->update(weights, weightGradient, biases, biasGradient);
}

int FullyConnLayer::getWeightCount()
{
    return weightCount;
}

int FullyConnLayer::getBiasCount()
{
    return neuronCount;
}

const FloatType * FullyConnLayer::getWeightedOutput()
{
    return z;
}

const FloatType * FullyConnLayer::getActivationOutput()
{
    return a;
}

FloatType* FullyConnLayer::getDelta()
{
    return delta;
}

void FullyConnLayer::setDropoutFraction(FloatType dropoutFraction)
{
    /*输出层不能dropout*/
    dropoutCount = activator == OUTPUT_ACTIVATOR ? 0 : static_cast<int>(neuronCount * dropoutFraction);
}

FullyConnLayer::~FullyConnLayer()
{
    delete[] biases;
    delete[] weights;
    delete[] weightGradient;
    delete[] biasGradient;
    delete[] z;
    delete[] a;
    delete[] delta;
    delete[] neuronIds;
}