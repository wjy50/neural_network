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

    weights = new double[neuronCount * inputDim];
    weightGradient = new double[neuronCount * inputDim];
    transposedWeights = nullptr;

    biases = new double[neuronCount];
    /*bias全部置0*/
    memset(biases, 0, sizeof(double) * neuronCount);
    biasGradient = new double[neuronCount];

    z = new double[neuronCount];
    a = new double[neuronCount];

    delta = new double[neuronCount];

    regParam = 0;
    dropoutCount = 0;
    neuronIds = nullptr;
}

void FullyConnLayer::setRegParam(double regParam)
{
    this->regParam = regParam;
}

void FullyConnLayer::initialize()
{
    random_device rd;
    normal_distribution<double> distribution(0, sqrt((double)2/inputDim));
    /*随机生成weight*/
    for (int i = 0; i < neuronCount * inputDim; ++i) {
        weights[i] = distribution(rd);
    }
}

void FullyConnLayer::feedForward(const double *x)
{
    if (dropoutCount == 0) {
        multiplyMVTo(z, weights, x, neuronCount, inputDim);

        if (activator == OUTPUT_ACTIVATOR) {
            if (neuronCount > 1) softMaxInto(a, z, neuronCount);
            else {
                for (int i = 0; i < neuronCount; ++i) {
                    a[i] = sigmoid(z[i]);
                }
            }
        } else {
            for (int i = 0; i < neuronCount; ++i) {
                a[i] = activation(z[i]);
            }
        }
    } else {
        if (!neuronIds) neuronIds = new int[neuronCount];
        randomPermutation<int>(neuronIds, neuronCount);
        /*for (int i = 0; i < neuronCount; ++i) {
            cout << neuronIds[i] << ' ';
        }
        cout << endl;*/

        int begin = 0;
        for (int i = 0; i < neuronCount; ++i) {
            if (neuronIds[i] < dropoutCount) {
                z[i] = 0;
                if (i > begin) {
                    multiplyMVTo(z + begin, weights + begin * inputDim, x, i - begin, inputDim);
                }
                begin = i + 1;
            }
        }
        if (begin + 1 < neuronCount) {
            multiplyMVTo(z + begin, weights + begin * inputDim, x, neuronCount - 1 - begin, inputDim);
        }
        for (int i = 0; i < neuronCount; ++i) {
            a[i] = neuronIds[i] < dropoutCount ? 0 : activation(z[i]);
        }
    }
}

void FullyConnLayer::computeOutputDelta(const double *y)
{
    if (activator == OUTPUT_ACTIVATOR) {
        subMMTo(delta, a, y, neuronCount, 1);
    }
}

void FullyConnLayer::computeBackPropDelta(double *backPropDelta)
{
    if (!transposedWeights) transposedWeights = new double[neuronCount * inputDim];
    transposeMTo(transposedWeights, weights, neuronCount, inputDim);
    multiplyMVTo(backPropDelta, transposedWeights, delta, inputDim, neuronCount);
}

void FullyConnLayer::backPropagateDelta()
{
    if (dropoutCount == 0) {
        for (int i = 0; i < neuronCount; ++i) {
            delta[i] *= dActivation_dx(z[i]);
        }
    } else {
        for (int i = 0; i < neuronCount; ++i) {
            if (neuronIds[i] < dropoutCount) delta[i] = 0;
            else delta[i] *= dActivation_dx(z[i]);
        }
    }
}

void FullyConnLayer::clearGradient()
{
    memset(weightGradient, 0, neuronCount * inputDim * sizeof(double));
    memset(biasGradient, 0, neuronCount * sizeof(double));
}

void FullyConnLayer::accumulateGradient(const double *prevActivation)
{
    addMMTo(biasGradient, biasGradient, delta, neuronCount, 1);
    for (int i = 0; i < neuronCount; ++i) {
        if (delta[i] != 0) {
            double *wi = weightGradient + i * inputDim;
            for (int j = 0; j < inputDim; ++j) {
                wi[j] += prevActivation[j] * delta[i];
            }
        }
    }
}

void FullyConnLayer::updateParameters(size_t batchSize, size_t trainSetSize)
{
    double eta = -learningRate;
    for (int i = 0; i < neuronCount; ++i) {
        biases[i] += eta * biasGradient[i] / batchSize;
    }
    double ws = (1 + eta * regParam / trainSetSize);
    for (int i = 0; i < neuronCount * inputDim; ++i) {
        weights[i] = weights[i] * ws + eta * weightGradient[i] / batchSize;
    }
}

const double * FullyConnLayer::getWeightedOutput()
{
    return z;
}

const double * FullyConnLayer::getActivationOutput()
{
    return a;
}

double* FullyConnLayer::getDelta()
{
    return delta;
}

void FullyConnLayer::setDropoutFraction(double dropoutFraction)
{
    /*输出层不能dropout*/
    dropoutCount = activator == OUTPUT_ACTIVATOR ? 0 : static_cast<int>(neuronCount * dropoutFraction);
}

FullyConnLayer::~FullyConnLayer()
{
    delete[] biases;
    delete[] weights;
    delete[] transposedWeights;
    delete[] weightGradient;
    delete[] biasGradient;
    delete[] z;
    delete[] a;
    delete[] delta;
    delete[] neuronIds;
}