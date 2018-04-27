/**
 * Created by wjy50 on 18-4-24.
 */

#include <random>
#include <cstring>

#include "FullyConnLayer.h"
#include "../../../math/Matrix.h"
#include "../../../utils/UniquePointerExt.h"

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

    biases = new double[neuronCount];
    /*bias全部置0*/
    memset(biases, 0, sizeof(double) * neuronCount);
    biasGradient = new double[neuronCount];

    z = new double[neuronCount];
    a = new double[neuronCount];

    delta = new double[neuronCount];

    regParam = 0;
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
}

void FullyConnLayer::computeOutputDelta(const double *y)
{
    if (activator == OUTPUT_ACTIVATOR) {
        subMMTo(delta, a, y, neuronCount, 1);
    }
}

void FullyConnLayer::computeBackPropDelta(double *backPropDelta)
{
    unique_ptr<double[]> transposedWeights = make_unique_array<double[]>((size_t)neuronCount * (size_t)inputDim);
    transposeMTo(transposedWeights.get(), weights, neuronCount, inputDim);
    multiplyMVTo(backPropDelta, transposedWeights.get(), delta, inputDim, neuronCount);
}

void FullyConnLayer::backPropagateDelta()
{
    for (int i = 0; i < neuronCount; ++i) {
        delta[i] *= dActivation_dx(z[i]);
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
        double *wi = weightGradient + i * inputDim;
        for (int j = 0; j < inputDim; ++j) {
            wi[j] += prevActivation[j] * delta[i];
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

FullyConnLayer::~FullyConnLayer()
{
    delete[] biases;
    delete[] weights;
    delete[] weightGradient;
    delete[] biasGradient;
    delete[] z;
    delete[] a;
    delete[] delta;
}