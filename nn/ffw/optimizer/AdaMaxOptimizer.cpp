/**
 * Created by wjy50 on 18-5-6.
 */

#include <algorithm>
#include <cstring>
#include "AdaMaxOptimizer.h"

AdaMaxOptimizer::AdaMaxOptimizer(int weightCount, int biasCount, FloatType learningRate, FloatType beta1,
                                 FloatType beta2) : AbsOptimizer(weightCount, biasCount)
{
    int count = weightCount + biasCount;
    m = new FloatType[count]();
    u = new FloatType[count]();

    this->learningRate = learningRate;
    this->beta1 = beta1;
    this->beta2 = beta2;
    oneMBeta1 = 1 - beta1;
    beta1T = beta1;
}

void AdaMaxOptimizer::update(FloatType *weights, const FloatType *weightGradients, FloatType *biases,
                             const FloatType *biasGradients)
{
    for (int i = 0; i < weightSize; ++i) {
        m[i] = beta1 * m[i] + oneMBeta1 * weightGradients[i];
    }
    FloatType *cm = m + weightSize;
    for (int i = 0; i < biasSize; ++i) {
        cm[i] = beta1 * cm[i] + oneMBeta1 * biasGradients[i];
    }
    for (int i = 0; i < weightSize; ++i) {
        u[i] = std::max(beta2 * u[i], std::fabs(weightGradients[i]));
    }
    FloatType *cu = u + weightSize;
    for (int i = 0; i < biasSize; ++i) {
        cu[i] = std::max(beta2 * cu[i], std::fabs(biasGradients[i]));
    }
    FloatType beta1TMOne = beta1T - 1;
    for (int i = 0; i < weightSize; ++i) {
        weights[i] += learningRate * m[i] / (beta1TMOne * u[i] + 1e-6);
    }
    for (int i = 0; i < biasSize; ++i) {
        biases[i] += learningRate * cm[i] / (beta1TMOne * cu[i] + 1e-6);
    }
    beta1T *= beta1;
}

AdaMaxOptimizer::~AdaMaxOptimizer()
{
    delete[] m;
    delete[] u;
}