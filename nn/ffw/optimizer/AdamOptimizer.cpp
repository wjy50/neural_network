/**
 * Created by wjy50 on 18-5-5.
 */

#include <cstring>
#include <cmath>
#include "AdamOptimizer.h"

AdamOptimizer::AdamOptimizer(int weightSize, int biasSize, FloatType learningRate, FloatType beta1, FloatType beta2)
        : AbsOptimizer(weightSize, biasSize)
{
    this->beta1 = beta1;
    oneMBeta1 = 1 - beta1;
    this->beta2 = beta2;
    oneMBeta2 = 1 - beta2;
    int count = weightSize + biasSize;
    s = new FloatType[count]();
    r = new FloatType[count]();
    beta1T = beta1;
    beta2T = beta2;
    epsilon = -learningRate;
}

void AdamOptimizer::update(FloatType *weights, const FloatType *weightGradients, FloatType *biases,
                           const FloatType *biasGradients)
{
    for (int i = 0; i < weightSize; ++i) {
        s[i] = beta1 * s[i] + oneMBeta1 * weightGradients[i];
    }
    FloatType *cs = s + weightSize;
    for (int i = 0; i < biasSize; ++i) {
        cs[i] = beta1 * cs[i] + oneMBeta1 * biasGradients[i];
    }
    for (int i = 0; i < weightSize; ++i) {
        r[i] = beta2 * r[i] + oneMBeta2 * weightGradients[i] * weightGradients[i];
    }
    FloatType *cr = r + weightSize;
    for (int i = 0; i < biasSize; ++i) {
        cr[i] = beta2 * cr[i] + oneMBeta2 * biasGradients[i] * biasGradients[i];
    }
    FloatType oneMBeta1T = 1 - beta1T;
    FloatType oneMBeta2T = 1 - beta2T;
    for (int i = 0; i < weightSize; ++i) {
        /*FloatType sn = s[i] / (1 - beta1T);
        FloatType rn = r[i] / (1 - beta2T);
        weights[i] += epsilon * sn / (std::sqrt(rn) + 1e-8);*/
        weights[i] += epsilon * s[i] / (oneMBeta1T * std::sqrt(r[i] / oneMBeta2T) + 1e-6);
    }
    for (int i = 0; i < biasSize; ++i) {
        /*FloatType sn = cs[i] / (1 - beta1T);
        FloatType rn = cr[i] / (1 - beta2T);
        biases[i] += epsilon * sn / (std::sqrt(rn) + 1e-8);*/
        biases[i] += epsilon * cs[i] / (oneMBeta1T * std::sqrt(cr[i] / oneMBeta2T) + 1e-6);
    }
    beta1T *= beta1;
    beta2T *= beta2;
}

AdamOptimizer::~AdamOptimizer()
{
    delete[] s;
    delete[] r;
}