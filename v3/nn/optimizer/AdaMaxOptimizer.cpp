/**
 * Created by wjy50 on 18-5-18.
 */

#include "AdaMaxOptimizer.h"
#include "../../interface/interface.h"

AdaMaxOptimizer::AdaMaxOptimizer(FloatType learningRate, FloatType beta1, FloatType beta2)
{
    this->learningRate = learningRate;
    this->beta1 = beta1;
    this->beta2 = beta2;
    oneMBeta1 = 1 - beta1;
    beta1T = beta1;

    m = nullptr;
    u = nullptr;
}

void AdaMaxOptimizer::update()
{
    adamFirstMomentEstimate(m, beta1, oneMBeta1, gradients, paramCount);
    adaMaxEWIN(u, beta2, gradients, paramCount);
    adaMaxUpdate(params, m, u, paramCount, learningRate, beta1T - 1);
    beta1T *= beta1;
}

void AdaMaxOptimizer::onAttachedToLayer()
{
    m = allocArray<FloatType>(paramCount);
    clearArray<FloatType>(m, paramCount);
    u = allocArray<FloatType>(paramCount);
    clearArray<FloatType>(u, paramCount);
}

AdaMaxOptimizer::~AdaMaxOptimizer()
{
    freeArray(m);
    freeArray(u);
}