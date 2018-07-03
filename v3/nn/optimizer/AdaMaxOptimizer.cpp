/**
 * Created by wjy50 on 18-5-18.
 */

#include "AdaMaxOptimizer.h"
#include "../../interface/interface.h"
#include "../../utils/debug.h"

AdaMaxOptimizer::AdaMaxOptimizer(FloatType learningRate, FloatType beta1, FloatType beta2, FloatType weightDecay, int decayRange)
{
    this->learningRate = learningRate;
    this->beta1 = beta1;
    this->beta2 = beta2;
    oneMBeta1 = 1 - beta1;
    beta1T = beta1;

    m = nullptr;
    u = nullptr;

    this->weightDecay = weightDecay;
    this->decayRange = decayRange;
}

void AdaMaxOptimizer::update()
{
    adaMaxEstimate(m, u, beta1, oneMBeta1, beta2, gradients, paramCount);
    adaMaxUpdate(params, m, u, paramCount, learningRate, beta1T - 1, weightDecay, decayRange);
    beta1T *= beta1;
}

void AdaMaxOptimizer::onAttachedToLayer()
{
    m = allocArray<FloatType>(paramCount);
    clearArray<FloatType>(m, paramCount);
    u = allocArray<FloatType>(paramCount);
    clearArray<FloatType>(u, paramCount);
}

void AdaMaxOptimizer::setLearningRate(FloatType learningRate)
{
    this->learningRate = learningRate;
}

FloatType AdaMaxOptimizer::getLearningRate()
{
    return learningRate;
}

AdaMaxOptimizer::~AdaMaxOptimizer()
{
    freeArray(m);
    freeArray(u);
}