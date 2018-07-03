/**
 * Created by wjy50 on 18-5-18.
 */

#include "AdamOptimizer.h"
#include "../../interface/interface.h"

AdamOptimizer::AdamOptimizer(FloatType learningRate, FloatType beta1, FloatType beta2, FloatType weightDecay, int decayRange)
{
    this->beta1 = beta1;
    oneMBeta1 = 1 - beta1;
    this->beta2 = beta2;
    oneMBeta2 = 1 - beta2;
    beta1T = beta1;
    beta2T = beta2;
    alpha = -learningRate;

    m = nullptr;
    v = nullptr;

    this->weightDecay = weightDecay;
    this->decayRange = decayRange;
}

void AdamOptimizer::update()
{
    adamEstimate(m, v, beta1, oneMBeta1, beta2, oneMBeta2, gradients, paramCount);
    adamUpdate(params, m, v, paramCount, alpha, 1 - beta1T, 1 - beta2T, weightDecay, decayRange);
    beta1T *= beta1;
    beta2T *= beta2;
}

void AdamOptimizer::onAttachedToLayer()
{
    m = allocArray<FloatType>(paramCount);
    clearArray<FloatType>(m, paramCount);
    v = allocArray<FloatType>(paramCount);
    clearArray<FloatType>(v, paramCount);
}

void AdamOptimizer::setLearningRate(FloatType learningRate)
{
    this->alpha = -learningRate;
}

FloatType AdamOptimizer::getLearningRate()
{
    return -alpha;
}

AdamOptimizer::~AdamOptimizer()
{
    freeArray(m);
    freeArray(v);
}