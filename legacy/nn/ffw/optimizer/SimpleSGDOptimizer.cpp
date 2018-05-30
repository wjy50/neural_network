/**
 * Created by wjy50 on 18-5-5.
 */

#include "SimpleSGDOptimizer.h"

SimpleSGDOptimizer::SimpleSGDOptimizer(FloatType learningRate, FloatType regParam, int trainSetSize, int weightSize, int biasSize) : AbsOptimizer(weightSize, biasSize)
{
    eta = -learningRate;
    reg = 1 + eta * regParam / trainSetSize;
}

void SimpleSGDOptimizer::update(FloatType *weights, const FloatType *weightGradients, FloatType *biases,
                                const FloatType *biasGradients)
{
    for (int i = 0; i < biasSize; ++i) {
        biases[i] += eta * biasGradients[i];
    }
    for (int i = 0; i < weightSize; ++i) {
        weights[i] = weights[i] * reg + eta * weightGradients[i];
    }
}