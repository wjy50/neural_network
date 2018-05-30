/**
 * Created by wjy50 on 18-5-18.
 */

#include "SGDOptimizer.h"
#include "../../interface/interface.h"

SGDOptimizer::SGDOptimizer(FloatType learningRate, FloatType regParam, int regRange)
{
    setParams(learningRate, regParam, regRange);
}

void SGDOptimizer::setParams(FloatType learningRate, FloatType regParam, int regRange)
{
    eta = -learningRate;
    reg = 1 + eta * regParam;
    this->regRange = regRange;
}

void SGDOptimizer::update()
{
    if (regRange == 0) {
        sgd(params, gradients, eta, paramCount);
    } else {
        l2SGD(params, gradients, eta, reg, regRange);
        sgd(params + regRange, gradients + regRange, eta, paramCount);
    }
}