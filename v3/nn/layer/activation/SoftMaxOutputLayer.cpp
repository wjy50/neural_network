/**
 * Created by wjy50 on 18-5-18.
 */

#include <cassert>
#include "SoftMaxOutputLayer.h"
#include "../../../interface/interface.h"

SoftMaxOutputLayer::SoftMaxOutputLayer(int dim) : LayerBase(dim, dim) {}

const FloatType * SoftMaxOutputLayer::feedForward(const FloatType *x)
{
    softMaxOutput(output, x, outputDim, 1);
    return output;
}

const FloatType *SoftMaxOutputLayer::feedForwardForOptimization(const FloatType *x)
{
    softMaxOutput(output, x, outputDim, miniBatchSize);
    return output;
}

void SoftMaxOutputLayer::backPropagate(const FloatType *y)
{
    subtractVTo(deltaOutput, output, y, outputDim * miniBatchSize);
}

FloatType* SoftMaxOutputLayer::getDelta()
{
    //assert("You cannot get delta from a soft max output layer" == nullptr);
}

void SoftMaxOutputLayer::computeGradients() {}