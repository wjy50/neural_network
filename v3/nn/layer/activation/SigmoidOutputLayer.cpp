/**
 * Created by wjy50 on 18-5-18.
 */

#include <cassert>
#include "SigmoidOutputLayer.h"
#include "../../../interface/interface.h"

SigmoidOutputLayer::SigmoidOutputLayer(int dim) : LayerBase(dim, dim) {}

const FloatType * SigmoidOutputLayer::feedForward(const FloatType *x)
{
    sigmoidOutput(output, x, outputDim);
    return output;
}

const FloatType *SigmoidOutputLayer::feedForwardForOptimization(const FloatType *x)
{
    sigmoidOutput(output, x, outputDim * miniBatchSize);
    return output;
}

void SigmoidOutputLayer::backPropagate(const FloatType *y)
{
    subtractVTo(deltaOutput, output, y, outputDim * miniBatchSize);
}

FloatType* SigmoidOutputLayer::getDelta()
{
    //assert("You cannot get delta from a sigmoid output layer" == nullptr);
}

void SigmoidOutputLayer::computeGradients() {}