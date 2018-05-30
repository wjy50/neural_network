/**
 * Created by wjy50 on 18-5-18.
 */

#include "ReLULayer.h"
#include "../../../interface/interface.h"

ReLULayer::ReLULayer(int dim) : LayerBase(dim, dim) {}

const FloatType * ReLULayer::feedForward(const FloatType *x)
{
    reLU(output, x, outputDim);
    return output;
}

const FloatType *ReLULayer::feedForwardForOptimization(const FloatType *x)
{
    in = x;
    reLU(output, x, outputDim * miniBatchSize);
    return output;
}

void ReLULayer::backPropagate(const FloatType *y)
{
    reLU_bp(deltaOutput, in, y, inputDim * miniBatchSize);
}

FloatType* ReLULayer::getDelta()
{
    return deltaOutput;
}

void ReLULayer::computeGradients() {}
