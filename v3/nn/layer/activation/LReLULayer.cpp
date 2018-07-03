/**
 * Created by wjy50 on 18-5-18.
 */

#include <cassert>
#include "LReLULayer.h"
#include "../../../interface/interface.h"

LReLULayer::LReLULayer(int dim, FloatType l) : LayerBase(dim, dim)
{
    assert(l > 0);
    this->l = l;
}

const FloatType * LReLULayer::feedForward(const FloatType *x, int count)
{
    leakyReLU(output, x, outputDim * count, l);
    return output;
}

const FloatType *LReLULayer::feedForwardForOptimization(const FloatType *x)
{
    in = x;
    leakyReLU(output, x, outputDim * miniBatchSize, l);
    return output;
}

void LReLULayer::backPropagate(const FloatType *y)
{
    leakyReLU_bp(deltaOutput, in, y, inputDim * miniBatchSize, l);
}

FloatType* LReLULayer::getDelta()
{
    return deltaOutput;
}

void LReLULayer::computeGradients() {}
