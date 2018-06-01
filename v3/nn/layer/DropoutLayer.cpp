/**
 * Created by wjy50 on 18-5-31.
 */

#include "DropoutLayer.h"
#include "../../utils/permutation.h"
#include "../../interface/interface.h"

DropoutLayer::DropoutLayer(int dim) : LayerBase(dim, dim)
{
    dropoutCount = 0;
    dropoutIds = new int[dim];
}

const FloatType* DropoutLayer::feedForward(const FloatType *x)
{
    return x;
}

const FloatType* DropoutLayer::feedForwardForOptimization(const FloatType *x)
{
    if (dropoutCount != 0) {
        randomPermutation<int>(dropoutIds, outputDim);
        linearDropout(const_cast<FloatType *>(x), outputDim, dropoutIds, dropoutCount, miniBatchSize);
    }
    return x;
}

void DropoutLayer::backPropagate(const FloatType *y)
{
    if (dropoutCount != 0) linearDropout(deltaOutput, outputDim, dropoutIds, dropoutCount, miniBatchSize);
}

void DropoutLayer::setDropoutFraction(FloatType f)
{
    dropoutCount = static_cast<int>(outputDim * f);
}

FloatType* DropoutLayer::getDelta()
{
    return deltaOutput;
}

bool DropoutLayer::needIndependentOutput()
{
    return false;
}

void DropoutLayer::computeGradients() {}

DropoutLayer::~DropoutLayer()
{
    delete[] dropoutIds;
}