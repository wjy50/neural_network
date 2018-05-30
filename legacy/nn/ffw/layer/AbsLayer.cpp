/**
 * Created by wjy50 on 18-4-24.
 */

#include "AbsLayer.h"

using namespace ffw;

AbsLayer::AbsLayer(int neuronCount, int inputDim)
{
    optimizer = nullptr;
    this->neuronCount = neuronCount;
    this->inputDim = inputDim;
}

int AbsLayer::getNeuronCount()
{
    return neuronCount;
}

void AbsLayer::setOptimizer(AbsOptimizer *optimizer)
{
    this->optimizer = optimizer;
}

int AbsLayer::getInputDimension()
{
    return inputDim;
}