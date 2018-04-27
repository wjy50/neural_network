/**
 * Created by wjy50 on 18-4-24.
 */

#include "AbsLayer.h"

using namespace ffw;

AbsLayer::AbsLayer(int neuronCount, int inputDim)
{
    learningRate = 0;
    this->neuronCount = neuronCount;
    this->inputDim = inputDim;
}

int AbsLayer::getNeuronCount()
{
    return neuronCount;
}

void AbsLayer::setLearningRate(double learningRate)
{
    this->learningRate = learningRate;
}

int AbsLayer::getInputDimension()
{
    return inputDim;
}