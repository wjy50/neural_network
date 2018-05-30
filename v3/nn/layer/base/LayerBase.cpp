/**
 * Created by wjy50 on 18-5-18.
 */

#include <cassert>
#include <iostream>
#include "LayerBase.h"
#include "../../../interface/interface.h"

LayerBase::LayerBase(int inputDim, int outputDim)
{
    this->inputDim = inputDim;
    this->outputDim = outputDim;
    paramCount = 0;
    miniBatchSize = 0;
    optimizer = nullptr;
    params = nullptr;
    output = nullptr;
    deltaOutput = nullptr;
}

int LayerBase::getInputDim()
{
    return inputDim;
}

int LayerBase::getOutputDim()
{
    return outputDim;
}

void LayerBase::setOptimizer(OptimizerBase *optimizer)
{
    this->optimizer = optimizer;
    optimizer->attachToLayer(paramCount, params, gradients);
}

void LayerBase::updateParameters()
{
    if (optimizer) {
        computeGradients();
        optimizer->update();
    }
}

void LayerBase::initialize(int maxMiniBatchSize)
{
    this->miniBatchSize = maxMiniBatchSize;
    allocOutput();
    onInitialized();
}

void LayerBase::onInitialized() {}

void LayerBase::allocParamsAndGradients(int count)
{
    assert(!params);

    paramCount = count;
    params = allocArray<FloatType>(count);
    gradients = allocArray<FloatType>(count);
}

void LayerBase::allocOutput()
{
    assert(!output && miniBatchSize > 0);

    output = allocArray<FloatType>(outputDim * miniBatchSize);
}

void LayerBase::setDeltaOutput(FloatType *deltaOutput)
{
    this->deltaOutput = deltaOutput;
}

LayerBase::~LayerBase()
{
    freeArray(params);
    freeArray(gradients);
    freeArray(output);
}