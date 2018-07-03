/**
 * Created by wjy50 on 18-6-15.
 */

#include <cassert>
#include "ResidualBlock.h"
#include "../../interface/interface.h"

ResidualBlock::ResidualBlock(int dim) : LayerBase(dim, dim)
{
    built = false;
}

const FloatType* ResidualBlock::feedForward(const FloatType *x, int count)
{
    const FloatType *a = x;
    for (LayerBase *layer : layers) {
        a = layer->feedForward(a, count);
    }
    alphaXPlusY(1, x, const_cast<FloatType *>(a), outputDim * count);
    return a;
}

const FloatType* ResidualBlock::feedForwardForOptimization(const FloatType *x)
{
    const FloatType *a = x;
    for (LayerBase *layer : layers) {
        a = layer->feedForwardForOptimization(a);
    }
    alphaXPlusY(1, x, const_cast<FloatType *>(a), outputDim * miniBatchSize);
    return a;
}

void ResidualBlock::backPropagate(const FloatType *y)
{
    auto layerCount = static_cast<int>(layers.size());
    const FloatType *o = y;
    for (int i = layerCount - 1; i > 0; --i) {
        layers[i]->backPropagate(o);
        o = layers[i - 1]->getDelta();
    }
    if (deltaOutput) {
        layers[0]->backPropagate(o);
        alphaXPlusY(1, y, deltaOutput, outputDim * miniBatchSize);
    }
}

void ResidualBlock::updateParameters()
{
    for (LayerBase *layer : layers) {
        layer->updateParameters();
    }
}

FloatType* ResidualBlock::getDelta()
{
    if (!built) {
        assert(layers[0]->getInputDim() == inputDim && layers.back()->getOutputDim() == outputDim);
        FloatType *od = deltaOutput;
        for (LayerBase *layer : layers) {
            layer->initialize(miniBatchSize);
            layer->setDeltaOutput(od);
            od = layer->getDelta();
        }
        built = true;
        return od;
    }
    return layers.back()->getDelta();
}

void ResidualBlock::addLayer(LayerBase *layer)
{
    assert(!built);
    layers.push_back(layer);
}

bool ResidualBlock::needBackPropAtFirst()
{
    return true;
}

void ResidualBlock::computeGradients() {}

bool ResidualBlock::needIndependentOutput()
{
    return false;
}