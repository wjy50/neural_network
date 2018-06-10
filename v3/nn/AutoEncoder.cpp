/**
 * Created by wjy50 on 18-6-2.
 */

#include <cassert>
#include "AutoEncoder.h"
#include "../interface/interface.h"
#include "../utils/permutation.h"
#include "../utils/UniquePointerExt.h"

AutoEncoder::AutoEncoder()
{
    miniBatchSize = 0;
    built = false;
    inputs = nullptr;
    indices = nullptr;
    indexCap = 0;
}

void AutoEncoder::addLayer(LayerBase *layer)
{
    assert(!built);

    layers.push_back(layer);
}

void AutoEncoder::buildUpAutoEncoder(int miniBatchSize)
{
    assert(!built);
    built = true;

    this->miniBatchSize = miniBatchSize;
    inputs = allocArray<FloatType>(layers[0]->getInputDim() * miniBatchSize);
    FloatType *od = nullptr;
    for (LayerBase *layer : layers) {
        layer->initialize(miniBatchSize);
        if (od) layer->setDeltaOutput(od);
        od = layer->getDelta();
    }
}

const FloatType* AutoEncoder::feedForward(const FloatType *x)
{
    const FloatType *a = x;
    for (LayerBase *layer : layers) {
        a = layer->feedForward(a);
    }
    return a;
}

void AutoEncoder::optimize(DataSetBase &trainSet, int altTrainSetSize)
{
    int trainSetSize = altTrainSetSize > 0 ? altTrainSetSize : trainSet.getCount();
    ensureIndexCap(trainSetSize);
    randomPermutation(indices, trainSetSize);
    int miniBatchCount = trainSetSize / miniBatchSize;
    for (int t = 0; t < miniBatchCount; ++t) {
        trainSet.getBatch(inputs, nullptr, indices + miniBatchSize * t, miniBatchSize);

        const FloatType *in = inputs;
        for (LayerBase *layer : layers) {
            in = layer->feedForwardForOptimization(in);
        }

        auto layerCount = static_cast<int>(layers.size());
        in = inputs;
        for (int i = layerCount - 1; i > 0; --i) {
            layers[i]->backPropagate(in);
            in = layers[i - 1]->getDelta();
        }

        for (LayerBase *layer : layers) {
            layer->updateParameters();
        }
    }
}

void AutoEncoder::ensureIndexCap(int size)
{
    if (indexCap == 0) indexCap = 1;
    while (indexCap < size) indexCap <<= 1;
    freeArray(indices);
    indices = allocArray<int>(indexCap);
}

AutoEncoder::~AutoEncoder()
{
    freeArray(inputs);
    freeArray(indices);
}