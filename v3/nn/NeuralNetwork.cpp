/**
 * Created by wjy50 on 18-5-18.
 */

#include <cassert>
#include "NeuralNetwork.h"
#include "../interface/interface.h"
#include "../utils/UniquePointerExt.h"
#include "../utils/permutation.h"

NeuralNetwork::NeuralNetwork()
{
    built = false;
    miniBatchSize = 0;
    inputs = nullptr;
    labels = nullptr;
}

void NeuralNetwork::addLayer(LayerBase *layer)
{
    assert(!built);

    layers.push_back(layer);
}

void NeuralNetwork::buildUpNetwork(int miniBatchSize)
{
    assert(!built);
    built = true;

    this->miniBatchSize = miniBatchSize;
    inputs = allocArray<FloatType>(layers[0]->getInputDim() * miniBatchSize);
    labels = allocArray<FloatType>(layers.back()->getOutputDim() * miniBatchSize);
    FloatType *od = nullptr;
    for (LayerBase *layer : layers) {
        layer->initialize(miniBatchSize);
        if (od) layer->setDeltaOutput(od);
        od = layer->getDelta();
    }
}

const FloatType* NeuralNetwork::feedForward(const FloatType *x)
{
    const FloatType *a = x;
    for (LayerBase *layer : layers) {
        a = layer->feedForward(a);
    }
    return a;
}

void NeuralNetwork::optimize(DataSetBase &trainSet, int altTrainSetSize)
{
    int trainSetSize = altTrainSetSize > 0 ? altTrainSetSize : trainSet.getCount();
    std::unique_ptr<int[]> indices = make_unique_array<int[]>(static_cast<size_t>(trainSetSize));
    randomPermutation<int>(indices.get(), trainSetSize);
    int miniBatchCount = trainSetSize / miniBatchSize;
    for (int t = 0; t < miniBatchCount; ++t) {
        trainSet.getBatch(inputs, labels, indices.get() + miniBatchSize * t, miniBatchSize);

        const FloatType *in = inputs;
        for (LayerBase *layer : layers) {
            in = layer->feedForwardForOptimization(in);
        }

        auto layerCount = static_cast<int>(layers.size());
        in = labels;
        for (int i = layerCount - 1; i > 0; --i) {
            layers[i]->backPropagate(in);
            in = layers[i - 1]->getDelta();
        }

        for (LayerBase *layer : layers) {
            layer->updateParameters();
        }
    }
}

NeuralNetwork::~NeuralNetwork()
{
    freeArray(inputs);
    freeArray(labels);
}