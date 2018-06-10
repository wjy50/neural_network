/**
 * Created by wjy50 on 18-5-18.
 */

#include <cassert>
#include <iostream>
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
    indices = nullptr;
    indexCap = 0;
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
    ensureIndexCap(trainSetSize);
    randomPermutation(indices, trainSetSize);
    int miniBatchCount = trainSetSize / miniBatchSize;
    for (int t = 0; t < miniBatchCount; ++t) {
        trainSet.getBatch(inputs, labels, indices + miniBatchSize * t, miniBatchSize);

        const FloatType *in = inputs;
        for (LayerBase *layer : layers) {
            //long st = clock();
            in = layer->feedForwardForOptimization(in);
            //std::cout << clock() - st << std::endl;
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
        //std::cout << std::endl;
    }
}

void NeuralNetwork::ensureIndexCap(int size)
{
    if (indexCap == 0) indexCap = 1;
    while (indexCap < size) indexCap <<= 1;
    freeArray(indices);
    indices = allocArray<int>(indexCap);
}

NeuralNetwork::~NeuralNetwork()
{
    freeArray(inputs);
    freeArray(labels);
    freeArray(indices);
}