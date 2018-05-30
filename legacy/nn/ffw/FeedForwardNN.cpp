/**
 * Created by wjy50 on 18-4-24.
 */

#include <cassert>
#include <ctime>
#include <iostream>

#include "FeedForwardNN.h"
#include "../../utils/UniquePointerExt.h"
#include "../../v3/utils/permutation.h"
#include "../../data/DataSet.h"

using namespace ffw;

FeedForwardNN::FeedForwardNN()
{
    built = false;
    inputs = nullptr;
    labels = nullptr;
    miniBatchSize = 0;
}

void FeedForwardNN::addLayer(ffw::AbsLayer *layer)
{
    assert(!built);

    layers.push_back(layer);
}

void FeedForwardNN::buildUpNetwork(int miniBatchSize)
{
    assert(!built);
    built = true;

    this->miniBatchSize = miniBatchSize;
    inputs = new FloatType[layers[0]->getInputDimension() * miniBatchSize];
    labels = new FloatType[layers.back()->getNeuronCount() * miniBatchSize];
    for (AbsLayer *layer : layers) {
        layer->initialize(miniBatchSize);
    }
}

const FloatType *FeedForwardNN::feedForward(const FloatType *x)
{
    const FloatType *a = x;
    for (AbsLayer *layer : layers) {
        layer->feedForward(a);
        a = layer->getActivationOutput();
    }
    return a;
}

void FeedForwardNN::SGD(DataSet &trainSet, int altTrainSetSize)
{
    int trainSetSize = altTrainSetSize > 0 ? altTrainSetSize : trainSet.getSize();
    unique_ptr<int[]> indices = make_unique_array<int[]>(static_cast<size_t>(trainSetSize));
    randomPermutation<int>(indices.get(), trainSetSize);
    int miniBatchCount = trainSetSize / miniBatchSize;
    for (int t = 0; t < miniBatchCount; ++t) {
        trainSet.getBatch(inputs, labels, indices.get() + t * miniBatchSize, miniBatchSize);

        const FloatType *in = inputs;
        for (AbsLayer *layer : layers) {
            layer->feedForwardForOptimization(in);
            in = layer->getActivationOutput();
        }

        const auto layerCount = static_cast<int>(layers.size());
        layers[layerCount - 1]->computeOutputDelta(labels);
        for (int j = layerCount - 2; j >= 0; --j) {
            layers[j + 1]->computeBackPropDelta(layers[j]->getDelta());
            layers[j]->backPropagateDelta();
        }

        in = inputs;
        for (AbsLayer *layer : layers) {
            layer->computeGradient(in);
            layer->updateParameters();
            in = layer->getActivationOutput();
        }
    }
}

FeedForwardNN::~FeedForwardNN()
{
    delete[] inputs;
    delete[] labels;
}