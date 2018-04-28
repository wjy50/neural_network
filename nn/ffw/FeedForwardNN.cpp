/**
 * Created by wjy50 on 18-4-24.
 */

#include <cassert>
#include <ctime>
#include <iostream>

#include "FeedForwardNN.h"
#include "../../utils/UniquePointerExt.h"
#include "../../math/permutation.h"
#include "../../data/DataSet.h"

using namespace ffw;

FeedForwardNN::FeedForwardNN()
{
    built = false;
}

void FeedForwardNN::addLayer(ffw::AbsLayer *layer)
{
    assert(!built);

    layers.push_back(layer);
}

void FeedForwardNN::buildUpNetwork()
{
    built = true;
    for (AbsLayer *layer : layers) {
        layer->initialize();
    }
}

const double * FeedForwardNN::feedForward(const double *x)
{
    const double *a = x;
    for (AbsLayer *layer : layers) {
        layer->feedForward(a);
        a = layer->getActivationOutput();
    }
    return a;
}

void FeedForwardNN::SGD(DataSet &trainSet, size_t miniBatchSize, size_t altTrainSetSize)
{
    size_t trainSetSize = altTrainSetSize > 0 ? altTrainSetSize : trainSet.getSize();
    unique_ptr<size_t[]> indices = make_unique_array<size_t[]>(trainSetSize);
    randomPermutation<size_t>(indices.get(), trainSetSize);
    size_t miniBatchCount = trainSetSize / miniBatchSize;
    for (int t = 0; t < miniBatchCount; ++t) {
        for (AbsLayer *layer : layers) {
            layer->clearGradient();
        }

        size_t *ind = indices.get()+t*miniBatchSize;
        for (int i = 0; i < miniBatchSize; ++i) {
            //long st = clock();
            const double *in = trainSet.getData(ind[i]);
            const double *a = in;
            /*for (int j = 0; j < 28; ++j) {
                for (int k = 0; k < 28; ++k) {
                    cout << ((int)(in[j*28+k]*9) == 0 ? ' ' : '0') << ' ';
                }
                cout << endl;
            }
            cout << endl;*/
            for (AbsLayer *layer : layers) {
                layer->feedForward(in);
                in = layer->getActivationOutput();
            }
            /*for (int j = 0; j < layers[layers.size() - 1]->getNeuronCount(); ++j) {
                cout << ((int)(in[j]*10)) << ' ';
            }
            cout << endl;*/

            auto layerCount = static_cast<int>(layers.size());
            layers[layerCount-1]->computeOutputDelta(trainSet.getLabel(ind[i]));
            for (int j = layerCount-2; j >= 0; --j) {
                layers[j+1]->computeBackPropDelta(layers[j]->getDelta());
                layers[j]->backPropagateDelta();
            }

            for (AbsLayer *layer : layers) {
                layer->accumulateGradient(a);
                a = layer->getActivationOutput();
            }
            //cout << clock()-st << endl;
        }

        for (AbsLayer *layer : layers) {
            layer->updateParameters(miniBatchSize, trainSetSize);
        }
    }
}