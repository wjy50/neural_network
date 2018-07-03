/**
 * Created by wjy50 on 18-5-18.
 */

#ifndef NEURAL_NETWORK_NEURALNETWORK_H
#define NEURAL_NETWORK_NEURALNETWORK_H


#include "../data/base/DataSetBase.h"
#include "layer/base/LayerBase.h"

#include <vector>

class NeuralNetwork
{
private:
    std::vector<LayerBase*> layers;

    bool built;

    FloatType *inputs, *labels;

    int *indices;
    int indexCap;

    int miniBatchSize;
public:
    NeuralNetwork();

    void addLayer(LayerBase *layer);

    void buildUpNetwork(int miniBatchSize);

    const FloatType *feedForward(const FloatType *x, int count = 1);

    void optimize(DataSetBase &trainSet, int altTrainSetSize = 0);

    void ensureIndexCap(int size);

    FloatType *getInputBuffer();

    FloatType *getLabelBuffer();

    ~NeuralNetwork();
};


#endif //NEURAL_NETWORK_NEURALNETWORK_H
