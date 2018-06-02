/**
 * Created by wjy50 on 18-6-2.
 */

#ifndef NEURAL_NETWORK_AUTOENCODER_H
#define NEURAL_NETWORK_AUTOENCODER_H

#include <vector>

#include "layer/base/LayerBase.h"
#include "../data/base/DataSetBase.h"

class AutoEncoder
{
private:
    std::vector<LayerBase*> layers;

    bool built;

    FloatType *inputs;

    int miniBatchSize;
public:
    AutoEncoder();

    void addLayer(LayerBase *layer);

    void buildUpAutoEncoder(int miniBatchSize);

    const FloatType *feedForward(const FloatType *x);

    void optimize(DataSetBase &trainSet, int altTrainSetSize = 0);

    ~AutoEncoder();
};


#endif //NEURAL_NETWORK_AUTOENCODER_H
