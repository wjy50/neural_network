/**
 * Created by wjy50 on 18-5-18.
 */

#ifndef NEURAL_NETWORK_DATASETBASE_H
#define NEURAL_NETWORK_DATASETBASE_H


#include "../../def/type.h"

class DataSetBase
{
public:
    virtual int getCount() = 0;

    virtual void getBatch(FloatType *data, FloatType *labels, const int *indices, int count) = 0;
};


#endif //NEURAL_NETWORK_DATASETBASE_H
