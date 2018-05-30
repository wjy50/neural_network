/**
 * Created by wjy50 on 18-5-18.
 */

#ifndef NEURAL_NETWORK_CIFAR10DATASET_H
#define NEURAL_NETWORK_CIFAR10DATASET_H


#include "../base/DataSetBase.h"
#include "../../utils/UniquePointerExt.h"

class CIFAR10DataSet : public DataSetBase
{
private:
    int count;

    std::unique_ptr<unsigned char[]> buffer;
public:
    CIFAR10DataSet(const char **path, int n);

    int getCount() override;

    void getBatch(FloatType *data, FloatType *labels, const int *indices, int count) override;
};


#endif //NEURAL_NETWORK_CIFAR10DATASET_H
