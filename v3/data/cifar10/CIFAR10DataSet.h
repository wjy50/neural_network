/**
 * Created by wjy50 on 18-5-18.
 */

#ifndef NEURAL_NETWORK_CIFAR10DATASET_H
#define NEURAL_NETWORK_CIFAR10DATASET_H


#include "../base/DataSetBase.h"
#include "../../utils/UniquePointerExt.h"
#include "../base/Data2Bmp.h"

class CIFAR10Data2Bmp : public Data2Bmp
{
public:
    explicit CIFAR10Data2Bmp(const char *path, int w = 32, int h = 32);

    void writeData(const FloatType *data) override;
};

class CIFAR10DataSet : public DataSetBase
{
private:
    int count;

    unsigned char *buffer;
public:
    CIFAR10DataSet(const char **path, int n);

    int getCount() override;

    void getBatch(FloatType *data, FloatType *labels, const int *indices, int count) override;

    ~CIFAR10DataSet();
};


#endif //NEURAL_NETWORK_CIFAR10DATASET_H
