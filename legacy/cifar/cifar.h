/**
 * Created by wjy50 on 18-4-27.
 */

#ifndef NEURAL_NETWORK_CIFAR_H
#define NEURAL_NETWORK_CIFAR_H

#include "../data/DataSet.h"

class CIFARDataSet : public DataSet
{
private:
    int size;

    int count;

    unsigned char *buffer;
public:
    explicit CIFARDataSet(const char **path, int n);

    int getSize() override;

    void getBatch(FloatType *data, FloatType *label, const int *indices, int n) override;

    /*void normalize();*/
};

class CIFARNormalizer : public DataNormalizer
{
private:
    FloatType avg[32*32*3];
    FloatType dev[32*32*3];

    bool confirmed;
    bool finished;

    int sampleCount;
    int sampleCount1;
public:
    CIFARNormalizer();

    void add(CIFARDataSet &dataSet);

    void div(CIFARDataSet &dataSet);

    void confirm();

    void finish();

    void normalize(FloatType *x) override;
};

#endif //NEURAL_NETWORK_CIFAR_H
