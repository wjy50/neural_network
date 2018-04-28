/**
 * Created by wjy50 on 18-4-27.
 */

#ifndef NEURAL_NETWORK_CIFAR_H
#define NEURAL_NETWORK_CIFAR_H

#include "../data/DataSet.h"

class CIFARDataSet : public DataSet
{
private:
    size_t size;

    size_t count;

    unsigned char *buffer;

    double image[32*32*3];
    double label[10];
public:
    explicit CIFARDataSet(const char *path);

    size_t getSize() override;

    const double *getData(size_t i) override;

    const double *getLabel(size_t i) override;
};

#endif //NEURAL_NETWORK_CIFAR_H
