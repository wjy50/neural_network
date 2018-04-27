/**
 * Created by wjy50 on 18-4-25.
 */

#ifndef NEURAL_NETWORK_DATASET_H
#define NEURAL_NETWORK_DATASET_H


#include <cstddef>

class DataSet
{
public:
    virtual size_t getSize() = 0;
    virtual double *get(size_t i) = 0;
};


#endif //NEURAL_NETWORK_DATASET_H
