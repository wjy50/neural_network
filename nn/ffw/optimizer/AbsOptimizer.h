/**
 * Created by wjy50 on 18-5-5.
 */

#ifndef NEURAL_NETWORK_ABSOPTIMIZER_H
#define NEURAL_NETWORK_ABSOPTIMIZER_H


#include <cstddef>
#include "../../../math/mtype.h"

class AbsOptimizer
{
protected:
    int weightSize, biasSize;
public:
    AbsOptimizer(int weightSize, int biasSize);

    virtual void update(FloatType *weights, const FloatType *weightGradients, FloatType *biases,
                        const FloatType *biasGradients) = 0;
};


#endif //NEURAL_NETWORK_ABSOPTIMIZER_H
