/**
 * Created by wjy50 on 18-5-5.
 */

#ifndef NEURAL_NETWORK_SIMPLESGDOPTIMIZER_H
#define NEURAL_NETWORK_SIMPLESGDOPTIMIZER_H


#include <cstddef>
#include "AbsOptimizer.h"

class SimpleSGDOptimizer : public AbsOptimizer
{
private:
    FloatType eta;
    FloatType reg;
public:
    SimpleSGDOptimizer(FloatType learningRate, FloatType regParam, int trainSetSize, int weightSize, int biasSize);

    void update(FloatType *weights, const FloatType *weightGradients, FloatType *biases,
                const FloatType *biasGradients) override;
};


#endif //NEURAL_NETWORK_SIMPLESGDOPTIMIZER_H
