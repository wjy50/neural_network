/**
 * Created by wjy50 on 18-5-5.
 */

#ifndef NEURAL_NETWORK_ADAMOPTIMIZER_H
#define NEURAL_NETWORK_ADAMOPTIMIZER_H


#include "AbsOptimizer.h"

class AdamOptimizer : public AbsOptimizer
{
private:
    FloatType *s, *r;
    FloatType beta1, beta2;
    FloatType oneMBeta1, oneMBeta2;
    FloatType beta1T, beta2T;
    FloatType epsilon;
public:
    AdamOptimizer(int weightSize, int biasSize, FloatType learningRate = 0.001, FloatType beta1 = 0.9, FloatType beta2 = 0.999);

    void update(FloatType *weights, const FloatType *weightGradients, FloatType *biases,
                const FloatType *biasGradients) override;

    ~AdamOptimizer();
};


#endif //NEURAL_NETWORK_ADAMOPTIMIZER_H
