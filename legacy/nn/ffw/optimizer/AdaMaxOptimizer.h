/**
 * Created by wjy50 on 18-5-6.
 */

#ifndef NEURAL_NETWORK_ADAMAXOPTIMIZER_H
#define NEURAL_NETWORK_ADAMAXOPTIMIZER_H


#include "AbsOptimizer.h"

class AdaMaxOptimizer : public AbsOptimizer
{
protected:
    FloatType *m, *u;
    FloatType learningRate, beta1, beta2;
    FloatType oneMBeta1;
    FloatType beta1T;
public:
    AdaMaxOptimizer(int weightCount, int biasCount, FloatType learningRate = 0.002, FloatType beta1 = 0.9, FloatType beta2 = 0.999);

    void update(FloatType *weights, const FloatType *weightGradients, FloatType *biases,
                const FloatType *biasGradients) override;

    ~AdaMaxOptimizer();
};

#endif //NEURAL_NETWORK_ADAMAXOPTIMIZER_H
