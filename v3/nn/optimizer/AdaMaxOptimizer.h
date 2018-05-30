/**
 * Created by wjy50 on 18-5-18.
 */

#ifndef NEURAL_NETWORK_ADAMAXOPTIMIZER_H
#define NEURAL_NETWORK_ADAMAXOPTIMIZER_H


#include "base/OptimizerBase.h"

class AdaMaxOptimizer : public OptimizerBase
{
private:
    FloatType *m, *u;
    FloatType learningRate, beta1, beta2;
    FloatType oneMBeta1;
    FloatType beta1T;
public:
    explicit AdaMaxOptimizer(FloatType learningRate = 0.002, FloatType beta1 = 0.9, FloatType beta2 = 0.999);

    void onAttachedToLayer() override;

    void update() override;

    ~AdaMaxOptimizer();
};


#endif //NEURAL_NETWORK_ADAMAXOPTIMIZER_H
