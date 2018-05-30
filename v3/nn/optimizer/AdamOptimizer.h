/**
 * Created by wjy50 on 18-5-18.
 */

#ifndef NEURAL_NETWORK_ADAMOPTIMIZER_H
#define NEURAL_NETWORK_ADAMOPTIMIZER_H


#include "base/OptimizerBase.h"

class AdamOptimizer : public OptimizerBase
{
private:
    FloatType *m, *v;
    FloatType beta1, beta2;
    FloatType oneMBeta1, oneMBeta2;
    FloatType beta1T, beta2T;
    FloatType alpha;
public:
    explicit AdamOptimizer(FloatType learningRate = 0.001, FloatType beta1 = 0.9, FloatType beta2 = 0.999);

    void onAttachedToLayer() override;

    void update() override;

    ~AdamOptimizer();
};


#endif //NEURAL_NETWORK_ADAMOPTIMIZER_H
