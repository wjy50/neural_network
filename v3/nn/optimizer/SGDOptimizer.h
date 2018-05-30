/**
 * Created by wjy50 on 18-5-18.
 */

#ifndef NEURAL_NETWORK_SGDOPTIMIZER_H
#define NEURAL_NETWORK_SGDOPTIMIZER_H


#include "base/OptimizerBase.h"

class SGDOptimizer : public OptimizerBase
{
private:
    FloatType eta;
    FloatType reg;
    int regRange;
public:
    SGDOptimizer(FloatType learningRate, FloatType regParam, int regRange);

    void setParams(FloatType learningRate, FloatType regParam, int regRange);

    void update() override;
};


#endif //NEURAL_NETWORK_SGDOPTIMIZER_H
