/**
 * Created by wjy50 on 18-5-18.
 */

#ifndef NEURAL_NETWORK_OPTIMIZERBASE_H
#define NEURAL_NETWORK_OPTIMIZERBASE_H


#include "../../../def/type.h"

class OptimizerBase
{
protected:
    bool attached;

    int paramCount;
    FloatType *params;
    FloatType *gradients;
public:
    OptimizerBase();

    void attachToLayer(int paramCount, FloatType *params, FloatType *gradients);

    virtual void onAttachedToLayer();

    virtual void update() = 0;

    virtual ~OptimizerBase() = default;
};


#endif //NEURAL_NETWORK_OPTIMIZERBASE_H
