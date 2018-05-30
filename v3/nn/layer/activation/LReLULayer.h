/**
 * Created by wjy50 on 18-5-18.
 */

#ifndef NEURAL_NETWORK_LRELULAYER_H
#define NEURAL_NETWORK_LRELULAYER_H


#include "../base/LayerBase.h"

class LReLULayer : public LayerBase
{
private:
    FloatType l;

    const FloatType *in;
protected:
    void computeGradients() override;
public:
    LReLULayer(int dim, FloatType l);

    const FloatType * feedForward(const FloatType *x) override;

    const FloatType *feedForwardForOptimization(const FloatType *x) override;

    void backPropagate(const FloatType *y) override;

    FloatType *getDelta() override;
};


#endif //NEURAL_NETWORK_LRELULAYER_H
