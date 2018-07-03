/**
 * Created by wjy50 on 18-5-18.
 */

#ifndef NEURAL_NETWORK_RELULAYER_H
#define NEURAL_NETWORK_RELULAYER_H


#include "../base/LayerBase.h"

class ReLULayer : public LayerBase
{
private:
    const FloatType *in;
protected:
    void computeGradients() override;
public:
    explicit ReLULayer(int dim);

    const FloatType * feedForward(const FloatType *x, int count) override;

    const FloatType *feedForwardForOptimization(const FloatType *x) override;

    void backPropagate(const FloatType *y) override;

    FloatType *getDelta() override;
};


#endif //NEURAL_NETWORK_RELULAYER_H
