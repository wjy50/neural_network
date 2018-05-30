/**
 * Created by wjy50 on 18-5-18.
 */

#ifndef NEURAL_NETWORK_SOFTMAXLAYER_H
#define NEURAL_NETWORK_SOFTMAXLAYER_H


#include "../base/LayerBase.h"

class SoftMaxOutputLayer : public LayerBase
{
protected:
    void computeGradients() override;
public:
    explicit SoftMaxOutputLayer(int dim);

    FloatType *getDelta() override;

    const FloatType * feedForward(const FloatType *x) override;

    const FloatType *feedForwardForOptimization(const FloatType *x) override;

    void backPropagate(const FloatType *y) override;
};


#endif //NEURAL_NETWORK_SOFTMAXLAYER_H
