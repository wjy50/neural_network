/**
 * Created by wjy50 on 18-5-18.
 */

#ifndef NEURAL_NETWORK_LINEARLAYER_H
#define NEURAL_NETWORK_LINEARLAYER_H


#include "base/LayerBase.h"

class LinearLayer : public LayerBase
{
private:
    FloatType *weights;
    FloatType *biases;

    FloatType *weightGradients;
    FloatType *biasGradients;

    FloatType *delta;

    const FloatType *in;
protected:
    void computeGradients() override;

    void onInitialized() override;
public:
    LinearLayer(int inputDim, int neuronCount, bool noBias = false);

    FloatType *getDelta() override;

    const FloatType *feedForward(const FloatType *x, int count) override;

    const FloatType *feedForwardForOptimization(const FloatType *x) override;

    void backPropagate(const FloatType *y) override;

    ~LinearLayer() override;
};


#endif //NEURAL_NETWORK_LINEARLAYER_H
