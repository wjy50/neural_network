/**
 * Created by wjy50 on 18-6-1.
 */

#ifndef NEURAL_NETWORK_BATCHNORMLAYER_H
#define NEURAL_NETWORK_BATCHNORMLAYER_H


#include "base/LayerBase.h"

class BatchNormLayer : public LayerBase
{
private:
    FloatType *delta, *normOut;
    FloatType *gamma, *beta;
    FloatType *gammaGradient, *betaGradient;
    FloatType *avg, *var, *oneDivDev;
    FloatType *xSubAvg, *deltaAvg, *normDelta, *deltaMulCenter;
    FloatType *globalAvg, *globalVar, *globalOneDivDev;

    int miniBatchCount;
protected:
    void onInitialized() override;

    void computeGradients() override;
public:
    explicit BatchNormLayer(int dim);

    const FloatType *feedForward(const FloatType *x) override;

    const FloatType *feedForwardForOptimization(const FloatType *x) override;

    void backPropagate(const FloatType *y) override;

    FloatType *getDelta() override;

    ~BatchNormLayer() override;
};


#endif //NEURAL_NETWORK_BATCHNORMLAYER_H
