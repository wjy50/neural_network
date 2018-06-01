/**
 * Created by wjy50 on 18-5-18.
 */

#ifndef NEURAL_NETWORK_LAYERBASE_H
#define NEURAL_NETWORK_LAYERBASE_H


#include "../../../def/type.h"
#include "../../optimizer/base/OptimizerBase.h"

class LayerBase
{
protected:
    int inputDim, outputDim;

    int miniBatchSize;

    OptimizerBase *optimizer;

    int paramCount;
    FloatType *params;
    FloatType *gradients;

    void allocParamsAndGradients(int count);

    FloatType *output;

    void allocOutput();

    FloatType *deltaOutput;

    virtual void computeGradients() = 0;

    virtual void onInitialized();

    virtual bool needIndependentOutput();
public:
    LayerBase(int inputDim, int outputDim);

    int getInputDim();

    int getOutputDim();

    void setOptimizer(OptimizerBase *optimizer);

    void initialize(int miniBatchSize);

    void setDeltaOutput(FloatType *deltaOutput);

    virtual FloatType *getDelta() = 0;

    virtual const FloatType *feedForward(const FloatType *x) = 0;

    virtual const FloatType *feedForwardForOptimization(const FloatType *x) = 0;

    virtual void backPropagate(const FloatType *y) = 0;

    void updateParameters();

    virtual ~LayerBase();
};


#endif //NEURAL_NETWORK_LAYERBASE_H
