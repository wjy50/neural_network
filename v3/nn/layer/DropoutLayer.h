/**
 * Created by wjy50 on 18-5-31.
 *
 * DropoutLayer应当放在隐含LinearLayer的激活后面
 */

#ifndef NEURAL_NETWORK_DROPOUTLAYER_H
#define NEURAL_NETWORK_DROPOUTLAYER_H


#include "base/LayerBase.h"

class DropoutLayer : public LayerBase
{
private:
    int dropoutCount;
    int *dropoutIds;
protected:
    bool needIndependentOutput() override;

    void computeGradients() override;
public:
    explicit DropoutLayer(int dim);

    void setDropoutFraction(FloatType f);

    const FloatType *feedForward(const FloatType *x) override;

    const FloatType *feedForwardForOptimization(const FloatType *x) override;

    void backPropagate(const FloatType *y) override;

    FloatType *getDelta() override;

    ~DropoutLayer() override;
};


#endif //NEURAL_NETWORK_DROPOUTLAYER_H
