/**
 * Created by wjy50 on 18-5-18.
 */

#ifndef NEURAL_NETWORK_MEANPOOLINGLAYER_H
#define NEURAL_NETWORK_MEANPOOLINGLAYER_H


#include "base/LayerBase.h"

class MeanPoolingLayer : public LayerBase
{
private:
    int inputWidth, inputHeight;
    int inputChannel;
    int windowWidth, windowHeight;
    int xStride, yStride;

    int outputWidth, outputHeight;

    FloatType *delta;
protected:
    void computeGradients() override;

    void onInitialized() override;
public:
    MeanPoolingLayer(
            int inputWidth, int inputHeight, int inputChannel,
            int windowWidth, int windowHeight,
            int xStride = 0, int yStride = 0
    );

    FloatType *getDelta() override;

    const FloatType *feedForward(const FloatType *x) override;

    const FloatType *feedForwardForOptimization(const FloatType *x) override;

    void backPropagate(const FloatType *y) override;

    int getOutputWidth();

    int getOutputHeight();

    int getChannelCount();

    ~MeanPoolingLayer() override;
};


#endif //NEURAL_NETWORK_MEANPOOLINGLAYER_H
