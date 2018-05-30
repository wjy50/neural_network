/**
 * Created by wjy50 on 18-5-18.
 */

#ifndef NEURAL_NETWORK_CONVLAYER_H
#define NEURAL_NETWORK_CONVLAYER_H


#include "base/LayerBase.h"

class ConvLayer : public LayerBase
{
private:
    FloatType *kernels;
    FloatType *biases;

    FloatType *kernelGradients;
    FloatType *biasGradients;

    int inputWidth, inputHeight;
    int inputChannel;
    int kernelWidth, kernelHeight;
    int kernelCount;
    int outputWidth, outputHeight;

    int xStride, yStride;
    int xPadding, yPadding;

    FloatType *delta;

    const FloatType *in;
protected:
    void onInitialized() override;

    void computeGradients() override;
public:
    ConvLayer(
            int inputWidth, int inputHeight, int inputChannel,
            int kernelWidth, int kernelHeight, int kernelCount,
            int xStride = 1, int yStride = 1, int xPadding = 0, int yPadding = 0,
            bool noBias = false
    );

    FloatType *getDelta() override;

    const FloatType *feedForward(const FloatType *x) override;

    const FloatType *feedForwardForOptimization(const FloatType *x) override;

    void backPropagate(const FloatType *y) override;

    int getOutputWidth();

    int getOutputHeight();

    int getKernelCount();

    ~ConvLayer() override;
};


#endif //NEURAL_NETWORK_CONVLAYER_H
