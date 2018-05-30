/**
 * Created by wjy50 on 18-4-24.
 */

#ifndef NEURAL_NETWORK_CONVLAYER_H
#define NEURAL_NETWORK_CONVLAYER_H


#include "AbsLayer.h"
#include "../../../cuda/CUDAHelper.h"

namespace ffw
{
    class ConvLayer : public AbsLayer
    {
    private:
        Activator activator;
        FloatType (*activation)(FloatType);          /*激活函数*/
        FloatType (*dActivation_dx)(FloatType);      /*激活函数导函数*/

        int inputWidth, inputHeight;
        int inputChannel;
        int inputSize;

        int xStride, yStride;
        int xPadding, yPadding;

        int kernelWidth, kernelHeight;
        int kernelSize;
        int kernelCount;

        int outputWidth, outputHeight;
        int outputSize;

        /*if ENABLE_CUDA, these should be allocated in GPU memory*/
        FloatType *kernels;
        FloatType *biases;

        FloatType *weightGradient;
        FloatType *biasGradient;

        FloatType *z;
        FloatType *a;

        FloatType *delta;

        int right, bottom;

        int weightCount;

        int miniBatchSize;

#if ENABLE_CUDA
        FloatType *cuKernels;
        FloatType *cuBiases;
#else
        void convolution(const FloatType *input, const FloatType *kernel, FloatType *out);
#endif

    public:
        /**
         * 构造卷积层
         * @param inputWidth 输入数据宽度
         * @param inputHeight 输入数据高度
         * @param inputChannel 输入通道数
         * @param kernelWidth 卷积核宽度
         * @param kernelHeight 卷积核高度
         * @param kernelCount 卷积核数
         * @param xStride x方向步长
         * @param yStride y方向步长
         * @param xPadding x方向'0'填充数（单边）
         * @param yPadding y方向'0'填充数（单边）
         * @param activator 激活函数
         */
        ConvLayer(int inputWidth, int inputHeight, int inputChannel, int kernelWidth, int kernelHeight,
                  int kernelCount, int xStride, int yStride, int xPadding, int yPadding,
                  Activator activator);

        void initialize(int miniBatchSize) override;

        void feedForward(const FloatType *x) override;

        void feedForwardForOptimization(const FloatType *x) override;

        const FloatType * getWeightedOutput() override;

        const FloatType * getActivationOutput() override;

        void computeOutputDelta(const FloatType *y) override;

        void computeBackPropDelta(FloatType *backPropDelta) override;

        void backPropagateDelta() override;

        void computeGradient(const FloatType *prevActivation) override;

        void updateParameters() override;

        /**
         * 获取输出矩阵的宽度（列数）
         * @return 输出宽度
         */
        int getOutputWidth();

        /**
         * 获取输出矩阵的高度（行数）
         * @return 输出高度
         */
        int getOutputHeight();

        /**
         * 获取卷积核数
         * @return 卷积核数
         */
        int getKernelCount();

        FloatType *getDelta() override;

        int getWeightCount() override;

        int getBiasCount() override;

        ~ConvLayer();
    };
}


#endif //NEURAL_NETWORK_CONVLAYER_H
