/**
 * Created by wjy50 on 18-4-24.
 */

#ifndef NEURAL_NETWORK_CONVLAYER_H
#define NEURAL_NETWORK_CONVLAYER_H


#include "AbsLayer.h"

namespace ffw
{
    class ConvLayer : public AbsLayer
    {
    private:
        Activator activator;
        double (*activation)(double);          /*激活函数*/
        double (*dActivation_dx)(double);      /*激活函数导函数*/

        double regParam;

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

        double *kernels;
        double *biases;

        double *weightGradient;
        double *biasGradient;

        double *z;
        double *a;

        double *delta;
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
                          int kernelCount, int xStride, int yStride, int xPadding, int yPadding, Activator activator);

        void initialize() override;

        /**
         * 设置L2正规化参数
         * @param regParam
         */
        void setRegParam(double regParam);

        void feedForward(const double *x) override;

        const double * getWeightedOutput() override;

        const double * getActivationOutput() override;

        void computeOutputDelta(const double *y) override;

        void computeBackPropDelta(double *backPropDelta) override;

        void backPropagateDelta() override;

        void clearGradient() override;

        void accumulateGradient(const double *prevActivation) override;

        void updateParameters(size_t batchSize, size_t trainSetSize) override;

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

        double *getDelta() override;

        ~ConvLayer();
    };
}


#endif //NEURAL_NETWORK_CONVLAYER_H
