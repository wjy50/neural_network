/**
 * Created by wjy50 on 18-4-24.
 */

#ifndef NEURAL_NETWORK_POOLINGLAYER_H
#define NEURAL_NETWORK_POOLINGLAYER_H


#include "AbsLayer.h"

namespace ffw
{
    typedef enum POOLING_METHOD
    {
        MAX_POOLING,
        MEAN_POOLING
    }PoolingMethod;

    class PoolingLayer : public AbsLayer
    {
    private:
        int inputWidth, inputHeight;
        int inputSize;
        int xStride, yStride;
        int outputWidth, outputHeight;
        int outputSize;
        int channel;

        PoolingMethod poolingMethod;

        double *output;
        int *xOffset;
        int *yOffset;

        double *delta;

        /**
         * 找出数组中最大元素，返回下标
         * @param arr 数组
         * @param n 数组长度
         * @return 最大元素下标
         */
        int indexOfMax(const double *arr, int n);

        /**
         * 最大值池化
         * @param x 输入
         */
        void maxPooling(const double *x);

        /**
         * 平均值池化
         * @param x 输入
         */
        void meanPooling(const double *x);

        /**
         * 最大值池化下反向传播
         * @param backPropDelta 前一层误差项容器
         */
        void maxBackProp(double *backPropDelta);

        /**
         * 平均值池化下反向传播
         * @param backPropDelta 前一层误差项容器
         */
        void meanBackProp(double *backPropDelta);
    public:
        /**
         * 构造池化层
         * @param inputWidth 输入数据宽度
         * @param inputHeight 输入数据高度
         * @param xStride x方向步长，即窗口宽度
         * @param yStride y方向步长，即窗口高度
         * @param channel 通道数
         * @param poolingMethod 池化方法
         */
        PoolingLayer(int inputWidth, int inputHeight, int xStride, int yStride, int channel, PoolingMethod poolingMethod);

        void initialize() override;

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
         * 获取通道数
         * @return 通道数
         */
        int getChannelCount();

        double *getDelta() override;

        ~PoolingLayer();
    };
}


#endif //NEURAL_NETWORK_POOLINGLAYER_H
