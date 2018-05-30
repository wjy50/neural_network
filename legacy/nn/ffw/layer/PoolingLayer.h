/**
 * Created by wjy50 on 18-4-24.
 */

#ifndef NEURAL_NETWORK_POOLINGLAYER_H
#define NEURAL_NETWORK_POOLINGLAYER_H


#include "AbsLayer.h"
#include "../../../cuda/CUDAHelper.h"

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
        int windowWidth, windowHeight;
        int windowSize;
        int xStride, yStride;
        int outputWidth, outputHeight;
        int outputSize;
        int channel;

        PoolingMethod poolingMethod;

        /*if ENABLE_CUDA, these should be allocated in GPU memory*/
        FloatType *output;
        int *xOffset;
        int *yOffset;

        FloatType *delta;

        int miniBatchSize;

#if ENABLE_CUDA

        //TODO cuda

#else

        /**
         * 找出数组中最大元素，返回下标
         * @param arr 数组
         * @param n 数组长度
         * @return 最大元素下标
         */
        int indexOfMax(const FloatType *arr, int n);

        /**
         * 最大值池化
         * @param x 输入
         * @param output 输出
         * @param xOffset x位置记录
         * @param yOffset y位置记录
         */
        void maxPooling(const FloatType *x, FloatType *output, int *xOffset, int *yOffset);

        /**
         * 平均值池化
         * @param x 输入
         * @param output 输出
         */
        void meanPooling(const FloatType *x, FloatType *output);

        /**
         * 最大值池化下反向传播
         * @param backPropDelta 前一层误差项容器
         * @param delta 本层误差项
         * @param xOffset x位置记录
         * @param yOffset y位置记录
         */
        void maxBackProp(FloatType *backPropDelta, const FloatType *delta, int *xOffset, int *yOffset);

        /**
         * 平均值池化下反向传播
         * @param backPropDelta 前一层误差项容器
         * @param delta 本层误差项
         */
        void meanBackProp(FloatType *backPropDelta, const FloatType *delta);

#endif
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
        PoolingLayer(int inputWidth, int inputHeight, int windowWidth, int windowHeight, int xStride, int yStride, int channel,
                     PoolingMethod poolingMethod);

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
         * 获取通道数
         * @return 通道数
         */
        int getChannelCount();

        FloatType *getDelta() override;

        int getWeightCount() override;

        int getBiasCount() override;

        ~PoolingLayer();
    };
}


#endif //NEURAL_NETWORK_POOLINGLAYER_H
