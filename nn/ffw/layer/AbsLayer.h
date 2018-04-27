/**
 * Created by wjy50 on 18-4-24.
 */

#ifndef NEURAL_NETWORK_ABSLAYER_H
#define NEURAL_NETWORK_ABSLAYER_H

#include <cstddef>
#include "../../../math/Activator.h"

namespace ffw
{
    static double (*ACTIVATION_FUNCTIONS[])(double) = {
            sigmoid,
            ReLU,
            lReLU
    };

    static double (*D_ACTIVATION_FUNCTIONS[])(double) = {
            dSigmoid_dx,
            dReLU_dx,
            dLReLU_dx
    };

    typedef enum ACTIVATOR
    {
        SIGMOID,
        RE_LU,
        L_RE_LU,
        OUTPUT_ACTIVATOR
    }Activator;

    class AbsLayer
    {
    protected:
        double learningRate;
        int neuronCount;
        int inputDim;
    public:
        AbsLayer(int neuronCount, int inputDim);

        /**
         * 获取合法的输入数据（视为向量）的维数
         * @return 输入数据的维数
         */
        int getInputDimension();

        /**
         * 初始化神经网络（参数等）
         */
        virtual void initialize() = 0;

        /**
         * 获取神经元个数
         * @return 神经元个数
         */
        int getNeuronCount();

        /**
         * 设置学习率
         * @param learningRate 学习率
         */
        void setLearningRate(double learningRate);

        /**
         * 前馈计算输出
         * @param x 输入向量/矩阵
         */
        virtual void feedForward(const double *x) = 0;

        /**
         * 获取（上一次输入的）带权输出
         * @return 带权输出向量/矩阵
         */
        virtual const double * getWeightedOutput() = 0;

        /**
         * 获取（上一次输入的）激活输出
         * @return 激活输出向量/矩阵
         */
        virtual const double * getActivationOutput() = 0;

        /**
         * （从上次输出）计算误差项，只能用于输出层
         * @param y 期望输出
         */
        virtual void computeOutputDelta(const double *y) = 0;

        /**
         * 预处理反向传播中前一层需要用到的误差信息
         * @param backPropDelta 前一层误差项容器
         */
        virtual void computeBackPropDelta(double *backPropDelta) = 0;

        /**
         * 根据后一层提供的误差信息反向传播计算误差项
         */
        virtual void backPropagateDelta() = 0;

        /**
         * 获取误差项
         */
        virtual double *getDelta() = 0;

        /**
         * 清空梯度信息
         */
        virtual void clearGradient() = 0;

        /**
         * 完成一次反向传播后累积梯度信息
         * 以备一个batch或mini batch后更新参数
         * @param prevActivation 前一层的激活输出
         */
        virtual void accumulateGradient(const double *prevActivation) = 0;

        /**
         * 完成一个batch或mini batch后
         * 根据已累积的梯度信息更新参数（权重和偏置）
         * @param batchSize batch或mini batch大小
         * @param trainSetSize 当前epoch的训练集大小
         */
        virtual void updateParameters(size_t batchSize, size_t trainSetSize) = 0;
    };
}

#endif //NEURAL_NETWORK_ABSLAYER_H
