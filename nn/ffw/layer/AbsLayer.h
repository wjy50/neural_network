/**
 * Created by wjy50 on 18-4-24.
 */

#ifndef NEURAL_NETWORK_ABSLAYER_H
#define NEURAL_NETWORK_ABSLAYER_H

#include <cstddef>
#include "../../../math/Activator.h"
#include "../optimizer/AbsOptimizer.h"

namespace ffw
{
    static FloatType (*ACTIVATION_FUNCTIONS[])(FloatType) = {
            sigmoid,
            ReLU,
            lReLU
    };

    static FloatType (*D_ACTIVATION_FUNCTIONS[])(FloatType) = {
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
        int neuronCount;
        int inputDim;

        AbsOptimizer *optimizer;
    public:
        AbsLayer(int neuronCount, int inputDim);

        /**
         * 获取合法的输入数据（视为向量）的维数
         * @return 输入数据的维数
         */
        int getInputDimension();

        /**
         * 初始化神经网络（参数等）
         * @param miniBatchSize 训练时mini batch的大小
         */
        virtual void initialize(int miniBatchSize) = 0;

        /**
         * 获取神经元个数
         * @return 神经元个数
         */
        int getNeuronCount();

        /**
         * 前馈计算输出，仅在测试或应用时使用，可减少一些不必要的运算
         * @param x 单个的输入向量/矩阵
         */
        virtual void feedForward(const FloatType *x) = 0;

        /**
         * 前馈计算输出，仅在训练时使用
         * @param x 包含miniBatchSize个输入向量/矩阵
         */
        virtual void feedForwardForOptimization(const FloatType *x) = 0;

        /**
         * 设置优化器（算法）
         * @param optimizer
         */
        void setOptimizer(AbsOptimizer *optimizer);

        /**
         * 获取（上一次输入的）带权输出
         * @return 带权输出向量/矩阵
         */
        virtual const FloatType * getWeightedOutput() = 0;

        /**
         * 获取（上一次输入的）激活输出
         * @return 激活输出向量/矩阵
         */
        virtual const FloatType * getActivationOutput() = 0;

        /**
         * （从上次输出）计算误差项，只能用于输出层
         * @param y 期望输出
         */
        virtual void computeOutputDelta(const FloatType *y) = 0;

        /**
         * 预处理反向传播中前一层需要用到的误差信息
         * @param backPropDelta 前一层误差项容器
         */
        virtual void computeBackPropDelta(FloatType *backPropDelta) = 0;

        /**
         * 根据后一层提供的误差信息反向传播计算误差项
         */
        virtual void backPropagateDelta() = 0;

        /**
         * 获取误差项
         */
        virtual FloatType *getDelta() = 0;

        /**
         * 根据误差项计算梯度信息
         * @param prevActivation 前一层的激活输出
         */
        virtual void computeGradient(const FloatType *prevActivation) = 0;

        /**
         * 根据梯度信息更新参数（权重和偏置）
         * @param batchSize （mini）batch大小
         */
        virtual void updateParameters() = 0;

        /**
         * 获取权重参数个数
         * @return
         */
        virtual int getWeightCount() = 0;

        /**
         * 获取偏置参数个数
         * @return
         */
        virtual int getBiasCount() = 0;
    };
}

#endif //NEURAL_NETWORK_ABSLAYER_H
