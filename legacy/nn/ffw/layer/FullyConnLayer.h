/**
 * Created by wjy50 on 18-4-24.
 */

#ifndef NEURAL_NETWORK_FULLYCONNLAYER_H
#define NEURAL_NETWORK_FULLYCONNLAYER_H


#include "AbsLayer.h"
#include "../../../cuda/CUDAHelper.h"

namespace ffw
{
    class FullyConnLayer : public AbsLayer
    {
    private:
        FloatType *weights;
        FloatType *biases;

        /*if ENABLE_CUDA, these should be allocated in GPU memory*/
        FloatType *weightGradient;
        FloatType *biasGradient;

        FloatType *z;
        FloatType *a;
        FloatType *delta;

        Activator activator;
        FloatType (*activation)(FloatType);          /*激活函数*/
        FloatType (*dActivation_dx)(FloatType);      /*激活函数导函数*/

        int dropoutCount = 0;
        /*if ENABLE_CUDA, this should be allocated in GPU memory*/
        int *neuronIds;

        int weightCount;

        int miniBatchSize;

#if ENABLE_CUDA

        FloatType *cuWeights;
        FloatType *cuBiases;

#endif

    public:
        /**
         * 构造全连接层
         * @param neuronCount 神经元数
         * @param inputDim 输入数据维数
         * @param activator 激活函数
         */
        FullyConnLayer(int neuronCount, int inputDim, Activator activator);

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

        FloatType *getDelta() override;

        /**
         * 设置dropout比例
         * @param dropoutFraction [0, 1)的浮点数
         */
        void setDropoutFraction(FloatType dropoutFraction);

        int getWeightCount() override;

        int getBiasCount() override;

        ~FullyConnLayer();
    };
}


#endif //NEURAL_NETWORK_FULLYCONNLAYER_H
