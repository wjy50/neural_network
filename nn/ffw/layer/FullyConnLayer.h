/**
 * Created by wjy50 on 18-4-24.
 */

#ifndef NEURAL_NETWORK_FULLYCONNLAYER_H
#define NEURAL_NETWORK_FULLYCONNLAYER_H


#include "AbsLayer.h"

namespace ffw
{
    class FullyConnLayer : public AbsLayer
    {
    private:
        double *weights;
        double *transposedWeights;
        double *biases;

        double *weightGradient;
        double *biasGradient;

        double *z;
        double *a;
        double *delta;

        Activator activator;
        double (*activation)(double);          /*激活函数*/
        double (*dActivation_dx)(double);      /*激活函数导函数*/

        double regParam;

        int dropoutCount = 0;
        int *neuronIds;
    public:
        /**
         * 构造全连接层
         * @param neuronCount 神经元数
         * @param inputDim 输入数据维数
         * @param activator 激活函数
         */
        FullyConnLayer(int neuronCount, int inputDim, Activator activator);

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

        double *getDelta() override;

        /**
         * 设置dropout比例
         * @param dropoutFraction [0, 1)的浮点数
         */
        void setDropoutFraction(double dropoutFraction);

        ~FullyConnLayer();
    };
}


#endif //NEURAL_NETWORK_FULLYCONNLAYER_H
