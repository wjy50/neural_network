/**
 * Created by wjy50 on 18-4-16.
 */

#ifndef NEURAL_NETWORK_ACTIVATOR_H
#define NEURAL_NETWORK_ACTIVATOR_H

#include <complex>

#ifdef __cplusplus
extern "C"
{
#endif

    double sigmoid(double x);

    double dSigmoid_dx(double x);

    double ReLU(double x);

    double dReLU_dx(double x);

    double lReLU(double x);

    double dLReLU_dx(double x);

    /**
     * SoftMax激活函数，用于输出层
     * @param x 输出层的带权输出向量
     * @param n 维数
     */
    void softMax(double *x, int n);

#ifdef __cplusplus
};
#endif

#endif //NEURAL_NETWORK_ACTIVATOR_H
