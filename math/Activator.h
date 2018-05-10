/**
 * Created by wjy50 on 18-4-16.
 */

#ifndef NEURAL_NETWORK_ACTIVATOR_H
#define NEURAL_NETWORK_ACTIVATOR_H

#include "mtype.h"

#ifdef __cplusplus
extern "C"
{
#endif

FloatType sigmoid(FloatType x);

FloatType dSigmoid_dx(FloatType x);

FloatType ReLU(FloatType x);

FloatType dReLU_dx(FloatType x);

FloatType lReLU(FloatType x);

FloatType dLReLU_dx(FloatType x);

/**
 * SoftMax激活函数，用于输出层
 * @param x 输出层的带权输出向量
 * @param n 维数
 */
void softMax(FloatType *x, int n);

/**
 * softMax激活函数，用于输出层，
 * @param r 结果容器
 * @param x 待激活向量
 * @param n 向量维数
 */
void softMaxInto(FloatType *r, FloatType *x, int n);

#ifdef __cplusplus
};
#endif

#endif //NEURAL_NETWORK_ACTIVATOR_H
