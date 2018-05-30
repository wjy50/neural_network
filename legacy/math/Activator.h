/**
 * Created by wjy50 on 18-4-16.
 */

#ifndef NEURAL_NETWORK_ACTIVATOR_H
#define NEURAL_NETWORK_ACTIVATOR_H

#include "../cuda/CUDAHelper.h"

#if ENABLE_CUDA
//TODO cuda
#else

#include "mtype.h"

FloatType sigmoid(FloatType x);

FloatType dSigmoid_dx(FloatType x);

FloatType ReLU(FloatType x);

FloatType dReLU_dx(FloatType x);

FloatType lReLU(FloatType x);

FloatType dLReLU_dx(FloatType x);

/**
 * softMax激活函数，用于输出层，
 * @param r 结果容器
 * @param x 待激活向量
 * @param n 向量维数
 */
void softMax(FloatType *r, FloatType *x, int n);

#endif //ENABLE_CUDA

#endif //NEURAL_NETWORK_ACTIVATOR_H
