/**
 * Created by wjy50 on 18-4-20.
 */

#ifndef NEURAL_NETWORK_PERMUTATION_H
#define NEURAL_NETWORK_PERMUTATION_H

#include <random>
#include "../def/CUDAEnvironment.h"

#if !ENABLE_CUDA

/**
 * 生成从0到n的随机排列
 * @param arr 结果容器
 * @param n
 */
void randomPermutation(int *arr, int n, int bias = 0);

#endif

#endif //NEURAL_NETWORK_PERMUTATION_H
