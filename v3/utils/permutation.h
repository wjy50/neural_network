/**
 * Created by wjy50 on 18-4-20.
 */

#ifndef NEURAL_NETWORK_PERMUTATION_H
#define NEURAL_NETWORK_PERMUTATION_H

#include <random>

/**
 * 生成从0到n的随机排列
 * @param arr 结果容器
 * @param n
 */
template<typename _IntType>
void randomPermutation(_IntType *arr, _IntType n)
{
    static_assert(std::is_integral<_IntType>::value,
                  "template argument not an integral type");
    for (_IntType i = 0; i < n; ++i) {
        arr[i] = i;
    }
    std::random_device rd;
    std::uniform_int_distribution<_IntType> distribution(0, n);
    for (_IntType i = n-1; i > 0; --i) {
        _IntType d = distribution(rd) % i;
        _IntType tmp = arr[d];
        arr[d] = arr[i];
        arr[i] = tmp;
    }
}

#endif //NEURAL_NETWORK_PERMUTATION_H
