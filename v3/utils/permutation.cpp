/**
 * Created by wjy50 on 18-6-10.
 */

#include "permutation.h"

#if !ENABLE_CUDA

void randomPermutation(int *arr, int n, int bias)
{
    for (int i = 0; i < n; ++i) {
        arr[i] = i + bias;
    }
    std::random_device rd;
    std::uniform_int_distribution<int> distribution(0, n);
    for (int i = n-1; i > 0; --i) {
        int d = distribution(rd) % i;
        int tmp = arr[d];
        arr[d] = arr[i];
        arr[i] = tmp;
    }
}

#endif