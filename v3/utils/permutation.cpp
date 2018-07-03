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
    std::uniform_real_distribution<float> distribution(0, 1);
    for (int i = n - 1; i > 0; --i) {
        auto d = static_cast<int>(distribution(rd) * (i + 1));
        if (i != d) {
            int tmp = arr[d];
            arr[d] = arr[i];
            arr[i] = tmp;
        }
    }
}

#endif