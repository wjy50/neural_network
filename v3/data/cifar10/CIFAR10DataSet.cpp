/**
 * Created by wjy50 on 18-5-18.
 */

#include <fstream>
#include "CIFAR10DataSet.h"
#include "../../interface/interface.h"

CIFAR10DataSet::CIFAR10DataSet(const char **path, int n)
{
    count = 10000 * n;
    buffer = make_unique_array<unsigned char[]>(3073 * static_cast<size_t>(count));
    for (int i = 0; i < n; ++i) {
        std::ifstream file(path[i], std::ios::binary);
        file.read(reinterpret_cast<char *>(buffer.get() + i * 3073 * 10000), 3073 * 10000);
        file.close();
    }
}

void CIFAR10DataSet::getBatch(FloatType *data, FloatType *labels, const int *indices, int count)
{
    getCIFAR10Batch(data, labels, buffer.get(), indices, count);
}

int CIFAR10DataSet::getCount()
{
    return count;
}