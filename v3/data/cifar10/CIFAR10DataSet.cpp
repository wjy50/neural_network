/**
 * Created by wjy50 on 18-5-18.
 */

#include <fstream>
#include "CIFAR10DataSet.h"
#include "../../interface/interface.h"

CIFAR10DataSet::CIFAR10DataSet(const char **path, int n)
{
    count = 10000 * n;
    auto *temp = new unsigned char[3073 * count];
    for (int i = 0; i < n; ++i) {
        std::ifstream file(path[i], std::ios::binary);
        file.read(reinterpret_cast<char *>(temp + i * 3073 * 10000), 3073 * 10000);
        file.close();
    }
#if ENABLE_CUDA
    buffer = allocArray<unsigned char>(3073 * count);
    cudaMemcpy(buffer, temp, 3073 * count * sizeof(unsigned char), cudaMemcpyHostToDevice);
    delete[] temp;
#else
    buffer = temp;
#endif
}

void CIFAR10DataSet::getBatch(FloatType *data, FloatType *labels, const int *indices, int count)
{
    getCIFAR10Batch(data, labels, buffer, indices, count);
}

int CIFAR10DataSet::getCount()
{
    return count;
}

CIFAR10Data2Bmp::CIFAR10Data2Bmp(const char *path, int w, int h) : Data2Bmp(path, w, h, 24) {}

void CIFAR10Data2Bmp::writeData(const FloatType *data)
{
    std::unique_ptr<unsigned char[]> buffer = make_unique_array<unsigned char[]>(static_cast<size_t>(w) * static_cast<size_t>(h) * 3);
    for (int i = 0; i < 3; ++i) {
        const FloatType *curData = data + w * h * i;
        unsigned char *curBuffer = buffer.get() + i;
        for (int j = 0; j < w * h; ++j) {
            curBuffer[j * 3] = static_cast<unsigned char>(curData[j] >= 1e-3 ? (static_cast<unsigned int>(curData[j] * 0xff) & 0xffu) : 0);
        }
    }
    stream.write(reinterpret_cast<const char *>(buffer.get()), w * h * 3);
}

CIFAR10DataSet::~CIFAR10DataSet()
{
    freeArray(buffer);
}