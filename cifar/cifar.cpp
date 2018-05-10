/**
 * Created by wjy50 on 18-4-27.
 */

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <fstream>
#include "cifar.h"

CIFARDataSet::CIFARDataSet(const char **path, int n)
{
    count = 10000 * n;
    size = 3073 * count;
    buffer = new unsigned char[size];
    for (int i = 0; i < n; ++i) {
        std::ifstream file(path[i], std::ios::binary);
        file.read((char *)(buffer + i * 3073 * 10000), 3073 * 10000);
        file.close();
    }
}

int CIFARDataSet::getSize()
{
    return count;
}

void CIFARDataSet::getBatch(FloatType *data, FloatType *label, const int *indices, int n)
{
    for (int i = 0; i < n; ++i) {
        int index = indices[i];
        const unsigned char *b = buffer + index * 3073 + 1;
        for (int j = 0; j < 3072; ++j) {
            data[3072 * i + j] = (FloatType) b[j] / 0xff;
        }
        if (normalizer) normalizer->normalize(data + 3072 * i);
    }
    if (label) {
        memset(label, 0, n * 10 * sizeof(FloatType));
        for (int i = 0; i < n; ++i) {
            int index = indices[i];
            label[10 * i + buffer[index * 3073]] = 1;
        }
    }
}

CIFARNormalizer::CIFARNormalizer()
{
    memset(avg, 0, 32 * 32 * 3 * sizeof(FloatType));
    memset(dev, 0, 32 * 32 * 3 * sizeof(FloatType));
    confirmed = false;
    finished = false;
    sampleCount = 0;
    sampleCount1 = 0;
}

void CIFARNormalizer::add(CIFARDataSet &dataSet)
{
    assert(!confirmed && !finished);
    FloatType x[32 * 32 * 3];
    int count = dataSet.getSize();
    sampleCount += count;
    for (int i = 0; i < count; ++i) {
        dataSet.getBatch(x, nullptr, &i, 1);
        for (int j = 0; j < 32 * 32 * 3; ++j) {
            avg[j] += x[j];
        }
    }
}

void CIFARNormalizer::confirm()
{
    assert(!confirmed && !finished);
    confirmed = true;
    for (FloatType &a : avg) {
        a /= sampleCount;
    }
}

void CIFARNormalizer::div(CIFARDataSet &dataSet)
{
    assert(confirmed && !finished);
    FloatType x[32 * 32 * 3];
    int count = dataSet.getSize();
    sampleCount1 += count;
    for (int i = 0; i < count; ++i) {
        dataSet.getBatch(x, nullptr, &i, 1);
        for (int j = 0; j < 32 * 32 * 3; ++j) {
            FloatType d = x[j] - avg[j];
            dev[j] += d * d;
        }
    }
}

void CIFARNormalizer::finish()
{
    assert(confirmed && !finished && sampleCount == sampleCount1);
    finished = true;
    for (FloatType &d : dev) {
        d = std::sqrt(d / sampleCount);
    }
}

void CIFARNormalizer::normalize(FloatType *x)
{
    assert(confirmed && finished);
    for (int i = 0; i < 32 * 32 * 3; ++i) {
        x[i] = (x[i] - avg[i]) / dev[i];
    }
}