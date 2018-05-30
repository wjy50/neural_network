/**
 * Created by wjy50 on 18-4-18.
 */

#include <cstdio>
#include <random>
#include <cstring>
#include <cassert>
#include "mnist.h"

void invertEndian(void *p, int size)
{
    auto *c = static_cast<char *>(p);
    for (int i = 0; i < size / 2; ++i) {
        char tmp = c[i];
        c[i] = c[size - i - 1];
        c[size - i - 1] = tmp;
    }
}

MNISTDataSet::MNISTDataSet(const char *imagePath, const char *labelPath)
{
    FILE *file = fopen(imagePath, "rb");
    if (file) {
        fseek(file, 0, SEEK_END);
        imageSize = static_cast<int>(ftell(file));
        fseek(file, 0, SEEK_SET);
        unsigned int magic;
        fread(&magic, sizeof(int), 1, file);
        if (magic == 0x03080000) {
            count = 0;
            fread(&count, sizeof(int), 1, file);
            invertEndian(&count, sizeof(int));
            /*fread(&height, sizeof(int), 1, file);
            invertEndian(&height, sizeof(int));
            fread(&width, sizeof(int), 1, file);
            invertEndian(&width, sizeof(int));*/
            fseek(file, 2 * sizeof(int), SEEK_CUR);
            imageOffset = static_cast<int>(ftell(file));
            imageBuffer = new unsigned char[imageSize - imageOffset];
            fread(imageBuffer, sizeof(char), static_cast<size_t>(imageSize - imageOffset), file);
        } else {
            imageBuffer = nullptr;
        }
        fclose(file);
    } else {
        imageBuffer = nullptr;
    }

    file = fopen(labelPath, "rb");
    if (file) {
        fseek(file, 0, SEEK_END);
        labelSize = static_cast<int>(ftell(file));
        fseek(file, 0, SEEK_SET);
        unsigned int magic;
        fread(&magic, sizeof(int), 1, file);
        if (magic == 0x01080000) {
            count = 0;
            fread(&count, sizeof(int), 1, file);
            invertEndian(&count, sizeof(int));
            labelOffset = static_cast<int>(ftell(file));
            labelBuffer = new unsigned char[labelSize - labelOffset];
            fread(labelBuffer, sizeof(char), static_cast<size_t>(labelSize - labelOffset), file);
        } else {
            labelBuffer = nullptr;
        }
        fclose(file);
    }
}

void MNISTDataSet::getBatch(FloatType *data, FloatType *label, const int *indices, int n)
{
    for (int i = 0; i < n; ++i) {
        int index = indices[i];
        const unsigned char *r = imageBuffer + 28 * 28 * index;
        for (int j = 0; j < 28 * 28; ++j) {
            data[i * 28 * 28 + j] = static_cast<FloatType>(r[j]) / 0xff;
        }
        if (normalizer) normalizer->normalize(data + i * 28 * 28);
    }
    if (label) {
        memset(label, 0, 10 * n * sizeof(FloatType));
        for (int i = 0; i < n; ++i) {
            label[10 * i + labelBuffer[indices[i]]] = 1;
        }
    }
}

int MNISTDataSet::getSize()
{
    return count;
}

MNISTDataSet::~MNISTDataSet()
{
    delete[] imageBuffer;
    delete[] labelBuffer;
}

MNISTNormalizer::MNISTNormalizer()
{
    memset(avg, 0, 28 * 28 * sizeof(FloatType));
    memset(dev, 0, 28 * 28 * sizeof(FloatType));
    confirmed = false;
    finished = false;
    sampleCount = 0;
    sampleCount1 = 0;
}

void MNISTNormalizer::add(MNISTDataSet &dataSet, int lim)
{
    assert(!confirmed && !finished);
    FloatType x[28 * 28];
    int count = lim ? lim : dataSet.getSize();
    sampleCount += count;
    for (int i = 0; i < count; ++i) {
        dataSet.getBatch(x, nullptr, &i, 1);
        for (int j = 0; j < 28 * 28; ++j) {
            avg[j] += x[j];
        }
    }
}

void MNISTNormalizer::confirm()
{
    assert(!confirmed && !finished);
    confirmed = true;
    for (FloatType &a : avg) {
        a /= sampleCount;
    }
}

void MNISTNormalizer::div(MNISTDataSet &dataSet, int lim)
{
    assert(confirmed && !finished);
    FloatType x[28 * 28];
    int count = lim ? lim : dataSet.getSize();
    sampleCount1 += count;
    for (int i = 0; i < count; ++i) {
        dataSet.getBatch(x, nullptr, &i, 1);
        for (int j = 0; j < 28 * 28; ++j) {
            FloatType d = x[j] - avg[j];
            dev[j] += d * d;
        }
    }
}

void MNISTNormalizer::finish()
{
    assert(confirmed && !finished && sampleCount == sampleCount1);
    finished = true;
    for (FloatType &d : dev) {
        d = d == 0 ? 1 : std::sqrt(d / sampleCount);
    }
}

void MNISTNormalizer::normalize(FloatType *x)
{
    assert(confirmed && finished);
    for (int i = 0; i < 28 * 28; ++i) {
        x[i] = (x[i] - avg[i]) / dev[i];
    }
}