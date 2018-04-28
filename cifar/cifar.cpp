/**
 * Created by wjy50 on 18-4-27.
 */

#include <cstdio>
#include <cstring>
#include "cifar.h"

CIFARDataSet::CIFARDataSet(const char *path)
{
    FILE *file = fopen(path, "rb");
    if (file) {
        fseek(file, 0, SEEK_END);
        size = static_cast<size_t>(ftell(file));
        fseek(file, 0, SEEK_SET);
        buffer = new unsigned char[size];
        fread(buffer, sizeof(unsigned char), size, file);
        count = size / 3073;
        fclose(file);
    }
}

size_t CIFARDataSet::getSize()
{
    return count;
}

const double* CIFARDataSet::getData(size_t i)
{
    const unsigned char *b = buffer + i * 3073 + 1;
    for (int j = 0; j < 3072; ++j) {
        image[j] = (double)b[j] / 0xff;
    }
    return image;
}

const double* CIFARDataSet::getLabel(size_t i)
{
    memset(label, 0, 10 * sizeof(double));
    unsigned char c = buffer[i*3073];
    label[c] = 1;
    return label;
}