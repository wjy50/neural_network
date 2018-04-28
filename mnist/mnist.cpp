/**
 * Created by wjy50 on 18-4-18.
 */

#include <cstdio>
#include <random>
#include <cstring>
#include "mnist.h"

void invertEndian(void *p, size_t size)
{
    auto *c = static_cast<char *>(p);
    for (int i = 0; i < size >> 1; ++i) {
        char tmp = c[i];
        c[i] = c[size-i-1];
        c[size-i-1] = tmp;
    }
}

MNISTDataSet::MNISTDataSet(const char *imagePath, const char *labelPath)
{
    tx = ty = 0;
    FILE *file = fopen(imagePath, "rb");
    if (file) {
        fseek(file, 0, SEEK_END);
        imageSize = static_cast<size_t>(ftell(file));
        fseek(file, 0, SEEK_SET);
        unsigned int magic;
        fread(&magic, sizeof(int), 1, file);
        if (magic == 0x03080000) {
            count = 0;
            fread(&count, sizeof(int), 1, file);
            invertEndian(&count, sizeof(int));
            fread(&height, sizeof(int), 1, file);
            invertEndian(&height, sizeof(int));
            fread(&width, sizeof(int), 1, file);
            invertEndian(&width, sizeof(int));
            imageOffset = static_cast<size_t>(ftell(file));
            imageBuffer = new unsigned char[imageSize - imageOffset];
            fread(imageBuffer, sizeof(char), imageSize-imageOffset, file);
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
        labelSize = static_cast<size_t>(ftell(file));
        fseek(file, 0, SEEK_SET);
        unsigned int magic;
        fread(&magic, sizeof(int), 1, file);
        if (magic == 0x01080000) {
            count = 0;
            fread(&count, sizeof(int), 1, file);
            invertEndian(&count, sizeof(int));
            labelOffset = static_cast<size_t>(ftell(file));
            labelBuffer = new unsigned char[labelSize - labelOffset];
            fread(labelBuffer, sizeof(char), static_cast<size_t>(labelSize - labelOffset), file);
        } else {
            labelBuffer = nullptr;
        }
        fclose(file);
    }
}

void MNISTDataSet::setTranslation(int x, int y)
{
    tx = x;
    ty = y;
}

const double* MNISTDataSet::getData(size_t i)
{
    const unsigned char *r = imageBuffer+width*height*i;
    if (tx != 0 || ty != 0) {
        memset(image, 0, sizeof(double)*width*height);
        int dy = std::max(0, ty);
        int dx = std::max(0, tx);
        int yi = 0;
        for (int j = dy; j < height-dy; ++j, ++yi) {
            int xi = 0;
            for (int k = dx; k < width-dx; ++k) {
                image[yi*width+xi++] = (double)r[j*width+k] / 0xff;
            }
        }
    } else {
        for (int j = 0; j < width * height; ++j) {
            image[j] = (double)r[j] / 0xff;
        }
    }
    return image;
}

const double* MNISTDataSet::getLabel(size_t i)
{
    unsigned char c = labelBuffer[i];
    for (int j = 0; j < 10; ++j) {
        label[j] = c == j ? 1 : 0;
    }
    return label;
}

size_t MNISTDataSet::getSize()
{
    return count;
}

MNISTDataSet::~MNISTDataSet()
{
    delete[] imageBuffer;
    delete[] labelBuffer;
}