/**
 * Created by wjy50 on 18-4-18.
 */

#include <cstdio>
#include <random>
#include <cstring>
#include "mnist.h"

void toBE(void *p, size_t size)
{
    auto *c = static_cast<char *>(p);
    for (int i = 0; i < size >> 1; ++i) {
        char tmp = c[i];
        c[i] = c[size-i-1];
        c[size-i-1] = tmp;
    }
}

MNISTImage::MNISTImage(const char *path)
{
    tx = ty = 0;
    FILE *file = fopen(path, "rb");
    if (file) {
        fseek(file, 0, SEEK_END);
        size = static_cast<size_t>(ftell(file));
        fseek(file, 0, SEEK_SET);
        unsigned int magic;
        fread(&magic, sizeof(int), 1, file);
        if (magic == 0x03080000) {
            count = 0;
            fread(&count, sizeof(int), 1, file);
            toBE(&count, sizeof(int));
            fread(&height, sizeof(int), 1, file);
            toBE(&height, sizeof(int));
            fread(&width, sizeof(int), 1, file);
            toBE(&width, sizeof(int));
            offset = static_cast<size_t>(ftell(file));
            buffer = new unsigned char[size - offset];
            image = new double[width*height];
            fread(buffer, sizeof(char), size-offset, file);
        } else {
            buffer = nullptr;
            image = nullptr;
        }
        fclose(file);
    } else {
        buffer = nullptr;
        image = nullptr;
    }
}

void MNISTImage::setTranslation(int x, int y)
{
    tx = x;
    ty = y;
}

double* MNISTImage::get(size_t i)
{
    const unsigned char *r = buffer+width*height*i;
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

size_t MNISTImage::getSize()
{
    return count;
}

MNISTImage::~MNISTImage()
{
    delete[] buffer;
    delete[] image;
}

MNISTLabel::MNISTLabel(const char *path)
{
    FILE *file = fopen(path, "rb");
    if (file) {
        fseek(file, 0, SEEK_END);
        size = static_cast<size_t>(ftell(file));
        fseek(file, 0, SEEK_SET);
        unsigned int magic;
        fread(&magic, sizeof(int), 1, file);
        if (magic == 0x01080000) {
            count = 0;
            fread(&count, sizeof(int), 1, file);
            toBE(&count, sizeof(int));
            offset = static_cast<size_t>(ftell(file));
            buffer = new unsigned char[size - offset];
            fread(buffer, sizeof(char), static_cast<size_t>(size - offset), file);
        }
        fclose(file);
    }
}

double* MNISTLabel::get(size_t i)
{
    unsigned char c = buffer[i];
    for (int j = 0; j < 10; ++j) {
        y[j] = c == j ? 1 : 0;
    }
    return y;
}

size_t MNISTLabel::getSize()
{
    return count;
}

MNISTLabel::~MNISTLabel()
{
    delete[] buffer;
}