/**
 * Created by wjy50 on 18-4-18.
 */

#include <cstdio>
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
    FILE *file = fopen(path, "rb");
    if (file) {
        fseek(file, 0, SEEK_END);
        size = ftell(file);
        fseek(file, 0, SEEK_SET);
        unsigned int magic;
        fread(&magic, sizeof(int), 1, file);
        if (magic == 0x03080000) {
            fread(&count, sizeof(int), 1, file);
            toBE(&count, sizeof(int));
            fread(&height, sizeof(int), 1, file);
            toBE(&height, sizeof(int));
            fread(&width, sizeof(int), 1, file);
            toBE(&width, sizeof(int));
            offset = ftell(file);
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

double* MNISTImage::get(int i)
{
    const unsigned char *r = buffer+width*height*i;
    for (int j = 0; j < width * height; ++j) {
        image[j] = (double)r[j] / 0xff;
    }
    return image;
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
        size = ftell(file);
        fseek(file, 0, SEEK_SET);
        unsigned int magic;
        fread(&magic, sizeof(int), 1, file);
        if (magic == 0x01080000) {
            fread(&count, sizeof(int), 1, file);
            toBE(&count, sizeof(int));
            offset = ftell(file);
            buffer = new unsigned char[size - offset];
            fread(buffer, sizeof(char), size-offset, file);
        }
        fclose(file);
    }
}

double* MNISTLabel::get(int i)
{
    unsigned char c = buffer[i];
    for (int j = 0; j < 10; ++j) {
        y[j] = c == j ? 1 : 0;
    }
    return y;
}

MNISTLabel::~MNISTLabel()
{
    delete[] buffer;
}