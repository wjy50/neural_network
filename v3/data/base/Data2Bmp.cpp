/**
 * Created by wjy50 on 18-6-7.
 */

#include <cassert>
#include "Data2Bmp.h"

Data2Bmp::Data2Bmp(const char *path, int w, int h, int bitsPerPixel)
{
    assert(bitsPerPixel % 8 == 0 && bitsPerPixel <= 32);
    this->w = w;
    this->h = h;
    this->bitsPerPixel = bitsPerPixel;
    stream = std::ofstream(path);
    bmpSize = w * h * bitsPerPixel / 8;
    dataOffset = 14 + 40;
    size = dataOffset + bmpSize;
    writeBmpHeader();
}

void Data2Bmp::writeBmpHeader()
{
    int p = 0;
    stream.write(reinterpret_cast<const char *>(&magic), sizeof(short));
    stream.write(reinterpret_cast<const char *>(&size), sizeof(int));
    stream.write(reinterpret_cast<const char *>(&p), sizeof(int));
    stream.write(reinterpret_cast<const char *>(&dataOffset), sizeof(int));
    int bmpInfoSize = 40;
    stream.write(reinterpret_cast<const char *>(&bmpInfoSize), sizeof(int));
    stream.write(reinterpret_cast<const char *>(&w), sizeof(int));
    int nh = -h;
    stream.write(reinterpret_cast<const char *>(&nh), sizeof(int));
    int o = 1;
    stream.write(reinterpret_cast<const char *>(&o), sizeof(short));
    stream.write(reinterpret_cast<const char *>(&bitsPerPixel), sizeof(short));
    stream.write(reinterpret_cast<const char *>(&p), sizeof(int));
    stream.write(reinterpret_cast<const char *>(&p), sizeof(int));
    stream.write(reinterpret_cast<const char *>(&p), sizeof(int));
    stream.write(reinterpret_cast<const char *>(&p), sizeof(int));
    stream.write(reinterpret_cast<const char *>(&p), sizeof(int));
    stream.write(reinterpret_cast<const char *>(&p), sizeof(int));
}

void Data2Bmp::close()
{
    stream.close();
}