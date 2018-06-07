/**
 * Created by wjy50 on 18-6-7.
 */

#ifndef NEURAL_NETWORK_DATA2BMP_H
#define NEURAL_NETWORK_DATA2BMP_H

#include <fstream>
#include "../../def/type.h"

class Data2Bmp
{
private:
    const int magic = 0x4d42;

    int size, bmpSize;
    int dataOffset;
    int bitsPerPixel;

    void writeBmpHeader();

protected:
    int w, h;

    std::ofstream stream;
public:
    Data2Bmp(const char *path, int w, int h, int bitsPerPixel);

    virtual void writeData(const FloatType *data) = 0;

    void close();
};


#endif //NEURAL_NETWORK_DATA2BMP_H
