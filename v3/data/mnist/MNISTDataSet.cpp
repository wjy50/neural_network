/**
 * Created by wjy50 on 18-5-18.
 */

#include "MNISTDataSet.h"
#include "../../interface/interface.h"

#include <fstream>
#include <cassert>

using namespace std;

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
    ifstream stream(imagePath, ios::binary);
    if (stream.good()) {
        unsigned int magic;
        stream.read(reinterpret_cast<char *>(&magic), sizeof(unsigned int));
        if (magic == 0x03080000) {
            stream.read(reinterpret_cast<char *>(&count), sizeof(int));
            invertEndian(&count, sizeof(int));
            stream.seekg(2 * sizeof(int), ios::cur);
            buffer = make_unique_array<unsigned char[]>(static_cast<size_t>(count) * 28 * 28);
            stream.read(reinterpret_cast<char *>(buffer.get()), count * 28 * 28);
        }
        stream.close();
    }

    ifstream labelStream(labelPath, ios::binary);
    if (labelStream.good()) {
        unsigned int magic;
        labelStream.read(reinterpret_cast<char *>(&magic), sizeof(unsigned int));
        if (magic == 0x01080000) {
            int labelCount;
            labelStream.read(reinterpret_cast<char *>(&labelCount), sizeof(int));
            invertEndian(&labelCount, sizeof(int));
            assert(count == labelCount);
            labelBuffer = make_unique_array<unsigned char[]>(static_cast<size_t>(count));
            labelStream.read(reinterpret_cast<char *>(labelBuffer.get()), count);
        }
        labelStream.close();
    }
}

void MNISTDataSet::getBatch(FloatType *data, FloatType *labels, const int *indices, int count)
{
    getMNISTBatch(data, labels, buffer.get(), labelBuffer.get(), indices, count);
}

int MNISTDataSet::getCount()
{
    return count;
}

MNISTData2Bmp::MNISTData2Bmp(const char *path, int w, int h, bool invert) : Data2Bmp(path, w, h, 24)
{
    this->invert = invert;
}

void MNISTData2Bmp::writeData(const FloatType *data)
{
    unique_ptr<unsigned char[]> buffer = make_unique_array<unsigned char[]>(static_cast<size_t>(w) * static_cast<size_t>(h) * 3);
    if (invert) {
        for (int i = 0; i < w * h; ++i) {
            auto c = static_cast<unsigned char>(data[i] > 1e-3 ? ((static_cast<unsigned int>(data[i] * 0xff) & 0xffu) ^ 0xffu) : 0xffu);
            for (int j = 0; j < 3; ++j) {
                buffer[i * 3 + j] = c;
            }
        }
    } else {
        for (int i = 0; i < w * h; ++i) {
            auto c = static_cast<unsigned char>(data[i] > 1e-3 ? (static_cast<unsigned int>(data[i] * 0xff) & 0xffu) : 0);
            for (int j = 0; j < 3; ++j) {
                buffer[i * 3 + j] = c;
            }
        }
    }
    stream.write(reinterpret_cast<const char *>(buffer.get()), w * h * 3);
}