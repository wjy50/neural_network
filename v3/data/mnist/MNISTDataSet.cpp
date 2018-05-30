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