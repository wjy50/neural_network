/**
 * Created by wjy50 on 18-5-18.
 */

#ifndef NEURAL_NETWORK_MNISTDATASET_H
#define NEURAL_NETWORK_MNISTDATASET_H


#include "../base/DataSetBase.h"
#include "../../utils/UniquePointerExt.h"
#include "../base/Data2Bmp.h"

class MNISTData2Bmp : public Data2Bmp
{
private:
    bool invert;
public:
    explicit MNISTData2Bmp(const char *path, int w = 28, int h = 28, bool invert = true);

    void writeData(const FloatType *data) override;
};

class MNISTDataSet : public DataSetBase
{
private:
    int count;

    unsigned char *buffer;
    unsigned char *labelBuffer;
public:
    MNISTDataSet(const char *imagePath, const char *labelPath);

    int getCount() override;

    void getBatch(FloatType *data, FloatType *labels, const int *indices, int count) override;

    ~MNISTDataSet();
};


#endif //NEURAL_NETWORK_MNISTDATASET_H
