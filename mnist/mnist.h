/**
 * Created by wjy50 on 18-4-18.
 */

#ifndef NEURAL_NETWORK_MNIST_H
#define NEURAL_NETWORK_MNIST_H

#include "../data/DataSet.h"

/**
 * 字节序反转
 * @param p 待反转数据首地址
 * @param size 待反转数据字节数
 */
void invertEndian(void *p, int size);

class MNISTDataSet : public DataSet
{
private:
    int imageOffset, imageSize;
    int labelOffset, labelSize;
    unsigned char *imageBuffer;
    unsigned char *labelBuffer;
    int count;
public:
    explicit MNISTDataSet(const char *imagePath, const char *labelPath);

    void getBatch(FloatType *data, FloatType *label, const int *indices, int n) override;

    int getSize() override;

    ~MNISTDataSet();
};

class MNISTNormalizer : public DataNormalizer
{
private:
    FloatType avg[28*28];
    FloatType dev[28*28];

    bool confirmed;
    bool finished;

    int sampleCount;
    int sampleCount1;
public:
    MNISTNormalizer();

    void add(MNISTDataSet &x, int lim = 0);

    void confirm();

    void div(MNISTDataSet &x, int lim = 0);

    void finish();

    void normalize(FloatType *x) override;
};

#endif //NEURAL_NETWORK_MNIST_H
