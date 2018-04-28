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
void invertEndian(void *p, size_t size);

class MNISTDataSet : public DataSet
{
private:
    size_t imageOffset, imageSize;
    size_t labelOffset, labelSize;
    double image[28*28];
    double label[10];
    unsigned char *imageBuffer;
    unsigned char *labelBuffer;
    size_t count;
    int width, height;

    int tx, ty;
public:
    explicit MNISTDataSet(const char *imagePath, const char *labelPath);

    /**
     * 设置图像平移量限制
     * 取数据时在x和y方向上平移
     * 用于人工拓展训练集
     * @param x
     * @param y
     */
    void setTranslation(int x, int y);

    const double *getData(size_t i) override;

    const double *getLabel(size_t i) override;

    size_t getSize() override;

    ~MNISTDataSet();
};

#endif //NEURAL_NETWORK_MNIST_H
