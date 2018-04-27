/**
 * Created by wjy50 on 18-4-18.
 */

#ifndef NEURAL_NETWORK_MNIST_H
#define NEURAL_NETWORK_MNIST_H

#include "../data/DataSet.h"

void toBE(void *p, size_t size);

class MNISTImage : public DataSet
{
private:
    size_t offset, size;
    double *image;
    unsigned char *buffer;
    size_t count;
    int width, height;

    int tx, ty;
public:
    explicit MNISTImage(const char *path);

    /**
     * 设置图像平移量限制
     * 取数据时在x和y方向上平移
     * 用于人工拓展训练集
     * @param x
     * @param y
     */
    void setTranslation(int x, int y);

    /**
     * 获取第i张图片
     * @param i 序号
     * @return 第i张图片数据，大小为width * height的数组，不能在外部delete[]
     */
    double *get(size_t i) override;

    size_t getSize() override;

    ~MNISTImage();
};

class MNISTLabel : public DataSet
{
private:
    size_t offset, size;
    size_t count;
    unsigned char *buffer;
    double y[10];
public:
    explicit MNISTLabel(const char *path);

    /**
     * 获取第i个标签
     * 将n ∈ [0, 9]转换成十维向量，第n+1个分量为1，其余分量为0
     * @param i 序号
     * @return 标签向量，不能在外部delete[]
     */
    double *get(size_t i) override;

    size_t getSize() override;

    ~MNISTLabel();
};

#endif //NEURAL_NETWORK_MNIST_H
