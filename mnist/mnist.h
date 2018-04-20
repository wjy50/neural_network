/**
 * Created by wjy50 on 18-4-18.
 */

#ifndef NEURAL_NETWORK_MNIST_H
#define NEURAL_NETWORK_MNIST_H

void toBE(void *p, size_t size);

class MNISTImage
{
private:
    long offset, size;
    double *image;
    unsigned char *buffer;
    int count;
    int width, height;
public:
    explicit MNISTImage(const char *path);

    /**
     * 获取第i张图片
     * @param i 序号
     * @return 第i张图片数据，大小为width * height的数组，不能在外部delete[]
     */
    double *get(int i);

    ~MNISTImage();
};

class MNISTLabel
{
private:
    long offset, size;
    int count;
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
    double *get(int i);

    ~MNISTLabel();
};

#endif //NEURAL_NETWORK_MNIST_H
