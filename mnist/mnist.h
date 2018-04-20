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

    double *get(int i);

    ~MNISTLabel();
};

#endif //NEURAL_NETWORK_MNIST_H
