#include <iostream>
#include <memory>
#include "math/Matrix.h"
#include "mnist/mnist.h"
#include "nn/NeuralNetwork.h"
#include "nn/ffw/FeedForwardNN.h"
#include "nn/ffw/layer/FullyConnLayer.h"
#include "nn/ffw/layer/PoolingLayer.h"
#include "nn/ffw/layer/ConvLayer.h"

using namespace std;
using namespace ffw;

void legacyNN()
{
    int nums[] = {28*28, 100, 50, 10};
    ns_legacyNN::NeuralNetwork network(nums, 4, ns_legacyNN::L_RE_LU);
    network.initialize();
    //NeuralNetwork network("/home/wjy50/mnist/network4");
    network.setLearningRate(0.09);
    MNISTImage image("/home/wjy50/mnist/train-images.idx3-ubyte");
    MNISTLabel label("/home/wjy50/mnist/train-labels.idx1-ubyte");
    MNISTImage testImage("/home/wjy50/mnist/t10k-images.idx3-ubyte");
    MNISTLabel testLabel("/home/wjy50/mnist/t10k-labels.idx1-ubyte");
    int noImprovementOccurredFor = 0;
    int minError = 0x7fffffff;
    for (int k = 0; k < 90; ++k) {
        /*int p = offsets.get()[k % 9];
        int x = p % 3;
        int y = (p / 3) % 3;
        image.setTranslation(x-1, y-1);*/
        //image.setTranslation(1, 1);
        network.SGD(image, label, 50000, 20);
        int fail = 0;
        //image.setTranslation(0, 0);
        for (size_t j = 0; j < 10000; ++j) { /*训练集的后10000个数据作为验证集（validation_set）*/
            if (!network.test(image.get(50000+j), label.get(50000+j))) fail++;
        }
        cout << "epoch" << k+1 << ':' << fail << endl;
        if (fail < minError) {
            minError = fail;
            noImprovementOccurredFor = 0;
        } else noImprovementOccurredFor++;
        if (noImprovementOccurredFor > 15) break;
    }
    int fail = 0;
    //image.setTranslation(1, 1);
    for (size_t j = 0; j < 10000; ++j) {
        if (!network.test(image.get(50000+j), label.get(50000+j))) fail++;
    }
    cout << fail << endl;
    //network.save("/home/wjy50/mnist/network4");
}

bool test(const double *a, const double *y, int n)
{
    int maxOut = 0;
    for (int i = 1; i < n; ++i) {
        if (a[i] > a[maxOut]) maxOut = i;
    }
    return y[maxOut] == 1;
}

void newNN()
{
    FeedForwardNN nn;
    ConvLayer convLayer(28, 28, 1, 3, 3, 5, 1, 1, 1, 1, ffw::L_RE_LU);
    convLayer.setLearningRate(0.09);
    convLayer.setRegParam(5);
    nn.addLayer(&convLayer);
    /*FullyConnLayer layer(100, 28*28, ffw::L_RE_LU);
    layer.setLearningRate(0.09);
    layer.setRegParam(5);
    nn.addLayer(&layer);*/
    PoolingLayer poolingLayer(convLayer.getOutputWidth(), convLayer.getOutputHeight(), 2, 2, convLayer.getKernelCount(), MAX_POOLING);
    nn.addLayer(&poolingLayer);
    ConvLayer convLayer1(poolingLayer.getOutputWidth(), poolingLayer.getOutputHeight(), poolingLayer.getChannelCount(), 3, 3, 5, 1, 1, 0, 0, L_RE_LU);
    convLayer1.setRegParam(5);
    convLayer1.setLearningRate(0.09);
    nn.addLayer(&convLayer1);
    PoolingLayer poolingLayer1(convLayer1.getOutputWidth(), convLayer1.getOutputHeight(), 2, 2, convLayer1.getKernelCount(), MAX_POOLING);
    nn.addLayer(&poolingLayer1);
    FullyConnLayer layer(100, poolingLayer1.getNeuronCount(), ffw::L_RE_LU);
    layer.setLearningRate(0.09);
    layer.setRegParam(5);
    nn.addLayer(&layer);
    FullyConnLayer output(10, layer.getNeuronCount(), ffw::OUTPUT_ACTIVATOR);
    output.setLearningRate(0.09);
    output.setRegParam(5);
    nn.addLayer(&output);
    nn.buildUpNetwork();
    MNISTImage image("/home/wjy50/mnist/train-images.idx3-ubyte");
    MNISTLabel label("/home/wjy50/mnist/train-labels.idx1-ubyte");
    MNISTImage testImage("/home/wjy50/mnist/t10k-images.idx3-ubyte");
    MNISTLabel testLabel("/home/wjy50/mnist/t10k-labels.idx1-ubyte");
    int noImprovementOccurredFor = 0;
    int minError = 0x7fffffff;
    for (int k = 0; k < 90; ++k) {
        /*int p = offsets.get()[k % 9];
        int x = p % 3;
        int y = (p / 3) % 3;
        image.setTranslation(x-1, y-1);*/
        //image.setTranslation(0, 0);
        nn.SGD(image, label, 20, 50000);
        int fail = 0;
        //image.setTranslation(1, 1);
        for (size_t j = 0; j < 10000; ++j) { /*训练集的后10000个数据作为验证集（validation_set）*/
            const double *in = image.get(50000+j);
            const double *o = nn.feedForward(in);
            if (!test(o, label.get(50000+j), 10)) {
                /*for (int i = 0; i < 28; ++i) {
                    for (int l = 0; l < 28; ++l) {
                        cout << ((int)(in[i*28+l]*10) == 0 ? ' ' : '0') << ' ';
                    }
                    cout << endl;
                }
                cout << endl;

                for (int i = 0; i < 10; ++i) {
                    cout << (int)(o[i]*10) << ' ';
                }
                cout << endl;*/
                fail++;
            }
        }
        cout << "epoch" << k+1 << ':' << fail << endl;
        if (fail < minError) {
            minError = fail;
            noImprovementOccurredFor = 0;
        } else noImprovementOccurredFor++;
        if (noImprovementOccurredFor > 15) break;
    }
    int fail = 0;
    //image.setTranslation(1, 1);
    for (size_t j = 0; j < 10000; ++j) {
        const double *o = nn.feedForward(testImage.get(j));
        if (!test(o, testLabel.get(j), 10)) fail++;
    }
    cout << fail << endl;
}

int main()
{
    std::cout << "Hello, World!" << std::endl;
    newNN();
    return 0;
}