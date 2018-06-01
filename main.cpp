#include <iostream>
#include <memory>
#include <random>
#include <cublas.h>
#include <cstring>
#include "v3/nn/NeuralNetwork.h"
#include "v3/nn/layer/ConvLayer.h"
#include "v3/nn/layer/activation/LReLULayer.h"
#include "v3/nn/layer/MaxPoolingLayer.h"
#include "v3/nn/layer/MeanPoolingLayer.h"
#include "v3/nn/layer/LinearLayer.h"
#include "v3/nn/layer/activation/SoftMaxOutputLayer.h"
#include "v3/nn/optimizer/AdaMaxOptimizer.h"
#include "v3/data/mnist/MNISTDataSet.h"
#include "v3/interface/interface.h"
#include "v3/nn/optimizer/AdamOptimizer.h"
#include "v3/nn/layer/DropoutLayer.h"
#include "v3/data/cifar10/CIFAR10DataSet.h"
#include "v3/nn/layer/BatchNormLayer.h"

using namespace std;

bool test(const FloatType *a, const FloatType *y, int n)
{
    int maxOut = 0;
    for (int i = 1; i < n; ++i) {
        if (a[i] > a[maxOut]) maxOut = i;
    }
    return y[maxOut] == 1;
}

void newNNMNIST()
{
    int trainSetSize = 50000;

    NeuralNetwork nn;

    ConvLayer convLayer(28, 28, 1, 5, 5, 6, 1, 1, 2, 2, true);
    nn.addLayer(&convLayer);

    BatchNormLayer batchNormLayer(convLayer.getOutputDim());
    nn.addLayer(&batchNormLayer);

    LReLULayer lReLULayer(convLayer.getOutputDim(), 0.01);
    nn.addLayer(&lReLULayer);

    MaxPoolingLayer poolingLayer(convLayer.getOutputWidth(), convLayer.getOutputHeight(), convLayer.getKernelCount(), 2, 2, 2, 2);
    nn.addLayer(&poolingLayer);

    ConvLayer convLayer1(poolingLayer.getOutputWidth(), poolingLayer.getOutputHeight(), poolingLayer.getChannelCount(), 3, 3, 16, 1, 1, 0, 0);
    nn.addLayer(&convLayer1);

    BatchNormLayer batchNormLayer1(convLayer1.getOutputDim());
    nn.addLayer(&batchNormLayer1);

    LReLULayer lReLULayer1(convLayer1.getOutputDim(), 0.01);
    nn.addLayer(&lReLULayer1);

    MeanPoolingLayer poolingLayer1(convLayer1.getOutputWidth(), convLayer1.getOutputHeight(), convLayer1.getKernelCount(), 2, 2, 2, 2);
    nn.addLayer(&poolingLayer1);

    LinearLayer layer(poolingLayer1.getOutputDim(), 150);
    nn.addLayer(&layer);

    LReLULayer lReLULayer2(layer.getOutputDim(), 0.01);
    nn.addLayer(&lReLULayer2);

    DropoutLayer dropoutLayer(lReLULayer2.getOutputDim());
    dropoutLayer.setDropoutFraction(0.5);
    nn.addLayer(&dropoutLayer);

    LinearLayer layer1(dropoutLayer.getOutputDim(), 10);
    nn.addLayer(&layer1);

    SoftMaxOutputLayer output(layer1.getOutputDim());
    nn.addLayer(&output);

    nn.buildUpNetwork(10);

    AdamOptimizer optimizer1;
    convLayer.setOptimizer(&optimizer1);
    AdamOptimizer optimizer2;
    convLayer1.setOptimizer(&optimizer2);
    AdamOptimizer optimizer3;
    layer.setOptimizer(&optimizer3);
    AdamOptimizer optimizer4;
    layer1.setOptimizer(&optimizer4);
    AdamOptimizer optimizer5;
    batchNormLayer.setOptimizer(&optimizer5);
    AdamOptimizer optimizer6;
    batchNormLayer1.setOptimizer(&optimizer6);

    MNISTDataSet trainSet("/home/wjy50/mnist/train-images.idx3-ubyte", "/home/wjy50/mnist/train-labels.idx1-ubyte");
    MNISTDataSet testSet("/home/wjy50/mnist/t10k-images.idx3-ubyte", "/home/wjy50/mnist/t10k-labels.idx1-ubyte");

    /*MNISTNormalizer normalizer;

    normalizer.add(trainSet, trainSetSize);
    normalizer.confirm();
    normalizer.div(trainSet, trainSetSize);
    normalizer.finish();

    trainSet.setNormalizer(&normalizer);
    testSet.setNormalizer(&normalizer);*/

    int noImprovementOccurredFor = 0;
    int minError = 0x7fffffff;
    FloatType in[28 * 28];
    FloatType label[10];
    for (int k = 0; k < 200; ++k) {
        long st = clock();
        //layer.setDropoutFraction(0.5);
        nn.optimize(trainSet, trainSetSize);
        int fail = 0;
        //layer.setDropoutFraction(0);
        for (int j = 50000; j < 60000; ++j) {
            trainSet.getBatch(in, label, &j, 1);
            const FloatType *o = nn.feedForward(in);
            if (!test(o, label, 10)) {
                fail++;
            }
        }
        cout << "epoch" << k+1 << ':' << fail;
        if (fail < minError) {
            minError = fail;
            noImprovementOccurredFor = 0;
            cout << '*';
        } else noImprovementOccurredFor++;
        cout << endl << "time:" << clock() - st << endl;
        if (noImprovementOccurredFor > 30) break;
    }
    int fail = 0;
    //image.setTranslation(1, 1);
    for (int j = 0; j < 10000; ++j) {
        testSet.getBatch(in, label, &j, 1);
        const FloatType *o = nn.feedForward(in);
        if (!test(o, label, 10)) fail++;
    }
    cout << fail << endl;
}

void newNNCIFAR10()
{
    NeuralNetwork nn;

    int trainSetSize = 50000;

    ConvLayer convLayer(32, 32, 3, 5, 5, 6, 1, 1, 0, 0);
    nn.addLayer(&convLayer);

    LReLULayer lReLULayer(convLayer.getOutputDim(), 0.01);
    nn.addLayer(&lReLULayer);

    MaxPoolingLayer poolingLayer(convLayer.getOutputWidth(), convLayer.getOutputHeight(), convLayer.getKernelCount(), 2, 2, 2, 2);
    nn.addLayer(&poolingLayer);

    ConvLayer convLayer1(poolingLayer.getOutputWidth(), poolingLayer.getOutputHeight(), poolingLayer.getChannelCount(), 3, 3, 16, 1, 1, 0, 0);
    nn.addLayer(&convLayer1);

    LReLULayer lReLULayer1(convLayer1.getOutputDim(), 0.01);
    nn.addLayer(&lReLULayer1);

    MeanPoolingLayer poolingLayer1(convLayer1.getOutputWidth(), convLayer1.getOutputHeight(), convLayer1.getKernelCount(), 2, 2, 2, 2);
    nn.addLayer(&poolingLayer1);

    LinearLayer layer(poolingLayer1.getOutputDim(), 150);
    nn.addLayer(&layer);

    LReLULayer lReLULayer2(layer.getOutputDim(), 0.01);
    nn.addLayer(&lReLULayer2);

    DropoutLayer dropoutLayer(lReLULayer2.getOutputDim());
    dropoutLayer.setDropoutFraction(0.5);
    nn.addLayer(&dropoutLayer);

    LinearLayer layer1(dropoutLayer.getOutputDim(), 10);
    nn.addLayer(&layer1);

    SoftMaxOutputLayer output(layer1.getOutputDim());
    nn.addLayer(&output);

    nn.buildUpNetwork(20);

    /*FloatType learningRate = 0.09;
    SimpleSGDOptimizer optimizer1(learningRate, 5, trainSetSize, convLayer.getWeightCount(), convLayer.getBiasCount());
    convLayer.setOptimizer(&optimizer1);
    SimpleSGDOptimizer optimizer2(learningRate, 5, trainSetSize, convLayer1.getWeightCount(), convLayer1.getBiasCount());
    convLayer1.setOptimizer(&optimizer2);
    SimpleSGDOptimizer optimizer3(learningRate, 5, trainSetSize, layer.getWeightCount(), layer.getBiasCount());
    layer.setOptimizer(&optimizer3);
    SimpleSGDOptimizer optimizer4(learningRate, 5, trainSetSize, output.getWeightCount(), output.getBiasCount());
    output.setOptimizer(&optimizer4);*/

    AdamOptimizer optimizer1;
    convLayer.setOptimizer(&optimizer1);
    AdamOptimizer optimizer2;
    convLayer1.setOptimizer(&optimizer2);
    AdamOptimizer optimizer3;
    layer.setOptimizer(&optimizer3);
    AdamOptimizer optimizer4;
    layer1.setOptimizer(&optimizer4);

    const char *paths[6] = {
            "/home/wjy50/cifar/data_batch_1.bin",
            "/home/wjy50/cifar/data_batch_2.bin",
            "/home/wjy50/cifar/data_batch_3.bin",
            "/home/wjy50/cifar/data_batch_4.bin",
            "/home/wjy50/cifar/data_batch_5.bin",
            "/home/wjy50/cifar/test_batch.bin"
    };
    CIFAR10DataSet trainSet1(paths, 5);
    CIFAR10DataSet testSet(paths + 5, 1);

    /*CIFARNormalizer normalizer;

    normalizer.add(trainSet1);
    normalizer.confirm();

    normalizer.div(trainSet1);
    normalizer.finish();

    trainSet1.setNormalizer(&normalizer);
    testSet.setNormalizer(&normalizer);*/

    int noImprovementOccurredFor = 0;
    int minError = 0x7fffffff;
    FloatType in[32 * 32 * 3];
    FloatType label[10];
    for (int k = 0; k < 200; ++k) {
        long st = clock();
        //layer.setDropoutFraction(0.5);
        nn.optimize(trainSet1, trainSetSize);
        int fail = 0;
        //layer.setDropoutFraction(0);
        for (int j = 0; j < 10000; ++j) {
            testSet.getBatch(in, label, &j, 1);
            const FloatType *o = nn.feedForward(in);
            if (!test(o, label, 10)) {
                fail++;
            }
        }
        cout << "epoch" << k+1 << ':' << fail;
        if (fail < minError) {
            minError = fail;
            noImprovementOccurredFor = 0;
            cout << '*';
        } else noImprovementOccurredFor++;
        cout << endl << "time:" << clock() - st << endl;
        if (noImprovementOccurredFor > 30) break;
    }
}

template<typename T>
void printM(const T *m, int r, int c)
{
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            cout << m[i * c + j] << ' ';
        }
        cout << endl;
    }
    cout << endl;
}

int main()
{
    std::cout << "Hello, World!" << std::endl;
#if ENABLE_CUDA
    initializeCUDA();
#endif

    newNNMNIST();

    /*FloatType v[] = {
            1, 2, 3, 0,
            2, 3, 1, 4,
            3, 1, 2, 2,
            1, 5, 3, 0
    };
    FloatType k[] = {
            1, 2,
            2, 1,

            1
    };
    FloatType r[3 * 3];
    conv(v, 4, 4, 1, r, 3, 3, k, 2, 2, 1, 1, 1, 0, 0, 1);
    printM(r, 3, 3);
    convBP(r, 3, 3, v, 4, 4, 1, k, 2, 2, 1, 1, 1, 0, 0, 1);
    printM(v, 4, 4);
    convGradients(k, 2, 2, k + 4, 1, r, 3, 3, v, 4, 4, 1, 1, 1, 0, 0, 1);
    printM(k, 2, 2);
    printM(k + 4, 1, 1);*/

#if ENABLE_CUDA
    destroyCUDA();
#endif
    return 0;
}

//TODO Adam/AdaMax自动转SGD
//TODO batch normalization