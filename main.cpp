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
#include "v3/nn/AutoEncoder.h"
#include "v3/nn/layer/activation/SigmoidOutputLayer.h"

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
    int trainSetSize = 1000;

    NeuralNetwork nn;

    BatchNormLayer batchNormLayer(28 * 28);
    nn.addLayer(&batchNormLayer);

    ConvLayer convLayer(28, 28, 1, 5, 5, 6, 1, 1, 2, 2);
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

    MNISTDataSet trainSet("/home/wjy50/mnist/train-images.idx3-ubyte", "/home/wjy50/mnist/train-labels.idx1-ubyte");
    MNISTDataSet testSet("/home/wjy50/mnist/t10k-images.idx3-ubyte", "/home/wjy50/mnist/t10k-labels.idx1-ubyte");

    int noImprovementOccurredFor = 0;
    int minError = 0x7fffffff;
    FloatType in[28 * 28];
    FloatType label[10];
    for (int k = 0; k < 200; ++k) {
        long st = clock();
        nn.optimize(trainSet, trainSetSize);
        int fail = 0;
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

    BatchNormLayer batchNormLayer(32 * 32 * 3);
    nn.addLayer(&batchNormLayer);

    ConvLayer convLayer(32, 32, 3, 5, 5, 6, 1, 1, 0, 0, true);
    nn.addLayer(&convLayer);

    BatchNormLayer batchNormLayer1(convLayer.getOutputDim());
    nn.addLayer(&batchNormLayer1);

    LReLULayer lReLULayer(convLayer.getOutputDim(), 0.01);
    nn.addLayer(&lReLULayer);

    MaxPoolingLayer poolingLayer(convLayer.getOutputWidth(), convLayer.getOutputHeight(), convLayer.getKernelCount(), 2, 2, 2, 2);
    nn.addLayer(&poolingLayer);

    ConvLayer convLayer1(poolingLayer.getOutputWidth(), poolingLayer.getOutputHeight(), poolingLayer.getChannelCount(), 3, 3, 16, 1, 1, 0, 0, true);
    nn.addLayer(&convLayer1);

    BatchNormLayer batchNormLayer2(convLayer1.getOutputDim());
    nn.addLayer(&batchNormLayer2);

    LReLULayer lReLULayer1(convLayer1.getOutputDim(), 0.01);
    nn.addLayer(&lReLULayer1);

    MeanPoolingLayer poolingLayer1(convLayer1.getOutputWidth(), convLayer1.getOutputHeight(), convLayer1.getKernelCount(), 2, 2, 2, 2);
    nn.addLayer(&poolingLayer1);

    LinearLayer layer(poolingLayer1.getOutputDim(), 150, true);
    nn.addLayer(&layer);

    BatchNormLayer batchNormLayer3(layer.getOutputDim());
    nn.addLayer(&batchNormLayer3);

    LReLULayer lReLULayer2(layer.getOutputDim(), 0.01);
    nn.addLayer(&lReLULayer2);

    /*DropoutLayer dropoutLayer(lReLULayer2.getOutputDim());
    dropoutLayer.setDropoutFraction(0.5);
    nn.addLayer(&dropoutLayer);*/

    LinearLayer layer1(lReLULayer2.getOutputDim(), 10);
    nn.addLayer(&layer1);

    SoftMaxOutputLayer output(layer1.getOutputDim());
    nn.addLayer(&output);

    nn.buildUpNetwork(20);

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
    AdamOptimizer optimizer7;
    batchNormLayer2.setOptimizer(&optimizer7);
    AdamOptimizer optimizer8;
    batchNormLayer3.setOptimizer(&optimizer8);

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

    int noImprovementOccurredFor = 0;
    int minError = 0x7fffffff;
    FloatType in[32 * 32 * 3];
    FloatType label[10];
    for (int k = 0; k < 200; ++k) {
        long st = clock();
        nn.optimize(trainSet1, trainSetSize);
        int fail = 0;
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

void testAutoEncoder()
{
    AutoEncoder autoEncoder;

    int trainSetSize = 5000;

    ConvLayer encoder(28, 28, 1, 3, 3, 2);
    autoEncoder.addLayer(&encoder);

    LReLULayer lReLULayer(encoder.getOutputDim(), 0.01);
    autoEncoder.addLayer(&lReLULayer);

    ConvLayer decoder(26, 26, 2, 3, 3, 1, 1, 1, 2, 2);
    autoEncoder.addLayer(&decoder);

    SigmoidOutputLayer output(decoder.getOutputDim());
    autoEncoder.addLayer(&output);

    autoEncoder.buildUpAutoEncoder(20);

    AdamOptimizer optimizer1;
    encoder.setOptimizer(&optimizer1);
    AdamOptimizer optimizer2;
    decoder.setOptimizer(&optimizer2);

    MNISTDataSet trainSet("/home/wjy50/mnist/train-images.idx3-ubyte", "/home/wjy50/mnist/train-labels.idx1-ubyte");
    MNISTDataSet testSet("/home/wjy50/mnist/t10k-images.idx3-ubyte", "/home/wjy50/mnist/t10k-labels.idx1-ubyte");

    FloatType in[28 * 28];
    for (int k = 201; k < 10000; ++k) {
        autoEncoder.optimize(trainSet, trainSetSize);
        cout << "epoch" << k+1 << ':' << endl;
        int j = 50000 + k;
        trainSet.getBatch(in, nullptr, &j, 1);
        const FloatType *o = autoEncoder.feedForward(in);
        for (int i = 0; i < 28; ++i) {
            for (int l = 0; l < 28; ++l) {
                auto r = static_cast<int>(o[i * 28 + l] * 9);
                if (r > 0) cout << r << ' ';
                else cout << "  ";
            }
            cout << "    ";
            for (int l = 0; l < 28; ++l) {
                auto r = static_cast<int>(in[i * 28 + l] * 9);
                if (r > 0) cout << r << ' ';
                else cout << "  ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

int main()
{
    std::cout << "Hello, World!" << std::endl;
#if ENABLE_CUDA
    initializeCUDA();
#endif

    testAutoEncoder();

    //newNNCIFAR10();

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