#include <iostream>
#include <memory>
#include <random>
#include "math/Matrix.h"
#include "mnist/mnist.h"
#include "nn/ffw/FeedForwardNN.h"
#include "nn/ffw/layer/FullyConnLayer.h"
#include "nn/ffw/layer/PoolingLayer.h"
#include "nn/ffw/layer/ConvLayer.h"
#include "cifar/cifar.h"
#include "nn/ffw/optimizer/SimpleSGDOptimizer.h"
#include "nn/ffw/optimizer/AdamOptimizer.h"
#include "nn/ffw/optimizer/AdaMaxOptimizer.h"
#include "openblas/f77blas.h"

using namespace std;
using namespace ffw;

//void legacyNN()
//{
//    int nums[] = {28*28, 100, 50, 10};
//    ns_legacyNN::NeuralNetwork network(nums, 4, ns_legacyNN::L_RE_LU);
//    network.initialize();
//    //NeuralNetwork network("/home/wjy50/mnist/network4");
//    network.setLearningRate(0.09);
//
//    MNISTDataSet trainSet("/home/wjy50/mnist/trainSet-images.idx3-ubyte", "/home/wjy50/mnist/trainSet-labels.idx1-ubyte");
//    MNISTDataSet testSet("/home/wjy50/mnist/t10k-images.idx3-ubyte", "/home/wjy50/mnist/t10k-labels.idx1-ubyte");
//
//    int noImprovementOccurredFor = 0;
//    int minError = 0x7fffffff;
//    for (int k = 0; k < 90; ++k) {
//        /*int p = offsets.get()[k % 9];
//        int x = p % 3;
//        int y = (p / 3) % 3;
//        image.setTranslation(x-1, y-1);*/
//        //image.setTranslation(1, 1);
//        network.SGD(image, label, 50000, 20);
//        int fail = 0;
//        //image.setTranslation(0, 0);
//        for (size_t j = 0; j < 10000; ++j) { /*训练集的后10000个数据作为验证集（validation_set）*/
//            if (!network.test(image.get(50000+j), label.get(50000+j))) fail++;
//        }
//        cout << "epoch" << k+1 << ':' << fail << endl;
//        if (fail < minError) {
//            minError = fail;
//            noImprovementOccurredFor = 0;
//        } else noImprovementOccurredFor++;
//        if (noImprovementOccurredFor > 15) break;
//    }
//    int fail = 0;
//    //image.setTranslation(1, 1);
//    for (size_t j = 0; j < 10000; ++j) {
//        if (!network.test(image.get(50000+j), label.get(50000+j))) fail++;
//    }
//    cout << fail << endl;
//    //network.save("/home/wjy50/mnist/network4");
//}

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

    FeedForwardNN nn;

    ConvLayer convLayer(28, 28, 1, 5, 5, 6, 1, 1, 0, 0, L_RE_LU);
    nn.addLayer(&convLayer);

    PoolingLayer poolingLayer(convLayer.getOutputWidth(), convLayer.getOutputHeight(), 2, 2, 2, 2, convLayer.getKernelCount(), MAX_POOLING);
    nn.addLayer(&poolingLayer);

    ConvLayer convLayer1(poolingLayer.getOutputWidth(), poolingLayer.getOutputHeight(), poolingLayer.getChannelCount(), 3, 3, 16, 1, 1, 0, 0, L_RE_LU);
    nn.addLayer(&convLayer1);

    PoolingLayer poolingLayer1(convLayer1.getOutputWidth(), convLayer1.getOutputHeight(), 2, 2, 2, 2, convLayer1.getKernelCount(), MEAN_POOLING);
    nn.addLayer(&poolingLayer1);

    FullyConnLayer layer(150, poolingLayer1.getNeuronCount(), L_RE_LU);
    nn.addLayer(&layer);

    FullyConnLayer output(10, layer.getNeuronCount(), OUTPUT_ACTIVATOR);
    nn.addLayer(&output);

    nn.buildUpNetwork(20);

    /*FloatType learningRate = 0.01;
    SimpleSGDOptimizer optimizer1(learningRate, 5, trainSetSize, convLayer.getWeightCount(), convLayer.getBiasCount());
    convLayer.setOptimizer(&optimizer1);
    SimpleSGDOptimizer optimizer2(learningRate, 5, trainSetSize, convLayer1.getWeightCount(), convLayer1.getBiasCount());
    convLayer1.setOptimizer(&optimizer2);
    SimpleSGDOptimizer optimizer3(learningRate, 5, trainSetSize, layer.getWeightCount(), layer.getBiasCount());
    layer.setOptimizer(&optimizer3);
    SimpleSGDOptimizer optimizer4(learningRate, 5, trainSetSize, output.getWeightCount(), output.getBiasCount());
    output.setOptimizer(&optimizer4);*/

    AdaMaxOptimizer optimizer1(convLayer.getWeightCount(), convLayer.getBiasCount());
    convLayer.setOptimizer(&optimizer1);
    AdaMaxOptimizer optimizer2(convLayer1.getWeightCount(), convLayer1.getBiasCount());
    convLayer1.setOptimizer(&optimizer2);
    AdaMaxOptimizer optimizer3(layer.getWeightCount(), layer.getBiasCount());
    layer.setOptimizer(&optimizer3);
    AdaMaxOptimizer optimizer4(output.getWeightCount(), output.getBiasCount());
    output.setOptimizer(&optimizer4);

    MNISTDataSet trainSet("/home/wjy50/mnist/train-images.idx3-ubyte", "/home/wjy50/mnist/train-labels.idx1-ubyte");
    MNISTDataSet testSet("/home/wjy50/mnist/t10k-images.idx3-ubyte", "/home/wjy50/mnist/t10k-labels.idx1-ubyte");

    MNISTNormalizer normalizer;

    normalizer.add(trainSet, trainSetSize);
    normalizer.confirm();
    normalizer.div(trainSet, trainSetSize);
    normalizer.finish();

    trainSet.setNormalizer(&normalizer);
    testSet.setNormalizer(&normalizer);

    int noImprovementOccurredFor = 0;
    int minError = 0x7fffffff;
    FloatType in[28 * 28];
    FloatType label[10];
    for (int k = 0; k < 200; ++k) {
        long st = clock();
        layer.setDropoutFraction(0.5);
        nn.SGD(trainSet, trainSetSize);
        int fail = 0;
        layer.setDropoutFraction(0);
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

void newNNCIFAR()
{
    FeedForwardNN nn;

    int trainSetSize = 50000;

    ConvLayer convLayer(32, 32, 3, 5, 5, 6, 1, 1, 0, 0, L_RE_LU);
    nn.addLayer(&convLayer);

    PoolingLayer poolingLayer(convLayer.getOutputWidth(), convLayer.getOutputHeight(), 2, 2, 2, 2, convLayer.getKernelCount(), MAX_POOLING);
    nn.addLayer(&poolingLayer);

    ConvLayer convLayer1(poolingLayer.getOutputWidth(), poolingLayer.getOutputHeight(), poolingLayer.getChannelCount(), 3, 3, 16, 1, 1, 0, 0, L_RE_LU);
    nn.addLayer(&convLayer1);

    PoolingLayer poolingLayer1(convLayer1.getOutputWidth(), convLayer1.getOutputHeight(), 2, 2, 2, 2, convLayer1.getKernelCount(), MEAN_POOLING);
    nn.addLayer(&poolingLayer1);

    FullyConnLayer layer(150, poolingLayer1.getNeuronCount(), L_RE_LU);
    nn.addLayer(&layer);

    FullyConnLayer output(10, layer.getNeuronCount(), OUTPUT_ACTIVATOR);
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

    AdaMaxOptimizer optimizer1(convLayer.getWeightCount(), convLayer.getBiasCount());
    convLayer.setOptimizer(&optimizer1);
    AdaMaxOptimizer optimizer2(convLayer1.getWeightCount(), convLayer1.getBiasCount());
    convLayer1.setOptimizer(&optimizer2);
    AdaMaxOptimizer optimizer3(layer.getWeightCount(), layer.getBiasCount());
    layer.setOptimizer(&optimizer3);
    AdaMaxOptimizer optimizer4(output.getWeightCount(), output.getBiasCount());
    output.setOptimizer(&optimizer4);

    const char *paths[6] = {
            "/home/wjy50/cifar/data_batch_1.bin",
            "/home/wjy50/cifar/data_batch_2.bin",
            "/home/wjy50/cifar/data_batch_3.bin",
            "/home/wjy50/cifar/data_batch_4.bin",
            "/home/wjy50/cifar/data_batch_5.bin",
            "/home/wjy50/cifar/test_batch.bin"
    };
    CIFARDataSet trainSet1(paths, 5);
    CIFARDataSet testSet(paths + 5, 1);

    CIFARNormalizer normalizer;

    normalizer.add(trainSet1);
    normalizer.confirm();

    normalizer.div(trainSet1);
    normalizer.finish();

    trainSet1.setNormalizer(&normalizer);
    testSet.setNormalizer(&normalizer);

    int noImprovementOccurredFor = 0;
    int minError = 0x7fffffff;
    FloatType in[32 * 32 * 3];
    FloatType label[10];
    for (int k = 0; k < 200; ++k) {
        long st = clock();
        layer.setDropoutFraction(0.5);
        nn.SGD(trainSet1, trainSetSize);
        int fail = 0;
        layer.setDropoutFraction(0);
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

int main()
{
    std::cout << "Hello, World!" << std::endl;
    newNNMNIST();
    return 0;
}

//TODO Adam/AdaMax自动转SGD
//TODO batch normalization