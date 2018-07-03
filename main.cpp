#include <memory>
#include <random>
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
#include "v3/nn/layer/ResidualBlock.h"
#include "v3/utils/debug.h"

using namespace std;

template<typename T>
void printM(const T *m, int r, int c)
{
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            nout(DEBUG_LEVEL_INFO) << m[i * c + j] << ' ';
        }
        nout(DEBUG_LEVEL_INFO) << endl;
    }
    nout(DEBUG_LEVEL_INFO) << endl;
}

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

    BatchNormLayer batchNormLayer(28 * 28);
    //nn.addLayer(&batchNormLayer);

    ConvLayer convLayer(28, 28, 1, 5, 5, 64, 1, 1, 2, 2, true);
    nn.addLayer(&convLayer);

    BatchNormLayer batchNormLayer1(convLayer.getOutputDim());
    //nn.addLayer(&batchNormLayer1);

    LReLULayer lReLULayer(convLayer.getOutputDim(), 0.01);
    nn.addLayer(&lReLULayer);

    MaxPoolingLayer poolingLayer(convLayer.getOutputWidth(), convLayer.getOutputHeight(), convLayer.getKernelCount(), 2, 2, 2, 2);
    nn.addLayer(&poolingLayer);

    ConvLayer convLayer2(poolingLayer.getOutputWidth(), poolingLayer.getOutputHeight(), poolingLayer.getChannelCount(), 3, 3, 64, 1, 1, 1, 1, true);
    nn.addLayer(&convLayer2);

    BatchNormLayer batchNormLayer4(convLayer2.getOutputDim());
    //nn.addLayer(&batchNormLayer4);

    LReLULayer lReLULayer3(convLayer2.getOutputDim(), 0.01);
    nn.addLayer(&lReLULayer3);

    ConvLayer convLayer1(convLayer2.getOutputWidth(), convLayer2.getOutputHeight(), convLayer2.getKernelCount(), 3, 3, 64, 1, 1, 0, 0, true);
    nn.addLayer(&convLayer1);

    BatchNormLayer batchNormLayer2(convLayer1.getOutputDim());
    //nn.addLayer(&batchNormLayer2);

    LReLULayer lReLULayer1(convLayer1.getOutputDim(), 0.01);
    nn.addLayer(&lReLULayer1);

    MeanPoolingLayer poolingLayer1(convLayer1.getOutputWidth(), convLayer1.getOutputHeight(), convLayer1.getKernelCount(), 2, 2, 2, 2);
    nn.addLayer(&poolingLayer1);

    LinearLayer layer(poolingLayer1.getOutputDim(), 400, true);
    nn.addLayer(&layer);

    BatchNormLayer batchNormLayer3(layer.getOutputDim());
    nn.addLayer(&batchNormLayer3);

    LReLULayer lReLULayer2(layer.getOutputDim(), 0.01);
    nn.addLayer(&lReLULayer2);

    DropoutLayer dropoutLayer(lReLULayer2.getOutputDim());
    dropoutLayer.setDropoutFraction(0.5);
    //nn.addLayer(&dropoutLayer);

    LinearLayer layer1(lReLULayer2.getOutputDim(), 10);
    nn.addLayer(&layer1);

    SoftMaxOutputLayer output(layer1.getOutputDim());
    nn.addLayer(&output);

    nn.buildUpNetwork(100);

    AdaMaxOptimizer optimizer1;
    convLayer.setOptimizer(&optimizer1);
    AdaMaxOptimizer optimizer2;
    convLayer1.setOptimizer(&optimizer2);
    AdaMaxOptimizer optimizer3;
    layer.setOptimizer(&optimizer3);
    AdaMaxOptimizer optimizer4;
    layer1.setOptimizer(&optimizer4);
    AdaMaxOptimizer optimizer5;
    batchNormLayer.setOptimizer(&optimizer5);
    AdaMaxOptimizer optimizer6;
    batchNormLayer1.setOptimizer(&optimizer6);
    AdaMaxOptimizer optimizer7;
    batchNormLayer2.setOptimizer(&optimizer7);
    AdaMaxOptimizer optimizer8;
    batchNormLayer3.setOptimizer(&optimizer8);
    AdaMaxOptimizer optimizer9;
    convLayer2.setOptimizer(&optimizer9);
    AdaMaxOptimizer optimizer10;
    batchNormLayer4.setOptimizer(&optimizer10);

    MNISTDataSet trainSet("/home/wjy50/mnist/train-images.idx3-ubyte", "/home/wjy50/mnist/train-labels.idx1-ubyte");
    MNISTDataSet testSet("/home/wjy50/mnist/t10k-images.idx3-ubyte", "/home/wjy50/mnist/t10k-labels.idx1-ubyte");

    int noImprovementOccurredFor = 0;
    int minError = 0x7fffffff;
    auto *in = allocArray<FloatType>(28 * 28);
    auto *label = allocArray<FloatType>(10);
    FloatType out[10];
    FloatType localLabel[10];
    auto *indices = allocArray<int>(10000);
    incArray(indices, 10000, 50000);
    for (int k = 0; k < 200; ++k) {
        long st = clock();
        nn.optimize(trainSet, trainSetSize);
        int fail = 0;
        for (int i = 0; i < 10000; ++i) {
            trainSet.getBatch(in, label, indices + i, 1);
            const FloatType *o = nn.feedForward(in);
            cudaMemcpy(out, o, 10 * sizeof(FloatType), cudaMemcpyDeviceToHost);
            cudaMemcpy(localLabel, label, 10 * sizeof(FloatType), cudaMemcpyDeviceToHost);
            if (!test(out, localLabel, 10)) {
                fail++;
            }
        }
        nout(DEBUG_LEVEL_INFO) << "epoch" << k+1 << ':' << fail;
        if (fail < minError) {
            minError = fail;
            noImprovementOccurredFor = 0;
            nout(DEBUG_LEVEL_INFO) << '*';
        } else noImprovementOccurredFor++;
        nout(DEBUG_LEVEL_INFO) << endl << "time:" << clock() - st << endl;
        if (noImprovementOccurredFor > 30) break;
    }
    int fail = 0;
    incArray(indices, 10000);
    for (int j = 0; j < 10000; ++j) {
        testSet.getBatch(in, label, indices + j, 1);
        const FloatType *o = nn.feedForward(in);
        cudaMemcpy(out, o, 10 * sizeof(FloatType), cudaMemcpyDeviceToHost);
        cudaMemcpy(localLabel, label, 10 * sizeof(FloatType), cudaMemcpyDeviceToHost);
        if (!test(out, localLabel, 10)) {
            fail++;
        }
    }
    nout(DEBUG_LEVEL_INFO) << fail << endl;
    freeArray(in);
    freeArray(label);
    freeArray(indices);
}

void newNNCIFAR10()
{
    NeuralNetwork nn;

    int trainSetSize = 50000;

    BatchNormLayer batchNormLayer(32 * 32 * 3);
    nn.addLayer(&batchNormLayer);

    ConvLayer convLayer(32, 32, 3, 5, 5, 32, 1, 1, 0, 0, true);
    nn.addLayer(&convLayer);

    BatchNormLayer batchNormLayer1(convLayer.getOutputDim());
    nn.addLayer(&batchNormLayer1);

    LReLULayer lReLULayer(convLayer.getOutputDim(), 0.01);
    nn.addLayer(&lReLULayer);

    MaxPoolingLayer poolingLayer(convLayer.getOutputWidth(), convLayer.getOutputHeight(), convLayer.getKernelCount(), 2, 2, 2, 2);
    nn.addLayer(&poolingLayer);

    ConvLayer convLayer2(poolingLayer.getOutputWidth(), poolingLayer.getOutputHeight(), poolingLayer.getChannelCount(), 3, 3, 64, 1, 1, 1, 1, true);
    nn.addLayer(&convLayer2);

    BatchNormLayer batchNormLayer4(convLayer2.getOutputDim());
    nn.addLayer(&batchNormLayer4);

    LReLULayer lReLULayer3(convLayer2.getOutputDim(), 0.01);
    nn.addLayer(&lReLULayer3);

    ConvLayer convLayer1(convLayer2.getOutputWidth(), convLayer2.getOutputHeight(), convLayer2.getKernelCount(), 3, 3, 64, 1, 1, 1, 1, true);
    nn.addLayer(&convLayer1);

    BatchNormLayer batchNormLayer2(convLayer1.getOutputDim());
    nn.addLayer(&batchNormLayer2);

    LReLULayer lReLULayer1(convLayer1.getOutputDim(), 0.01);
    nn.addLayer(&lReLULayer1);

    MeanPoolingLayer poolingLayer1(convLayer1.getOutputWidth(), convLayer1.getOutputHeight(), convLayer1.getKernelCount(), 2, 2, 2, 2);
    nn.addLayer(&poolingLayer1);

    LinearLayer layer(poolingLayer1.getOutputDim(), 200, true);
    nn.addLayer(&layer);

    BatchNormLayer batchNormLayer3(layer.getOutputDim());
    nn.addLayer(&batchNormLayer3);

    LReLULayer lReLULayer2(layer.getOutputDim(), 0.01);
    nn.addLayer(&lReLULayer2);

    DropoutLayer dropoutLayer(lReLULayer2.getOutputDim());
    dropoutLayer.setDropoutFraction(0.5);
    //nn.addLayer(&dropoutLayer);

    LinearLayer layer1(lReLULayer2.getOutputDim(), 10);
    nn.addLayer(&layer1);

    SoftMaxOutputLayer output(layer1.getOutputDim());
    nn.addLayer(&output);

    nn.buildUpNetwork(100);

    AdaMaxOptimizer optimizer1;
    convLayer.setOptimizer(&optimizer1);
    AdaMaxOptimizer optimizer2;
    convLayer1.setOptimizer(&optimizer2);
    AdaMaxOptimizer optimizer3;
    layer.setOptimizer(&optimizer3);
    AdaMaxOptimizer optimizer4;
    layer1.setOptimizer(&optimizer4);
    AdaMaxOptimizer optimizer5;
    batchNormLayer.setOptimizer(&optimizer5);
    AdaMaxOptimizer optimizer6;
    batchNormLayer1.setOptimizer(&optimizer6);
    AdaMaxOptimizer optimizer7;
    batchNormLayer2.setOptimizer(&optimizer7);
    AdaMaxOptimizer optimizer8;
    batchNormLayer3.setOptimizer(&optimizer8);
    AdaMaxOptimizer optimizer9;
    convLayer2.setOptimizer(&optimizer9);
    AdaMaxOptimizer optimizer10;
    batchNormLayer4.setOptimizer(&optimizer10);

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
    auto *in = allocArray<FloatType>(32 * 32 * 3);
    auto *label = allocArray<FloatType>(10);
    FloatType out[10], localLabel[10];
    auto *indices = allocArray<int>(10000);
    incArray(indices, 10000);
    for (int k = 0; k < 200; ++k) {
        long st = clock();
        nn.optimize(trainSet1, trainSetSize);
        nout(DEBUG_LEVEL_INFO) << "training time = " << clock() - st << endl;
        int fail = 0;
        for (int j = 0; j < 10000; ++j) {
            testSet.getBatch(in, label, indices + j, 1);
            const FloatType *o = nn.feedForward(in);
            cudaMemcpy(out, o, 10 * sizeof(FloatType), cudaMemcpyDeviceToHost);
            cudaMemcpy(localLabel, label, 10 * sizeof(FloatType), cudaMemcpyDeviceToHost);
            if (!test(out, localLabel, 10)) {
                fail++;
            }
        }
        nout(DEBUG_LEVEL_INFO) << "epoch" << k+1 << ':' << fail;
        if (fail < minError) {
            minError = fail;
            noImprovementOccurredFor = 0;
            nout(DEBUG_LEVEL_INFO) << '*';
        } else noImprovementOccurredFor++;
        nout(DEBUG_LEVEL_INFO) << endl << "time:" << clock() - st << endl;
        if (noImprovementOccurredFor > 30) break;
    }
    freeArray(in);
    freeArray(label);
    freeArray(indices);
}

void testAutoEncoder()
{
    AutoEncoder autoEncoder;

    int trainSetSize = 50000;

    ConvLayer encoder(28, 28, 1, 3, 3, 6);
    autoEncoder.addLayer(&encoder);

    LReLULayer lReLULayer(encoder.getOutputDim(), 0.01);
    autoEncoder.addLayer(&lReLULayer);

    ConvLayer decoder(encoder.getOutputWidth(), encoder.getOutputHeight(), 6, 3, 3, 1, 1, 1, 2, 2);
    autoEncoder.addLayer(&decoder);

    SigmoidOutputLayer output(decoder.getOutputDim());
    autoEncoder.addLayer(&output);

    autoEncoder.buildUpAutoEncoder(100);

    AdamOptimizer optimizer1;
    encoder.setOptimizer(&optimizer1);
    AdamOptimizer optimizer2;
    decoder.setOptimizer(&optimizer2);

    MNISTDataSet trainSet("/home/wjy50/mnist/train-images.idx3-ubyte", "/home/wjy50/mnist/train-labels.idx1-ubyte");
    MNISTDataSet testSet("/home/wjy50/mnist/t10k-images.idx3-ubyte", "/home/wjy50/mnist/t10k-labels.idx1-ubyte");

    auto *in = allocArray<FloatType>(28 * 28);
    FloatType out[28 * 28];
    auto indices = allocArray<int>(10000);
    incArray(indices, 10000, 50000);
    for (int k = 201; k < 10000; ++k) {
        long st = clock();
        autoEncoder.optimize(trainSet, trainSetSize);
        long t = clock() - st;
        nout(DEBUG_LEVEL_INFO) << "epoch" << k+1 << ':' << endl;
        trainSet.getBatch(in, nullptr, indices + k, 1);
        const FloatType *o = autoEncoder.feedForward(in);
        cudaMemcpy(out, o, 28 * 28 * sizeof(FloatType), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 28; ++i) {
            for (int l = 0; l < 28; ++l) {
                auto r = static_cast<int>(out[i * 28 + l] * 9);
                if (r > 0) nout(DEBUG_LEVEL_INFO) << r << ' ';
                else nout(DEBUG_LEVEL_INFO) << "  ";
            }
            /*nout(DEBUG_LEVEL_INFO) << "    ";
            for (int l = 0; l < 28; ++l) {
                auto r = static_cast<int>(in[i * 28 + l] * 9);
                if (r > 0) nout(DEBUG_LEVEL_INFO) << r << ' ';
                else nout(DEBUG_LEVEL_INFO) << "  ";
            }*/
            nout(DEBUG_LEVEL_INFO) << endl;
        }
        /*if (k % 10 == 0) {
            MNISTData2Bmp data2Bmp("/home/wjy50/mnist1.bmp");
            data2Bmp.writeData(in);
            data2Bmp.close();
            MNISTData2Bmp data2BmpG("/home/wjy50/mnistG1.bmp");
            data2BmpG.writeData(o);
            data2BmpG.close();
        }*/
        nout(DEBUG_LEVEL_INFO) << t << endl;
    }
    freeArray(in);
    freeArray(indices);
}

void testCIFAR10ResNet()
{
    NeuralNetwork nn;

    int trainSetSize = 50000;
    int miniBatchSize = 100;

    FloatType learningRate = 0.002;
    FloatType beta1 = 0.9;
    FloatType beta2 = 0.999;
    FloatType weightDecay = 0.9999;

    BatchNormLayer batchNormLayer(32 * 32 * 3);
    nn.addLayer(&batchNormLayer);

    ConvLayer convLayer(32, 32, 3, 5, 5, 32, 1, 1, 0, 0, true);
    nn.addLayer(&convLayer);

    BatchNormLayer batchNormLayer1(convLayer.getOutputDim());
    nn.addLayer(&batchNormLayer1);

    LReLULayer lReLULayer(convLayer.getOutputDim(), 0.01);
    nn.addLayer(&lReLULayer);

    MaxPoolingLayer poolingLayer(convLayer.getOutputWidth(), convLayer.getOutputHeight(), convLayer.getKernelCount(), 2, 2, 2, 2);
    nn.addLayer(&poolingLayer);

    ConvLayer upConv(poolingLayer.getOutputWidth(), poolingLayer.getOutputHeight(), poolingLayer.getChannelCount(), 1, 1, 64, 1, 1, 0, 0, true);
    nn.addLayer(&upConv);


    vector<LayerBase *> dynamicLayers;
    vector<OptimizerBase *> dynamicOptimizers;
    vector<AdaMaxOptimizer *> optimizers;
    for (int i = 0; i < 2; ++i) {
        auto *residualBlock = new ResidualBlock(upConv.getOutputDim());
        nn.addLayer(residualBlock);
        dynamicLayers.push_back(residualBlock);


        auto *batchNormLayer2 = new BatchNormLayer(upConv.getOutputDim());
        residualBlock->addLayer(batchNormLayer2);
        dynamicLayers.push_back(batchNormLayer2);

        auto *optimizer8 = new AdaMaxOptimizer(learningRate, beta1, beta2, weightDecay, batchNormLayer2->getOutputDim());
        batchNormLayer2->setOptimizer(optimizer8);
        dynamicOptimizers.push_back(optimizer8);
        optimizers.push_back(optimizer8);


        auto *lReLULayer1 = new LReLULayer(batchNormLayer2->getOutputDim(), 0.01);
        residualBlock->addLayer(lReLULayer1);
        dynamicLayers.push_back(lReLULayer1);


        auto *convLayer1 = new ConvLayer(upConv.getOutputWidth(), upConv.getOutputHeight(), upConv.getKernelCount(), 3, 3, 64, 1, 1, 1, 1, true);
        residualBlock->addLayer(convLayer1);
        dynamicLayers.push_back(convLayer1);

        auto *optimizer7 = new AdaMaxOptimizer(learningRate, beta1, beta2, weightDecay, convLayer1->getKernelParamCount());
        convLayer1->setOptimizer(optimizer7);
        dynamicOptimizers.push_back(optimizer7);
        optimizers.push_back(optimizer7);


        auto *batchNormLayer3 = new BatchNormLayer(convLayer1->getOutputDim());
        residualBlock->addLayer(batchNormLayer3);
        dynamicLayers.push_back(batchNormLayer3);

        auto *optimizer10 = new AdaMaxOptimizer(learningRate, beta1, beta2, weightDecay, batchNormLayer3->getOutputDim());
        batchNormLayer3->setOptimizer(optimizer10);
        dynamicOptimizers.push_back(optimizer10);
        optimizers.push_back(optimizer10);


        auto *lReLULayer2 = new LReLULayer(batchNormLayer3->getOutputDim(), 0.01);
        residualBlock->addLayer(lReLULayer2);
        dynamicLayers.push_back(lReLULayer2);


        auto *convLayer2 = new ConvLayer(convLayer1->getOutputWidth(), convLayer1->getOutputHeight(), convLayer1->getKernelCount(), 3, 3, 64, 1, 1, 1, 1, true);
        residualBlock->addLayer(convLayer2);
        dynamicLayers.push_back(convLayer2);

        auto *optimizer9 = new AdaMaxOptimizer(learningRate, beta1, beta2, weightDecay, convLayer2->getKernelParamCount());
        convLayer2->setOptimizer(optimizer9);
        dynamicOptimizers.push_back(optimizer9);
        optimizers.push_back(optimizer9);
    }


    BatchNormLayer batchNormLayer3(upConv.getOutputDim());
    nn.addLayer(&batchNormLayer3);

    LReLULayer lReLULayer2(batchNormLayer3.getOutputDim(), 0.01);
    nn.addLayer(&lReLULayer2);

    MeanPoolingLayer poolingLayer1(upConv.getOutputWidth(), upConv.getOutputHeight(), upConv.getKernelCount(), 2, 2, 1, 1);
    nn.addLayer(&poolingLayer1);

    LinearLayer linearLayer(poolingLayer1.getOutputDim(), 200, true);
    nn.addLayer(&linearLayer);

    BatchNormLayer batchNormLayer4(linearLayer.getOutputDim());
    nn.addLayer(&batchNormLayer4);

    LReLULayer lReLULayer3(linearLayer.getOutputDim(), 0.01);
    nn.addLayer(&lReLULayer3);

    LinearLayer linearLayer1(lReLULayer3.getOutputDim(), 10);
    nn.addLayer(&linearLayer1);

    SoftMaxOutputLayer output(linearLayer1.getOutputDim());
    nn.addLayer(&output);

    nn.buildUpNetwork(miniBatchSize);

    AdaMaxOptimizer optimizer1(learningRate, beta1, beta2, weightDecay, convLayer.getKernelParamCount());
    convLayer.setOptimizer(&optimizer1);
    optimizers.push_back(&optimizer1);
    AdaMaxOptimizer optimizer2(learningRate, beta1, beta2, weightDecay, linearLayer.getOutputDim() * linearLayer.getInputDim());
    linearLayer.setOptimizer(&optimizer2);
    optimizers.push_back(&optimizer2);
    AdaMaxOptimizer optimizer3(learningRate, beta1, beta2, weightDecay, linearLayer1.getInputDim() * linearLayer1.getOutputDim());
    linearLayer1.setOptimizer(&optimizer3);
    optimizers.push_back(&optimizer3);
    AdaMaxOptimizer optimizer4(learningRate, beta1, beta2, weightDecay, batchNormLayer.getOutputDim());
    batchNormLayer.setOptimizer(&optimizer4);
    optimizers.push_back(&optimizer4);
    AdaMaxOptimizer optimizer5(learningRate, beta1, beta2, weightDecay, batchNormLayer1.getOutputDim());
    batchNormLayer1.setOptimizer(&optimizer5);
    optimizers.push_back(&optimizer5);
    AdaMaxOptimizer optimizer6(learningRate, beta1, beta2, weightDecay, batchNormLayer4.getOutputDim());
    batchNormLayer4.setOptimizer(&optimizer6);
    optimizers.push_back(&optimizer6);
    AdaMaxOptimizer upOptimizer(learningRate, beta1, beta2, weightDecay, upConv.getKernelParamCount());
    upConv.setOptimizer(&upOptimizer);
    optimizers.push_back(&upOptimizer);
    AdaMaxOptimizer optimizer7(learningRate, beta1, beta2, weightDecay, batchNormLayer3.getOutputDim());
    batchNormLayer3.setOptimizer(&optimizer7);
    optimizers.push_back(&optimizer7);

    const char *paths[6] = {
            "/home/wjy50/cifar/data_batch_1.bin",
            "/home/wjy50/cifar/data_batch_2.bin",
            "/home/wjy50/cifar/data_batch_3.bin",
            "/home/wjy50/cifar/data_batch_4.bin",
            "/home/wjy50/cifar/data_batch_5.bin",
            "/home/wjy50/cifar/test_batch.bin"
    };
    CIFAR10DataSet trainSet(paths, 5);
    CIFAR10DataSet testSet(paths + 5, 1);

    int noImprovementOccurredFor = 0;
    int minError = 0x7fffffff;
    int minTrainError = 0x7fffffff;
    FloatType *in = nn.getInputBuffer();
    FloatType *label = nn.getLabelBuffer();
    FloatType out[10 * miniBatchSize], localLabel[10 * miniBatchSize];
    auto *indices = allocArray<int>(10000);
    incArray(indices, 10000);
    for (int k = 0; k < 200; ++k) {
        long st = clock();
        nn.optimize(trainSet, trainSetSize);
        nout(DEBUG_LEVEL_INFO) << "epoch" << k + 1 << endl;
        nout(DEBUG_LEVEL_INFO) << "training time = " << clock() - st << endl;
        int error = 0;
        int trainError = 0;
        for (int j = 0; j < 10000 / miniBatchSize; ++j) {
            testSet.getBatch(in, label, indices + j * miniBatchSize, miniBatchSize);
            const FloatType *o = nn.feedForward(in, miniBatchSize);
            cudaMemcpy(out, o, 10 * miniBatchSize * sizeof(FloatType), cudaMemcpyDeviceToHost);
            cudaMemcpy(localLabel, label, 10 * miniBatchSize * sizeof(FloatType), cudaMemcpyDeviceToHost);
            for (int i = 0; i < miniBatchSize; ++i) {
                if (!test(out + i * 10, localLabel + i * 10, 10)) {
                    error++;
                }
            }
        }
        nout(DEBUG_LEVEL_INFO) << "test error:" << error;
        if (error < minError) {
            minError = error;
            noImprovementOccurredFor = 0;
            nout(DEBUG_LEVEL_INFO) << '*';
        } else noImprovementOccurredFor++;
        for (int j = 0; j < 10000 / miniBatchSize; ++j) {
            trainSet.getBatch(in, label, indices + j * miniBatchSize, miniBatchSize);
            const FloatType *o = nn.feedForward(in, miniBatchSize);
            cudaMemcpy(out, o, 10 * miniBatchSize * sizeof(FloatType), cudaMemcpyDeviceToHost);
            cudaMemcpy(localLabel, label, 10 * miniBatchSize * sizeof(FloatType), cudaMemcpyDeviceToHost);
            for (int i = 0; i < miniBatchSize; ++i) {
                if (!test(out + i * 10, localLabel + i * 10, 10)) {
                    trainError++;
                }
            }
        }
        nout(DEBUG_LEVEL_INFO) << endl << "train error:" << trainError;
        if (trainError < minTrainError) {
            minTrainError = trainError;
            nout(DEBUG_LEVEL_INFO) << '*';
        }
        nout(DEBUG_LEVEL_INFO) << endl << "time:" << clock() - st << endl;
        if ((k + 1) % 10 == 0) {
            for (AdaMaxOptimizer *optimizer : optimizers) {
                optimizer->setLearningRate(optimizer->getLearningRate() / 10);
            }
        }
        if (noImprovementOccurredFor > 30) break;
    }
    freeArray(indices);

    for (LayerBase *layer : dynamicLayers) {
        delete layer;
    }
    dynamicLayers.clear();

    for (OptimizerBase *optimizer : dynamicOptimizers) {
        delete optimizer;
    }
    dynamicOptimizers.clear();
}

void testGPU()
{
    NeuralNetwork nn;

    int trainSetSize = 50000;

    BatchNormLayer batchNormLayer(32 * 32 * 3);
    //nn.addLayer(&batchNormLayer);

    LinearLayer linearLayer(32 * 32 * 3, 200);
    nn.addLayer(&linearLayer);

    BatchNormLayer batchNormLayer4(linearLayer.getOutputDim());
    //nn.addLayer(&batchNormLayer4);

    LReLULayer lReLULayer3(linearLayer.getOutputDim(), 0.01);
    nn.addLayer(&lReLULayer3);

    LinearLayer linearLayer1(lReLULayer3.getOutputDim(), 10);
    nn.addLayer(&linearLayer1);

    SoftMaxOutputLayer output(linearLayer1.getOutputDim());
    nn.addLayer(&output);

    nn.buildUpNetwork(100);

    AdaMaxOptimizer optimizer;
    batchNormLayer.setOptimizer(&optimizer);
    AdaMaxOptimizer optimizer2;
    linearLayer.setOptimizer(&optimizer2);
    AdaMaxOptimizer optimizer3;
    linearLayer1.setOptimizer(&optimizer3);
    AdaMaxOptimizer optimizer6;
    batchNormLayer4.setOptimizer(&optimizer6);

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
    auto *in = allocArray<FloatType>(32 * 32 * 3);
    auto *label = allocArray<FloatType>(10);
    FloatType out[10], localLabel[10];
    auto *indices = allocArray<int>(10000);
    incArray(indices, 10000);
    for (int k = 0; k < 200; ++k) {
        long st = clock();
        nn.optimize(trainSet1, trainSetSize);
        nout(DEBUG_LEVEL_INFO) << "training time = " << clock() - st << endl;
        int fail = 0;
        for (int j = 0; j < 10000; ++j) {
            testSet.getBatch(in, label, indices + j, 1);
            const FloatType *o = nn.feedForward(in);
            cudaMemcpy(out, o, 10 * sizeof(FloatType), cudaMemcpyDeviceToHost);
            cudaMemcpy(localLabel, label, 10 * sizeof(FloatType), cudaMemcpyDeviceToHost);
            if (!test(out, localLabel, 10)) {
                fail++;
            }
        }
        nout(DEBUG_LEVEL_INFO) << "epoch" << k+1 << ':' << fail;
        if (fail < minError) {
            minError = fail;
            noImprovementOccurredFor = 0;
            nout(DEBUG_LEVEL_INFO) << '*';
        } else noImprovementOccurredFor++;
        nout(DEBUG_LEVEL_INFO) << endl << "time:" << clock() - st << endl;
        if (noImprovementOccurredFor > 30) break;
    }
    freeArray(in);
    freeArray(label);
    freeArray(indices);
}

int main()
{
    nout() << "Hello, World!" << endl;
#if ENABLE_CUDA
    initializeCUDA();
#endif

    testCIFAR10ResNet();

    /*unique_ptr<FloatType[]> input = make_unique_array<FloatType[]>(32 * 32 * 3 * 10);
    unique_ptr<FloatType[]> input2 = make_unique_array<FloatType[]>(32 * 32 * 3 * 10);
    unique_ptr<FloatType[]> kernel = make_unique_array<FloatType[]>(3 * 3 * 3 * 6);
    unique_ptr<FloatType[]> output = make_unique_array<FloatType[]>(30 * 30 * 6 * 10);

    FloatType *d_i, *d_k, *d_o;
    d_i = allocArray<FloatType>(32 * 32 * 3 * 10);
    d_k = allocArray<FloatType>(3 * 3 * 3 * 6);
    d_o = allocArray<FloatType>(30 * 30 * 6 * 10);

    random_device rd;
    uniform_real_distribution<FloatType> distribution(0, 1);

    for (int i = 0; i < 3 * 3 * 3 * 6; ++i) {
        kernel[i] = distribution(rd);
    }
    for (int i = 0; i < 30 * 30 * 6 * 10; ++i) {
        output[i] = distribution(rd);
    }

    cudaMemcpy(d_k, kernel.get(), 3 * 3 * 3 * 6 * sizeof(FloatType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_o, output.get(), 30 * 30 * 6 * 10 * sizeof(FloatType), cudaMemcpyHostToDevice);

    long st = clock();

    convBP(d_o, 30, 30, d_i, 32, 32, 3, d_k, 3, 3, 6, 1, 1, 0, 0, 10);

    long t1 = clock() - st;

    cudaMemcpy(input.get(), d_i, 32 * 32 * 3 * 10 * sizeof(FloatType), cudaMemcpyDeviceToHost);

    st = clock();

    convBP2(d_o, 30, 30, d_i, 32, 32, 3, d_k, 3, 3, 6, 1, 1, 0, 0, 10);

    long t2 = clock() - st;

    cudaMemcpy(input2.get(), d_i, 32 * 32 * 3 * 10 * sizeof(FloatType), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 32 * 32 * 3 * 10; ++i) {
        if (fabs(input[i] - input2[i]) > 1e-3) nout() << input[i] - input2[i] << ' ';
    }
    nout() << endl;

    nout() << t1 << endl;
    nout() << t2 << endl;*/

#if ENABLE_CUDA
    destroyCUDA();
#endif
    return 0;
}

//TODO Adam/AdaMax自动转SGD