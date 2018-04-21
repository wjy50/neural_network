/**
 * Created by wjy50 on 18-4-16.
 */

#include <iostream>

using namespace std;

#include <cstring>
#include <cmath>
#include <random>
#include <memory>
#include "NeuralNetwork.h"
#include "../math/Matrix.h"
#include "../math/permutation.h"
#include "../math/Activator.h"

NeuralNetwork::NeuralNetwork(const int *nums, int layerCount, Activator activator)
{
    this->layerCount = layerCount;
    this->nums = new int[layerCount];
    memcpy(this->nums, nums, layerCount * sizeof(int));
    weights = new double*[layerCount];
    biases = new double*[layerCount];
    zs = new double*[layerCount-1];
    as = new double*[layerCount];
    deltas = new double*[layerCount];
    this->activator = activator;
    this->activation = ACTIVATION_FUNCTIONS[activator];
    this->dActivation_dx = D_ACTIVATION_FUNCTIONS[activator];

    weights[0] = nullptr;
    for (int i = 1; i < layerCount; ++i) {
        /**
         * 第i层矩阵行数为该层神经元数
         *          列数为上一层神经元数
         */
        weights[i] = new double[nums[i]*nums[i-1]];
    }

    biases[0] = nullptr;
    for (int i = 1; i < layerCount; ++i) {
        biases[i] = new double[nums[i]];
    }

    nablaWeights = new double*[layerCount];
    nablaBiases = new double*[layerCount];
    nablaWeights[0] = nullptr;
    for (int i = 1; i < layerCount; ++i) {
        nablaWeights[i] = new double[nums[i]*nums[i-1]];
    }

    nablaBiases[0] = nullptr;
    for (int i = 1; i < layerCount; ++i) {
        nablaBiases[i] = new double[nums[i]];
    }

    zs[0] = nullptr;
    deltas[0] = nullptr;

    eta = -0.1;
    reg = 5;
}

NeuralNetwork::NeuralNetwork(const char *filePath)
{
    FILE *file = fopen(filePath, "rb");
    if (file) {
        fread(&eta, sizeof(double), 1, file);
        fread(&reg, sizeof(double), 1, file);
        fread(&activator, sizeof(Activator), 1, file);
        fread(&layerCount, sizeof(int), 1, file);
        nums = new int[layerCount];
        fread(nums, sizeof(int), static_cast<size_t>(layerCount), file);
        weights = new double*[layerCount];
        weights[0] = nullptr;
        for (int i = 1; i < layerCount; ++i) {
            weights[i] = new double[nums[i]*nums[i-1]];
            fread(weights[i], sizeof(double), (size_t)nums[i]*(size_t)nums[i-1], file);
        }

        biases = new double*[layerCount];
        biases[0] = nullptr;
        for (int i = 1; i < layerCount; ++i) {
            biases[i] = new double[nums[i]];
            fread(biases[i], sizeof(double), (size_t)nums[i], file);
        }
        fclose(file);

        activation = ACTIVATION_FUNCTIONS[activator];
        dActivation_dx = D_ACTIVATION_FUNCTIONS[activator];

        zs = new double*[layerCount-1];
        as = new double*[layerCount];
        deltas = new double*[layerCount];

        nablaWeights = new double*[layerCount];
        nablaBiases = new double*[layerCount];
        nablaWeights[0] = nullptr;
        for (int i = 1; i < layerCount; ++i) {
            nablaWeights[i] = new double[nums[i]*nums[i-1]];
        }

        nablaBiases[0] = nullptr;
        for (int i = 1; i < layerCount; ++i) {
            nablaBiases[i] = new double[nums[i]];
        }

        zs[0] = nullptr;
        deltas[0] = nullptr;
    }
}

double* NeuralNetwork::feedForward(double *x)
{
    double *a = x;
    for (int i = 1; i < layerCount-1; ++i) {
        double *z = multiplyMV(weights[i], a, nums[i], nums[i-1]);
        addMMTo(z, z, biases[i],  nums[i], 1);
        for (int j = 0; j < nums[i]; ++j) {
            z[j] = activation(z[j]);
        }
        if (i > 1) delete[] a;
        a = z;
    }

    int i = layerCount-1;
    a = multiplyMV(weights[i], a, nums[i], nums[i-1]);
    addMMTo(a, a, biases[i], nums[i], 1);
    softMax(a, nums[i]);

    return a;
}

void NeuralNetwork::tracedFeedForward(double *x)
{
    as[0] = x;
    for (int i = 1; i < layerCount-1; ++i) { /*对输出层使用SoftMax激活函数，需要特殊处理*/
        zs[i] = multiplyMV(weights[i], as[i-1], nums[i], nums[i-1]);
        addMMTo(zs[i], zs[i], biases[i], nums[i], 1);
        as[i] = new double[nums[i]];
        for (int j = 0; j < nums[i]; ++j) {
            as[i][j] = activation(zs[i][j]);
            zs[i][j] = dActivation_dx(zs[i][j]);/*后面用到的都是导数，直接在此处理*/
        }
    }

    /*输出层使用SoftMax激活函数能显著提高分类准确度和缩短训练时间*/
    int i = layerCount-1;
    as[i] = multiplyMV(weights[i], as[i-1], nums[i], nums[i-1]);
    addMMTo(as[i], as[i], biases[i], nums[i], 1);
    softMax(as[i], nums[i]);
}

void NeuralNetwork::backPropagate(double *y)
{
    /*使用交叉熵cost function，对a的偏导为a-y*/
    deltas[layerCount-1] = subMM(as[layerCount-1], y, nums[layerCount-1], 1);
    for (int i = layerCount-2; i > 0; --i) {
        double *tw = transposeM(weights[i+1], nums[i+1], nums[i]);
        deltas[i] = multiplyMV(tw, deltas[i+1], nums[i], nums[i+1]);
        multiplyMMElemTo(deltas[i], deltas[i], zs[i], nums[i], 1);
        delete[] tw;
    }
}

void NeuralNetwork::clearNabla()
{
    for (int i = 1; i < layerCount; ++i) {
        memset(nablaWeights[i], 0, sizeof(double)*nums[i]*nums[i-1]);
    }

    for (int i = 1; i < layerCount; ++i) {
        memset(nablaBiases[i], 0, sizeof(double)*nums[i]);
    }
}

void NeuralNetwork::calculateNabla(double *x, double *y)
{
    tracedFeedForward(x);

    /*反向传播计算delta*/
    backPropagate(y);

    /*梯度下降*/
    for (int i = layerCount-1; i > 0; --i) {
        for (int j = 0; j < nums[i]; ++j) {
            double *CWJ = multiplyNM(deltas[i][j], as[i-1], nums[i-1], 1);
            double CBJ = deltas[i][j];
            nablaBiases[i][j] += CBJ;
            addMMTo(nablaWeights[i]+j*nums[i-1], nablaWeights[i]+j*nums[i-1], CWJ, 1, nums[i-1]);
            delete[] CWJ;
        }
    }

    /*释放资源*/
    for (int i = 1; i < layerCount-1; ++i) {
        delete[] zs[i];
    }
    for (int i = 1; i < layerCount; ++i) {
        delete[] as[i];
    }
    for (int i = 1; i < layerCount; ++i) {
        delete[] deltas[i];
    }
}

void NeuralNetwork::SGD(MNISTImage &xs, MNISTLabel &ys, int trainSetSize, int miniBatchSize)
{
    unique_ptr<int> indices(new int[trainSetSize]);
    randomPermutation(indices.get(), trainSetSize);
    int miniBatchCount = trainSetSize / miniBatchSize;
    for (int t = 0; t < miniBatchCount; ++t) {
        clearNabla();

        int *ind = indices.get()+t*miniBatchSize;
        for (int e = 0; e < miniBatchSize; ++e) {
            double *x = xs.get(ind[e]);
            double *y = ys.get(ind[e]);

            calculateNabla(x, y);
        }

        /*更新w和b*/
        for (int i = 1; i < layerCount; ++i) {
            for (int j = 0; j < nums[i]*nums[i - 1]; ++j) {
                weights[i][j] = weights[i][j]*(1+eta*reg/trainSetSize) + eta*nablaWeights[i][j]/miniBatchSize;
            }
        }
        for (int i = 1; i < layerCount; ++i) {
            for (int j = 0; j < nums[i]; ++j) {
                biases[i][j] += eta*nablaBiases[i][j]/miniBatchSize;
            }
        }
    }
}

bool NeuralNetwork::test(double *x, double *y)
{
    double *a = feedForward(x);

    int maxOut = 0;
    for (int i = 1; i < nums[layerCount - 1]; ++i) {
        if (a[i] > a[maxOut]) maxOut = i;
    }
    bool re = y[maxOut] == 1;

    delete[] a;

    return re;
}

int NeuralNetwork::getLayerCount()
{
    return layerCount;
}

void NeuralNetwork::initialize()
{
    std::random_device rd;
    std::normal_distribution<double> distribution(0, 1);
    /*随机生成weight*/
    for (int i = 1; i < layerCount; ++i) {
        double ni = nums[i-1];// sqrt(nums[i-1]);
        for (int j = 0; j < nums[i]; ++j) {
            for (int k = 0; k < nums[i - 1]; ++k) {
                weights[i][j*nums[i-1]+k] = distribution(rd) / ni;
            }
        }
    }

    /*bias全部置0*/
    for (int i = 1; i < layerCount; ++i) {
        memset(biases[i], 0, sizeof(double)*nums[i]);
    }
}

void NeuralNetwork::setLearningRate(double l)
{
    eta = -l;
}

void NeuralNetwork::setRegularizationParam(double r)
{
    reg = r;
}

void NeuralNetwork::save(const char *path)
{
    FILE *file = fopen(path, "wb");
    if (file) {
        fwrite(&eta, sizeof(double), 1, file);
        fwrite(&reg, sizeof(double), 1, file);
        fwrite(&activator, sizeof(Activator), 1, file);
        fwrite(&layerCount, sizeof(int), 1, file);
        fwrite(nums, sizeof(int), static_cast<size_t>(layerCount), file);
        for (int i = 1; i < layerCount; ++i) {
            fwrite(weights[i], sizeof(double), (size_t)nums[i]*(size_t)nums[i-1], file);
        }
        for (int i = 1; i < layerCount; ++i) {
            fwrite(biases[i], sizeof(double), (size_t)nums[i], file);
        }
        fclose(file);
    }
}

NeuralNetwork::~NeuralNetwork()
{
    delete[] nums;
    for (int i = 1; i < layerCount; ++i) {
        delete[] weights[i];
    }
    delete[] weights;
    for (int i = 1; i < layerCount; ++i) {
        delete[] biases[i];
    }
    delete[] biases;

    for (int i = 1; i < layerCount; ++i) {
        delete[] nablaWeights[i];
    }
    delete[] nablaWeights;
    for (int i = 1; i < layerCount; ++i) {
        delete[] nablaBiases[i];
    }
    delete[] nablaBiases;

    delete[] zs;
    delete[] as;
}