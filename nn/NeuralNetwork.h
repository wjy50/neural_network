/**
 * Created by wjy50 on 18-4-16.
 */

#ifndef NEURAL_NETWORK_NEURALNETWORK_H
#define NEURAL_NETWORK_NEURALNETWORK_H

#include "../mnist/mnist.h"

class NeuralNetwork
{
private:
    int *nums;                             /*记录每层的神经元数*/

    double **weights;
    double **biases;

    double **nablaWeights;                 /*cost function对权重的梯度*/
    double **nablaBiases;                  /*cost function对bias的梯度*/

    double **zs;                           /*记录每层线性输出（实际上是激活函数导函数在z的取值）*/
    double **as;                           /*记录每层激活输出*/
    double **deltas;                       /*每层的δ*/

    int layerCount;
    double (*activation)(double);          /*激活函数*/
    double (*dActivation_dx)(double);      /*激活函数导函数*/

    /**
     * 接受输入计算输出
     * 同时保留每层的带权输出z和激活输出a
     * @param x 输入向量
     */
    void tracedFeedForward(double x[]);

    /**
     * 反向传播求每层的δ
     * @param y 期望的输出
     */
    void backPropagate(double *y);

    /**
     * 计算cost function关于weights和biases的梯度
     * 包含了前馈和反向传播
     * 是SGD的核心之一
     * @param x 输入
     * @param y 期望的输出
     */
    void calculateNabla(double *x, double *y);

    void clearNabla();
public:
    /**
     * 构造神经网络
     * @param nums 每层神经元个数
     * @param layerCount 神经元层数
     */
    NeuralNetwork(const int nums[], int layerCount, double (*activation)(double), double (*dActivation_dx)(double));

    /**
     * 获取层数
     * @return 层数
     */
    int getLayerCount();

    /**
     * 初始化神经网络
     */
    void initialize();

    /**
     * mini batch == 1 的GD
     * 对单个输入进行梯度下降，直接更新参数
     * @param x 输入列向量，维数必须与输入层神经元个数相同
     * @param y 标签，即输入对应的期望输出，维数与输出层神经元个数相同
     * @return 输出是否符合期望的输出
     */
    bool train(double x[], double y[]);

    /**
     * 测试网络对输入的计算结果是否符合期望输出，只针对MNIST手写数字数据集
     * @param x 输入
     * @param y 输出
     * @return 测试结果，符合则返回true
     */
    bool test(double x[], double y[]);

    /**
     * 随机梯度下降（Stochastic Gradient Descent）
     * 随机顺序遍历训练集
     * 每计算一个mini batch后更新参数
     * @param xs 训练集
     * @param ys 训练集标签
     * @param trainSetSize 训练集大小
     * @param miniBatchSize mini batch大小
     * @return
     */
    int SGD(MNISTImage &xs, MNISTLabel &ys, int trainSetSize, int miniBatchSize);

    /**
     * 接受输入并计算输出
     * @param x 输入向量
     * @return 输出向量
     */
    double *feedForward(double x[]);

    ~NeuralNetwork();
};


#endif //NEURAL_NETWORK_NEURALNETWORK_H
