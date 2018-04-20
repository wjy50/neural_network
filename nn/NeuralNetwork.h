/**
 * Created by wjy50 on 18-4-16.
 */

#ifndef NEURAL_NETWORK_NEURALNETWORK_H
#define NEURAL_NETWORK_NEURALNETWORK_H

#include "../mnist/mnist.h"

class NeuralNetwork
{
private:
    int *nums;

    double **weights;
    double **biases;

    double **nablaWeights;
    double **nablaBiases;

    double **zs;
    double **as;
    double **deltas;

    int layerCount;
    double (*activation)(double);
    double (*dActivation_dx)(double);
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
     * 输入矩阵(列向量)并计算输出
     * @param x 输入矩阵或列向量，向量维数必须与输入层神经元个数相同，若是矩阵则行数必须与输入层神经元个数相同
     * @return 输出矩阵(列向量)
     */
    //Matrix<double> &input(Matrix<double> &x);

    /**
     * 训练
     * 自动根据输入计算输出和cost function，并使用反向传播梯度下降法进行训练
     * @param x 输入列向量，维数必须与输入层神经元个数相同
     * @param y 标签，即输入对应的正确输出，维数与输出层神经元个数相同
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

    ~NeuralNetwork();
};


#endif //NEURAL_NETWORK_NEURALNETWORK_H
