/**
 * Created by wjy50 on 18-4-16.
 */

#ifndef NEURAL_NETWORK_NEURALNETWORK_H
#define NEURAL_NETWORK_NEURALNETWORK_H

#include "../mnist/mnist.h"
#include "../math/Activator.h"

static double (*ACTIVATION_FUNCTIONS[])(double) = {
        sigmoid,
        ReLU,
        lReLU
};

static double (*D_ACTIVATION_FUNCTIONS[])(double) = {
        dSigmoid_dx,
        dReLU_dx,
        dLReLU_dx
};

typedef enum ACTIVATOR
{
    SIGMOID,
    RE_LU,
    L_RE_LU
}Activator;

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
    Activator activator;
    double (*activation)(double);          /*激活函数*/
    double (*dActivation_dx)(double);      /*激活函数导函数*/

    double eta;                            /*学习率（的相反数）*/
    double reg;                            /*正规化参数regularization parameter*/

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
    NeuralNetwork(const int nums[], int layerCount, Activator activator);

    /**
     * 从文件中读取参数以构造神经网络
     * @param filePath 文件路径
     */
    explicit NeuralNetwork(const char *filePath);

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
     */
    void SGD(MNISTImage &xs, MNISTLabel &ys, int trainSetSize, int miniBatchSize);

    /**
     * 接受输入并计算输出
     * @param x 输入向量
     * @return 输出向量
     */
    double *feedForward(double x[]);

    /**
     * 设置学习率
     * @param l 学习率
     */
    void setLearningRate(double l);

    /**
     * 设置（L2）正规化参数
     * @param r
     */
    void setRegularizationParam(double r);

    /**
     * 将神经网络的参数保存到文件
     * @param path 文件路径
     */
    void save(const char *path);

    ~NeuralNetwork();
};


#endif //NEURAL_NETWORK_NEURALNETWORK_H
