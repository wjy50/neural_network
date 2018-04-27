/**
 * Created by wjy50 on 18-4-24.
 */

#ifndef NEURAL_NETWORK_FEEDFORWARDNN_H
#define NEURAL_NETWORK_FEEDFORWARDNN_H

#include <vector>

#include "layer/AbsLayer.h"
#include "../../data/DataSet.h"

using namespace std;

namespace ffw
{
    class FeedForwardNN
    {
    private:
        vector<AbsLayer*> layers;

        bool built;
    public:
        FeedForwardNN();

        /**
         * 添加神经网络层
         * @param layer
         */
        void addLayer(AbsLayer *layer);

        /**
         * 完成网络的建立
         * 调用后不允许再调用addLayer
         */
        void buildUpNetwork();

        /**
         * 随机梯度下降
         * @param trainSet 训练数据集
         * @param label 训练集标签
         * @param altTrainSetSize 可选训练集大小，如需取训练集的一部分请传大于零的数
         * @param miniBatchSize mini batch大小
         */
        void SGD(DataSet &trainSet, DataSet &label, size_t miniBatchSize, size_t altTrainSetSize = 0);

        /**
         * 前馈计算输出
         * @param x 符合输入要求的输入向量/矩阵
         * @return 计算结果，调用者不能delete[]
         */
        const double * feedForward(const double *x);
    };
}

#endif //NEURAL_NETWORK_FEEDFORWARDNN_H
