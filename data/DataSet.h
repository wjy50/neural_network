/**
 * Created by wjy50 on 18-4-25.
 */

#ifndef NEURAL_NETWORK_DATASET_H
#define NEURAL_NETWORK_DATASET_H


#include <cstddef>
#include "../math/mtype.h"

class DataNormalizer
{
public:
    /**
     * 数据标准化
     * @param x
     */
    virtual void normalize(FloatType *x) = 0;
};

class DataSet
{
protected:
    DataNormalizer *normalizer;
public:
    DataSet();

    /**
     * 获取数据集大小
     * @return 数据集大小
     */
    virtual int getSize() = 0;

    /**
     * 获取一批数据
     * @param data 数据容器，大小应为 数据大小 * 数据数量
     * @param label 标签容器，大小同理
     * @param indices 数据索引序列，大小为数据数量
     * @param n 数据数量
     */
    virtual void getBatch(FloatType *data, FloatType *label, const int *indices, int n) = 0;

    void setNormalizer(DataNormalizer *normalizer);
};


#endif //NEURAL_NETWORK_DATASET_H
