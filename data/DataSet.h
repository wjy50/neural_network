/**
 * Created by wjy50 on 18-4-25.
 */

#ifndef NEURAL_NETWORK_DATASET_H
#define NEURAL_NETWORK_DATASET_H


#include <cstddef>

class DataSet
{
public:
    /**
     * 获取数据集大小
     * @return 数据集大小
     */
    virtual size_t getSize() = 0;

    /**
     * 获取数据
     * @param i 数据编号
     * @return 存储数据的向量/矩阵
     */
    virtual double *get(size_t i) = 0;
};


#endif //NEURAL_NETWORK_DATASET_H
