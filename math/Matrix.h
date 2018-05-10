/**
 * Created by wjy50 on 18-4-16.
 */

#ifndef NEURAL_NETWORK_MATRIX_H
#define NEURAL_NETWORK_MATRIX_H

#include "mtype.h"

#ifdef MATRIX_CLASS

#include <cstring>

template<typename T>
class Matrix
{
private:
    T *elements;
    int r, c;
public:
    /**
     * 构造矩阵
     * @param r 行数
     * @param c 列数
     */
    Matrix(int r, int c)
    {
        this->r = r;
        this->c = c;
        elements = new T[r*c];
    }

    /**
     * 由数组构造矩阵
     * @param r 行数
     * @param c 列数
     * @param elements 数组
     * @param copy true表示复制数组内容,false表示直接接管elements数组
     */
    Matrix(int r, int c, T *elements, bool copy)
    {
        this->r = r;
        this->c = c;
        if (copy) {
            this->elements = new T[r*c];
            memcpy(this->elements, elements, r*c* sizeof(T));
        } else {
            this->elements = elements;
        }
    }

    /**
     * 操作符重载实现矩阵乘法
     * 运算结果是新的Matrix对象
     * @param m2 右边矩阵,行数必须等于this的列数
     * @return 运算结果,使用后注意delete
     */
    Matrix<T> &operator *(Matrix<T> &m2)
    {
        auto *n = new Matrix<T>(r, m2.c);
        for (int i = 0; i < r * m2.c; ++i) {
            n->elements[i] = 0;
        }
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                if (elements[i*c+j] != 0) for (int k = 0; k < m2.c; ++k) {
                        (*n)[i][k] += elements[i*c+j] * m2[j][k];
                    }
            }
        }
        return *n;
    }

    /**
     * 操作符重载实现矩阵数乘
     * 运算结果是新的Matrix对象
     * @param e 乘数
     * @return 运算结果,使用后注意delete
     */
    Matrix<T> &operator *(T &e)
    {
        auto *n = new Matrix<T>(r, c);
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                (*n)[i][j] = elements[i][j] * e;
            }
        }
        return *n;
    }

    /**
     * 操作符重载实现矩阵加法
     * 运算结果是新的Matrix对象
     * @param m2 行数列数与this相同
     * @return 运算结果,使用后注意delete
     */
    Matrix<T> &operator +(Matrix<T> &m2)
    {
        auto *n = new Matrix<T>(r, c);
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                (*n)[i][j] = elements[i][j] + m2[i][j];
            }
        }
        return *n;
    }

    /**
     * 操作符重载实现按元素加法
     * 运算结果是新的Matrix对象
     * @param e 加数
     * @return 运算结果,使用后注意delete
     */
    Matrix<T> &operator +(T &e)
    {
        auto *n = new Matrix<T>(r, c);
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                (*n)[i][j] = elements[i][j] + e;
            }
        }
        return *n;
    }

    /**
     * 操作符重载实现矩阵减法
     * 运算结果是新的Matrix对象
     * @param m2 行数列数与this相同
     * @return 运算结果,使用后注意delete
     */
    Matrix<T> &operator -(Matrix<T> &m2)
    {
        auto *n = new Matrix<T>(r, c);
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                (*n)[i][j] = elements[i][j] - m2[i][j];
            }
        }
        return *n;
    }

    /**
     * 取出某一行
     * @param r 行号,从0开始
     * @return 该行元素数组头的const指针
     */
    T *operator [](int r) { return elements + r*this->c; }

    /**
     * 获取行数
     * @return 行数
     */
    int getRowCount() { return r; }

    /**
     * 获取列数
     * @return 列数
     */
    int getColumnCount() { return c; }

    ~Matrix() { delete[] elements; }
};

#else

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * 矩阵乘法，用新的数组存放结果
 * @param lhs 左边矩阵
 * @param rhs 右边矩阵
 * @param x 左边矩阵行数
 * @param y 左边矩阵列数，也是右边矩阵行数
 * @param z 右边矩阵列数
 * @return 结果
 */
FloatType *multiplyMM(const FloatType *lhs, const FloatType *rhs, int x, int y, int z);

/**
 * 矩阵乘法，结果放在给定数组内
 * @param r 结果容器
 * @param lhs 左边矩阵
 * @param rhs 右边矩阵
 * @param x 左边矩阵行数
 * @param y 左边矩阵列数，也是右边矩阵行数
 * @param z 右边矩阵列数
 */
void multiplyMMTo(FloatType *r, const FloatType *lhs, const FloatType *rhs, int x, int y, int z);

/**
 * 矩阵同位元素相加，用新的数组存放结果
 * @param m1 矩阵1
 * @param m2 矩阵2
 * @param x 行数
 * @param y 列数
 * @return 结果
 */
FloatType *addMM(const FloatType *m1, const FloatType *m2, int x, int y);

/**
 * 矩阵同位元素相加，结果放在给定数组内
 * @param r 结果容器
 * @param m1 矩阵1
 * @param m2 矩阵2
 * @param x 行数
 * @param y 列数
 */
void addMMTo(FloatType *r, const FloatType *m1, const FloatType *m2, int x, int y);

/**
 * 矩阵同位元素相减，用新的数组存放结果
 * @param m1 矩阵1
 * @param m2 矩阵2
 * @param x 行数
 * @param y 列数
 * @return 结果
 */
FloatType *subMM(const FloatType *m1, const FloatType *m2, int x, int y);

/**
 * 矩阵同位元素相减，结果放在给定数组内
 * @param r 结果容器
 * @param m1 矩阵1
 * @param m2 矩阵2
 * @param x 行数
 * @param y 列数
 */
void subMMTo(FloatType *r, const FloatType *m1, const FloatType *m2, int x, int y);

/**
 * 矩阵乘向量，用新的数组存放结果
 * @param m 矩阵
 * @param v 向量
 * @param x 矩阵行数
 * @param y 矩阵列数，也是向量v维数
 * @return 结果向量
 */
FloatType *multiplyMV(const FloatType *m, const FloatType *v, int x, int y);

/**
 * 矩阵数乘，用新的数组存放结果
 * @param n 系数
 * @param m 矩阵
 * @param x 行数
 * @param y 列数
 * @return 结果
 */
FloatType *multiplyNM(FloatType n, const FloatType *m, int x, int y);

/**
 * 矩阵数乘，结果放在给定数组内
 * @param r 结果容器
 * @param n 系数
 * @param m 矩阵
 * @param x 行数
 * @param y 列数
 */
void multiplyNMTo(FloatType *r, FloatType n, const FloatType *m, int x, int y);

/**
 * 矩阵按元素加数，用新的数组存放结果
 * @param n 加数
 * @param m 矩阵
 * @param x 行数
 * @param y 列数
 * @return
 */
FloatType *addNM(FloatType n, const FloatType *m, int x, int y);

/**
 * 矩阵按元素加数，结果放在给定数组内
 * @param r 结果容器
 * @param n 加数
 * @param m 矩阵
 * @param x 行数
 * @param y 列数
 */
void addNMTo(FloatType *r, FloatType n, const FloatType *m, int x, int y);

/**
 * 矩阵数乘，结果就在m中
 * @param n 系数
 * @param m 矩阵
 * @param x 行数
 * @param y 列数
 */
void mMultiplyN(FloatType n, FloatType *m, int x, int y);

/**
 * 矩阵按元素加数，结果就在m中
 * @param n 加数
 * @param m 矩阵
 * @param x 行数
 * @param y 列数
 */
void mAddN(FloatType n, FloatType *m, int x, int y);

/**
 * 矩阵同位元素相乘，用新的数组存放结果
 * @param m1 矩阵1
 * @param m2 矩阵2
 * @param x 行数
 * @param y 列数
 * @return 结果
 */
FloatType *multiplyMMElem(const FloatType *m1, const FloatType *m2, int x, int y);

/**
 * 矩阵同位元素相乘，结果放在给定数组内
 * @param r 结果容器
 * @param m1 矩阵1
 * @param m2 矩阵2
 * @param x 行数
 * @param y 列数
 */
void multiplyMMElemTo(FloatType *r, const FloatType *m1, const FloatType *m2, int x, int y);

/**
 * 转置矩阵，用新的数组存放结果
 * @param m 矩阵
 * @param x 行数
 * @param y 列数
 * @return 结果
 */
FloatType *transposeM(const FloatType *m, int x, int y);

/**
 * 转置矩阵，结果放在给定数组内
 * @param r 结果容器
 * @param m 矩阵
 * @param x 行数
 * @param y 列数
 */
void transposeMTo(FloatType *r, const FloatType *m, int x, int y);

/**
 * 转置矩阵与向量相乘，即T(M) * V，T(M)为矩阵M转置
 * 直接计算免去转置的时间和空间开销
 * @param m 矩阵
 * @param v 向量
 * @param x 矩阵行数，因为是转置后相乘，也是向量维数
 * @param y 矩阵列数
 * @return 结果
 */
FloatType *multiplyTransposedMV(const FloatType *m, const FloatType *v, int x, int y);

/**
 * 转置矩阵与向量相乘，即T(M) * V，T(M)为矩阵M转置
 * 直接计算免去转置的时间和空间开销
 * 结果保存在指定数组
 * @param r 结果容器
 * @param m 矩阵
 * @param v 向量
 * @param x 矩阵行数，因为是转置后相乘，也是向量维数
 * @param y 矩阵列数
 */
void multiplyTransposedMVTo(FloatType *r, const FloatType *m, const FloatType *v, int x, int y);

#define USE_CBLAS 0

#if USE_CBLAS

#include "../openblas/cblas.h"

//#define multiplyTransposedMVTo(r, m, v, x, y) M_CBLAS_GEMV(CblasRowMajor, CblasTrans, (x), (y), 1, m, (y), (v), 1, 0, (r), 1)
#define multiplyMVTo(r, m, v, x, y) M_CBLAS_GEMV(CblasRowMajor, CblasNoTrans, (x), (y), 1, m, (y), (v), 1, 0, (r), 1)

#else

/**
 * 矩阵乘向量，结果放在给定数组内
 * @param r 结果容器
 * @param m 矩阵
 * @param v 向量
 * @param x 矩阵行数
 * @param y 矩阵列数，也是向量v维数
 */
void multiplyMVTo(FloatType *r, const FloatType *m, const FloatType *v, int x, int y);

#endif

#ifdef __cplusplus
};
#endif

#endif //MATRIX_CLASS

#endif //NEURAL_NETWORK_MATRIX_H
