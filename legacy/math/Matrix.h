/**
 * Created by wjy50 on 18-4-16.
 */

#ifndef NEURAL_NETWORK_MATRIX_H
#define NEURAL_NETWORK_MATRIX_H

#include "mtype.h"
#include "../cuda/CUDAHelper.h"

///**
// * 矩阵乘法，用新的数组存放结果
// * @param lhs 左边矩阵
// * @param rhs 右边矩阵
// * @param x 左边矩阵行数
// * @param y 左边矩阵列数，也是右边矩阵行数
// * @param z 右边矩阵列数
// * @return 结果
// */
//FloatType *multiplyMM(const FloatType *lhs, const FloatType *rhs, int x, int y, int z);
//
///**
// * 矩阵同位元素相加，用新的数组存放结果
// * @param m1 矩阵1
// * @param m2 矩阵2
// * @param x 行数
// * @param y 列数
// * @return 结果
// */
//FloatType *addMM(const FloatType *m1, const FloatType *m2, int x, int y);
//
///**
// * 矩阵同位元素相减，用新的数组存放结果
// * @param m1 矩阵1
// * @param m2 矩阵2
// * @param x 行数
// * @param y 列数
// * @return 结果
// */
//FloatType *subMM(const FloatType *m1, const FloatType *m2, int x, int y);
//
///**
// * 矩阵乘向量，用新的数组存放结果
// * @param m 矩阵
// * @param v 向量
// * @param x 矩阵行数
// * @param y 矩阵列数，也是向量v维数
// * @return 结果向量
// */
//FloatType *multiplyMV(const FloatType *m, const FloatType *v, int x, int y);
//
///**
// * 矩阵数乘，用新的数组存放结果
// * @param n 系数
// * @param m 矩阵
// * @param x 行数
// * @param y 列数
// * @return 结果
// */
//FloatType *multiplyNM(FloatType n, const FloatType *m, int x, int y);
//
///**
// * 矩阵按元素加数，用新的数组存放结果
// * @param n 加数
// * @param m 矩阵
// * @param x 行数
// * @param y 列数
// * @return
// */
//FloatType *addNM(FloatType n, const FloatType *m, int x, int y);
//
///**
// * 矩阵同位元素相乘，用新的数组存放结果
// * @param m1 矩阵1
// * @param m2 矩阵2
// * @param x 行数
// * @param y 列数
// * @return 结果
// */
//FloatType *multiplyMMElem(const FloatType *m1, const FloatType *m2, int x, int y);
//
///**
// * 转置矩阵，用新的数组存放结果
// * @param m 矩阵
// * @param x 行数
// * @param y 列数
// * @return 结果
// */
//FloatType *transposeM(const FloatType *m, int x, int y);
//
///**
// * 转置矩阵与向量相乘，即T(M) * V，T(M)为矩阵M转置
// * 直接计算免去转置的时间和空间开销
// * @param m 矩阵
// * @param v 向量
// * @param x 矩阵行数，因为是转置后相乘，也是向量维数
// * @param y 矩阵列数
// * @return 结果
// */
//FloatType *multiplyTransposedMV(const FloatType *m, const FloatType *v, int x, int y);

#if ENABLE_CUDA

static FloatType DEF_ALPHA = 1;
static FloatType DEF_BETA = 0;

#define multiplyTransposedMVTo(r, m, v, x, y) M_CUBLAS_GEMV(M_CUBLAS_HANDLE, CUBLAS_OP_N, (y), (x), &DEF_ALPHA, m, (y), (v), 1, &DEF_BETA, (r), 1)

#define multiplyMVTo(r, m, v, x, y) M_CUBLAS_GEMV(M_CUBLAS_HANDLE, CUBLAS_OP_T, (y), (x), &DEF_ALPHA, m, (y), (v), 1, &DEF_BETA, (r), 1)

#define multiplyMMTo(r, lhs, rhs, x, y, z) M_CUBLAS_GEMM(M_CUBLAS_HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, (z), (x), (y), &DEF_ALPHA, (rhs), (z), (lhs), (y), &DEF_BETA, r, (z))

//TODO cuda

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
 * 矩阵同位元素相加，结果放在给定数组内
 * @param r 结果容器
 * @param m1 矩阵1
 * @param m2 矩阵2
 * @param x 行数
 * @param y 列数
 */
void addMMTo(FloatType *r, const FloatType *m1, const FloatType *m2, int x, int y);

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
 * 矩阵数乘，结果放在给定数组内
 * @param r 结果容器
 * @param n 系数
 * @param m 矩阵
 * @param x 行数
 * @param y 列数
 */
void multiplyNMTo(FloatType *r, FloatType n, const FloatType *m, int x, int y);

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
 * 转置矩阵，结果放在给定数组内
 * @param r 结果容器
 * @param m 矩阵
 * @param x 行数
 * @param y 列数
 */
void transposeMTo(FloatType *r, const FloatType *m, int x, int y);

/**
 * 矩阵同位元素相乘，结果放在给定数组内
 * @param r 结果容器
 * @param m1 矩阵1
 * @param m2 矩阵2
 * @param x 行数
 * @param y 列数
 */
void multiplyMMElemTo(FloatType *r, const FloatType *m1, const FloatType *m2, int x, int y);

#endif //ENABLE_CUDA

#endif //NEURAL_NETWORK_MATRIX_H
