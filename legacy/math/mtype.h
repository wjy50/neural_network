/**
 * Created by wjy50 on 18-5-4.
 */

#ifndef NEURAL_NETWORK_MTYPE_H
#define NEURAL_NETWORK_MTYPE_H

#define USE_DOUBLE 0

#if USE_DOUBLE

typedef double FloatType;

#define M_CUBLAS_GEMV cublasDgemv_v2

#define M_CUBLAS_GEMM cublasDgemm_v2

#else

typedef float FloatType;

#define M_CUBLAS_GEMV cublasSgemv_v2

#define M_CUBLAS_GEMM cublasSgemm_v2

#endif

#endif //NEURAL_NETWORK_MTYPE_H
