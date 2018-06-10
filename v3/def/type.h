/**
 * Created by wjy50 on 18-5-18.
 */

#ifndef NEURAL_NETWORK_TYPE_H
#define NEURAL_NETWORK_TYPE_H

#define USE_DOUBLE 0

#if USE_DOUBLE

typedef double FloatType;

#define M_CUBLAS_GEMV cublasDgemv_v2

#define M_CUBLAS_GEMM cublasDgemm_v2

#define CU_EXP exp

#define CU_FMAX fmax

#define CU_FABS fabs

#else

typedef float FloatType;

#define M_CUBLAS_GEMV cublasSgemv_v2

#define M_CUBLAS_GEMM cublasSgemm_v2

#define CU_EXP expf

#define CU_FMAX fmaxf

#define CU_FABS fabsf

#endif

#endif //NEURAL_NETWORK_TYPE_H
