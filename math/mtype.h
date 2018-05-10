/**
 * Created by wjy50 on 18-5-4.
 */

#ifndef NEURAL_NETWORK_MTYPE_H
#define NEURAL_NETWORK_MTYPE_H

#define USE_DOUBLE 0

#if USE_DOUBLE

typedef double FloatType;

#define M_CBLAS_GEMV cblas_dgemv

#else

typedef float FloatType;

#define M_CBLAS_GEMV cblas_sgemv

#endif

#endif //NEURAL_NETWORK_MTYPE_H
