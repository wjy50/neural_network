#include "CUDAHelper.h"

/**
 * Created by wjy50 on 18-5-17.
 */

#if ENABLE_CUDA

cublasHandle_t M_CUBLAS_HANDLE;

void initializeCUDA()
{
    cublasCreate_v2(&M_CUBLAS_HANDLE);
}

void destroyCUDA()
{
    cublasDestroy_v2(M_CUBLAS_HANDLE);
    cudaThreadExit();
}

#endif //ENABLE_CUDA