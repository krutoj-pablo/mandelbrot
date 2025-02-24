#pragma once

#include <cuda_runtime.h> // Add the necessary include for CUDA types
#include <assert.h>
#ifdef __cplusplus
extern "C" {
#endif

#ifdef __CUDACC__
#define CUDA_CALL(x) {const cudaError_t a = (x); if (a != cudaSuccess){printf("\nCuda error: %s (err_num=%d)\n", cudaGetErrorString(a), a); cudaDeviceReset(); assert(0);}}

__host__ void cuda_error_check(const char *prefix, const char *postfix)
{
    if (cudaPeekAtLastError() != cudaSuccess)
    {
        printf("\n%s: %s :%s\n", prefix, cudaGetErrorString(cudaGetLastError()), postfix);
        cudaDeviceReset();
        exit(1);
    }
}

__host__ void wait_exit(void)
{
    printf("\nPress any key to exit");
    getchar();
}

typedef unsigned short int u16;
typedef unsigned int u32;

#else
#define CUDA_CALL(x) x
#endif

#ifdef __cplusplus
}
#endif
