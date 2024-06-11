#include <cuda_runtime.h>
#include <iostream>
#include <cublas_v2.h>
using namespace std;

cublasHandle_t handle;

void gemm(float* out, float* in, float* w, float* b, int N, int D) {
    float alpha = 1.0;
    int lda = D;
    int incx = 1;
    float beta = 0.0;
    int incy = 1;

    cublasStatus_t status =  cublasSgemv(handle, CUBLAS_OP_T, D, N, &alpha, w, lda, in, incx, &beta, out, incy);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cublasSgemv failed\n");
    }
}

int main(){

    cublasCreate(&handle);
    int N = 4096;
    int D = 4096;
    float *in, *w, *out, *b;
    cudaMalloc(&in, N * D * sizeof(float));
    cudaMalloc(&w, D * N * sizeof(float));
    cudaMalloc(&out, N * sizeof(float));
    cudaMalloc(&b, N * sizeof(float));

    gemm(out, in, w, b, N, D);

    cublasDestroy(handle);

    return 0;
}