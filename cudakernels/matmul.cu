#include <cuda_runtime.h>
#include <iostream>
#include <cublas_v2.h>
using namespace std;

void matmul(float* out, float* in, float* w, float* b, int N ,int D){
    //in is D, w is N,D, b is N, out is N
    int i;
    #pragma omp parallel for private(i)
    for(i = 0;i<N;i++){
        float sum = (b!=NULL) ? b[i] : 0;
        for(int j = 0;j<D;j++){
            sum += in[j] * w[i*D + j];
        }
        out[i] = sum;
    } 
}

__global__
void matmul_kernel(float* out, float* in, float* w, float* b, int N ,int D){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        float sum = (b!=NULL) ? b[i] : 0;
        for(int j = 0;j<D;j++){
            sum += in[j] * w[i*D + j];
        }
        out[i] = sum;
    }
}

void matmul_gpu(float* out, float* in, float* w, float* b, int N ,int D){
    int num_threads = 1024;
    int num_blocks = (N + num_threads - 1) / num_threads;
    matmul_kernel<<<num_blocks, num_threads>>>(out, in, w, b, N, D);
}

__global__ void add_bias(float* out, float* b, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        out[i] += b[i];
    }
}

void gemm(float* out, float* in, float* w, float* b, int N, int D) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0;
    int lda = D;
    int incx = 1;
    float beta = 0.0;
    int incy = 1;

    cublasStatus_t status =  cublasSgemv(handle, CUBLAS_OP_T, D, N, &alpha, w, lda, in, incx, &beta, out, incy);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cublasSgemv failed\n");
    }
    if(b != NULL)
    add_bias<<<(N + 1023) / 1024, 1024>>>(out, b, N);

    cublasDestroy(handle);
}

void rand_init(float* arr, int N){
    for(int i = 0;i<N;i++){
        arr[i] = (float)rand() / RAND_MAX;
    }
}

void isequal(float* a, float* b, int n){
    float maxval = -INFINITY;
    for(int i = 0;i<n;i++){
        maxval = fmaxf(maxval, fmaxf(a[i], b[i]));
    }
    float eps = 1e-5;
    for(int i = 0;i<n;i++){
        if(fabs(a[i] - b[i]) > eps * (maxval + 1)){
            cout << "Mismatch at index " << i << " CPU: " << a[i] << " GPU: " << b[i] << endl;
        }
    }
}

int main(){

    int N = 1024;
    int D = 2048;

    float* in = (float*)malloc(D * sizeof(float));
    float* w = (float*)malloc(N * D * sizeof(float));
    float* b = (float*)malloc(N * sizeof(float));
    float* out = (float*)malloc(N * sizeof(float));

    rand_init(in, D);
    rand_init(w, N * D);
    rand_init(b, N);

    float* d_in, *d_w, *d_b, *d_out;
    cudaMalloc(&d_in, D * sizeof(float));
    cudaMalloc(&d_w, N * D * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    cudaMemcpy(d_in, in, D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

    matmul(out, in, w, b, N, D);

    matmul_gpu(d_out, d_in, d_w, d_b, N, D);

    float* out_gpu = (float*)malloc(N * sizeof(float));
    cudaMemcpy(out_gpu, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    isequal(out, out_gpu, N);

    //----sgemm
    float* out_gemm;
    cudaMalloc(&out_gemm, N * sizeof(float));
    gemm(out_gemm, d_in, d_w, d_b, N, D);

    float* out_gemm_cpu = (float*)malloc(N * sizeof(float));
    cudaMemcpy(out_gemm_cpu, out_gemm, N * sizeof(float), cudaMemcpyDeviceToHost);

    isequal(out, out_gemm_cpu, N);

    return 0;
}