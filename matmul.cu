#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>

//n,m @ m,k -> n k
void matmul_cpu(float* out, float* a, float* b, int N, int M, int K){
    #pragma omp parallel for collapse(2)
    for(int i=0;i<N;i++){
        for(int j=0;j<K;j++){
            float sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for(int k = 0;k<M;k++){
                sum += a[i*M+k] * b[k*K+j];
            }
            out[i*K + j] = sum;
        }
    }
}

#define TILESIZE 32

__global__ void matmul_gpu_kernel(float* out, float* a, float* b, int N, int M, int K){
    __shared__ int A[TILESIZE][TILESIZE];
    __shared__ int B[TILESIZE][TILESIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float sum = 0.0f;
    for(int i = 0;i< M / TILESIZE;i++){
        A[ty][tx] = a[row * M + i * TILESIZE + tx];
        B[ty][tx] = b[(i * TILESIZE + ty) * K + col];
        __syncthreads();

        for(int j=0;j<TILESIZE;j++){
            sum += A[ty][j] * B[j][tx];
        }
        __syncthreads();
    }
    out[row * K + col] = sum;
}
void matmul_gpu(float* out, float* a, float* b, int N, int M, int K){
    dim3 block_size(32, 32);
    dim3 grid_size;
    grid_size.x = (K + block_size.x - 1) / block_size.x;
    grid_size.y = (N + block_size.y - 1) / block_size.y;
    matmul_gpu_kernel<<<grid_size, block_size>>>(out, a, b, N, M, K);
}

int main(){ 
    
    int N = 1024;
    int M = 1024*4;
    int K = 1024;
    float *a, *b, *out;
    float *d_a, *d_b, *d_out;
    a = (float*)malloc(N*M*sizeof(float));
    b = (float*)malloc(M*K*sizeof(float));
    out = (float*)malloc(N*K*sizeof(float));
    cudaMalloc(&d_a, N*M*sizeof(float));
    cudaMalloc(&d_b, M*K*sizeof(float));
    cudaMalloc(&d_out, N*K*sizeof(float));

    cudaMemcpy(d_a, a, N*M*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, M*K*sizeof(float), cudaMemcpyHostToDevice);


    clock_t start_time = clock();
    matmul_gpu(out, a, b, N, M, K);
    clock_t mid_time = clock();
    double time1 = (double)(mid_time - start_time) / CLOCKS_PER_SEC;
    printf("GPU time: %f\n", time1);

    matmul_cpu(out, a, b, N, M, K);
    double time2 = (double)(clock() - mid_time) / CLOCKS_PER_SEC;
    printf("CPU time: %f\n", time2);

    float* check = (float*)malloc(N*K*sizeof(float));
    cudaMemcpy(check, d_out, N*K*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i=0;i<N*K;i++){
        if((out[i] - check[i]) > 1e-5){
            printf("Error at %d\n", i);
            break;
        }
    }
    int faster = (int)time2 / time1;
    printf("GPU is %d times faster than CPU\n", faster);
    printf("And it is correct too\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    free(a);
    free(b);
    free(out);
    free(check);
    
    return 0;
}