#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

__device__ void warpReduce(volatile int* partial_sum, int tx){
    partial_sum[tx] += partial_sum[tx + 16];
    partial_sum[tx] += partial_sum[tx + 8];
    partial_sum[tx] += partial_sum[tx + 4];
    partial_sum[tx] += partial_sum[tx + 2];
    partial_sum[tx] += partial_sum[tx + 1];
}

__global__ void sum_reduction_kernel(float* out, float* a){
    __shared__ int partial_sum[256];
    int tx = threadIdx.x + blockIdx.x * blockDim.x*4;
    partial_sum[threadIdx.x] = a[tx] + a[tx + blockDim.x] + a[tx + 2*blockDim.x] + a[tx + 3*blockDim.x];
    __syncthreads();

    for(int s=blockDim.x/2; s>=32; s>>=1){

        if(threadIdx.x < s){
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    if(threadIdx.x < 16){
        warpReduce(partial_sum, threadIdx.x);
    }

    if(threadIdx.x==0){ 
        out[blockIdx.x] = partial_sum[0];
    }
}

void print_arr(float* arr, int N){
    for(int i=0;i<N;i++){
        printf("%f ", arr[i]);
    }
        printf("\n");
}

int main(){
    int N = 1<<16;
    int size = N*sizeof(float);
    float *h_a,*h_b;
    float *d_a,*d_b;
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    for(int i=0;i<N;i++)h_a[i] = 1.0f;
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    int tb = 256/4;
    int nb = (N+tb-1)/(4*tb);
    sum_reduction_kernel<<<nb, tb>>>(d_b, d_a);
    sum_reduction_kernel<<<1, tb>>>(d_b, d_b);    
    
    float* out = (float*)malloc(sizeof(float)*tb);
    cudaMemcpy(out, d_b, sizeof(float)*tb, cudaMemcpyDeviceToHost);
    printf("Sum: %f\n", out[0]);

    return 0;
}