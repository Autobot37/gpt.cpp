#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


__global__ void add_kernel(float* out, float* a, float* b, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        out[idx] += a[idx] + b[idx];
    }
}

void add(float* out, float* a, float* b, int N){
    dim3 block_size(256);
    dim3 grid_size((N + block_size.x - 1) / block_size.x);
    add_kernel<<<grid_size, block_size>>>(out, a, b, N);
}
void print_arr(float* arr, int N){
    for(int i=0;i<N;i++){
        printf("%f ", arr[i]);
    }
        printf("\n");
}

int main(){
    
    float* h_a = (float*)malloc(128*128*sizeof(float));
    for(int i=0;i<128*128;i++){
        h_a[i] = 1.0f;
    }

    float* arr;
    cudaMalloc(&arr, 128*128*sizeof(float));
    cudaMemcpy(arr, h_a, 128*128*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    float* arrpointer1 = arr;
    float* arrpointer2 = arr;
    add(arr, arrpointer1, arrpointer2, 128*128);
    cudaDeviceSynchronize();
    //print_arr(arr, 128);
    float* check = (float*)malloc(128*128*sizeof(float));
    cudaMemcpy(check, arr, 128*128*sizeof(float), cudaMemcpyDeviceToHost);
    print_arr(check, 128);

    return 0;
}