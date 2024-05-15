#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

#include "../utils.h"

void encoder_forward(float* out, int* inp, float* wte, float* wpe, int B, int T, int C){
    for(int b = 0;b<B;b++){
        for(int t = 0;t<T;t++){
            float* out_p = out + b * T * C + t * C;
            float* wte_p = wte + inp[b * T + t] * C;
            float* wpe_p = wpe + t * C;
            for(int i = 0;i<C;i++){
                out_p[i] = wte_p[i] + wpe_p[i];
            }
        }
    };
}

__global__ void encoder_forward_kernel(float* out, int* inp, float* wte, float* wpe, int B, int T, int C){
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < B && t < T) {
        float* out_p = out + b * T * C + t * C;
        float* wte_p = wte + inp[b * T + t] * C;
        float* wpe_p = wpe + t * C;
        for(int i = 0; i < C; i++) {
            out_p[i] = wte_p[i] + wpe_p[i];
        }
    }
}

void encoder_forward_gpu(float* out, int* inp, float* wte, float* wpe, int B, int T, int C){
    dim3 blockDim(16, 16); 
    dim3 gridDim((B + blockDim.x - 1) / blockDim.x, (T + blockDim.y - 1) / blockDim.y); // Adjust grid size
    encoder_forward_kernel<<<gridDim, blockDim>>>(out, inp, wte, wpe, B, T, C);
    cudaDeviceSynchronize();
    
}

void rand_init(float* arr, int size){
    for(int i = 0;i<size;i++){
        arr[i] = (float)rand() / RAND_MAX;
    }
}
void print_mat(float* arr,int size){
    for(int i = 0;i<size;i++){
	printf("%f",arr[i]);
    }
    printf("\n");
}

int main(){

    int mul = 4;
    int C = 256*mul;
    int T = 256*mul;
    int V = 8192*mul;
    int B = 8*mul;
    int* inp = (int*)mallocCheck(sizeof(int) * B * T);
    float* out = (float*)mallocCheck(sizeof(float) * B * T * C);
    float* wpe = (float*)mallocCheck(sizeof(float) * T * C);
    float* wte = (float*)mallocCheck(sizeof(float) * V * C);
    rand_init(wpe, T * C);
    rand_init(wte, V * C);
    for(int i = 0;i<B*T;i++){
        inp[i] = rand() % V;
    }

    clock_t start, end;
    double time_used;
    start = clock();

    encoder_forward(out, inp, wte, wpe, B,T,C);

    end = clock();
    time_used = ((double)(end - start)) / CLOCKS_PER_SEC;  
    printf("Time Used CPU: %lf seconds\n", time_used);
    
    //
    float *d_out;
    int *d_inp;
    float *d_wte;
    float *d_wpe;
    cudaMalloc(&d_out, B * T * C * sizeof(float));
    cudaMalloc(&d_inp, B * T * sizeof(int));
    cudaMalloc(&d_wte, V * C * sizeof(float)); 
    cudaMalloc(&d_wpe, T * C * sizeof(float));
    cudaMemcpy(d_inp, inp, B * T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wte, wte, V * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wpe, wpe, T * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaEvent_t start_gpu, stop_gpu;
    float gpu_time;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu, 0);

    encoder_forward_gpu(d_out, d_inp, d_wte, d_wpe, B,T,C);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_gpu, 0);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
    time_used = gpu_time / 1000.0;
    printf("Time Used GPU: %lf seconds\n", time_used);

    float* check;
    check = (float*)mallocCheck(sizeof(float) * B * T * C);
    cudaMemcpy(check, d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
	
   // print_mat(check, B*T*C);
   // print_mat(out, B*T*C);

    for(int i = 0;i<B*T*C;i++){
        if(abs(out[i] - check[i]) > 1e-5f){
            printf("Incorrect output Try Again\n");
            return 1;
        }
    }
    printf("Correct output Yay!\n");
            
    free(inp);
    free(out);
    free(wpe);
    free(wte);
    cudaFree(d_inp);
    cudaFree(d_out);
    cudaFree(d_wpe);
    cudaFree(d_wte);


    return 0;
}