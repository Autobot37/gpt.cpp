#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <omp.h>

#define TILESIZE 8

__global__ void matmul_gpu_kernel(float* out, float* inp, float* weight, float* bias, int B, int T, int C, int OC){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int bt = bx * blockDim.x + tx;
    int oc = by * blockDim.y + ty;

    if(bt<B*T && oc<OC){
        float* inp_p = inp + bt * C;
        float* out_p = out + bt * OC;
        float* weight_p = weight + oc * C;
        float val = (bias != NULL) ? bias[oc] : 0.0f;
        for(int i = 0;i<C;i++){
            val += inp_p[i] * weight_p[i];
        }
        out_p[oc] = val;
    }
    
}
void matmul_forward_gpu(float* out, float* inp, float* weight, float* bias, int B, int T, int C, int OC){
    dim3 block_size(32, 32);
    dim3 grid_size;
    grid_size.x = (B*T + block_size.x - 1) / block_size.x;
    grid_size.y = (OC + block_size.y - 1) / block_size.y;
    matmul_gpu_kernel<<<grid_size, block_size>>>(out, inp, weight, bias, B, T, C, OC);
}
//inp(B,T,C) @  weight(3*C, C).T -> out(B,T,3*C)
void matmul_forward(float* out, float* inp, float* weight, float* bias, int B, int T, int C, int OC){
    #pragma omp parallel for collapse(2) schedule(static)    
    for(int b = 0;b<B;b++){
        for(int t = 0;t<T;t++){
            float* inp_p = inp + b * T * C + t * C;
            float* out_p = out + b * T * OC + t * OC;
            for(int t2=0;t2<OC;t2++){
                float* weight_p = weight + t2* C;
                float val = (bias != NULL) ? bias[t2] : 0.0f;
                #pragma omp simd reduction(+:val) 
                for(int i = 0;i<C;i++){
                    val += inp_p[i] * weight_p[i];
                }
                out_p[t2] = val;
            }
        }
    }
}

void rand_init(float* arr, int size){
    for(int i = 0;i<size;i++){
        arr[i] = (float)rand() / RAND_MAX;
    }
}

int main(){

    int mul = 8;
    int B = 4*mul;
    int T = 128*mul;
    int C = 128*mul;
    int OC = 128*2*mul;

    float *inp, *weight, *bias, *out;
    float *d_inp, *d_weight, *d_bias, *d_out;
    inp = (float*)malloc(B*T*C*sizeof(float));
    weight = (float*)malloc(OC*C*sizeof(float));
    bias = (float*)malloc(OC*sizeof(float));
    out = (float*)malloc(B*T*OC*sizeof(float));
    rand_init(inp, B*T*C);
    rand_init(weight, OC*C);
    rand_init(bias, OC);


    cudaMalloc(&d_inp, B*T*C*sizeof(float));
    cudaMalloc(&d_weight, OC*C*sizeof(float));
    cudaMalloc(&d_bias, OC*sizeof(float));
    cudaMalloc(&d_out, B*T*OC*sizeof(float));

    cudaMemcpy(d_inp, inp, B*T*C*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, OC*C*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, OC*sizeof(float), cudaMemcpyHostToDevice);

    clock_t start, mid, end;
    double cpu_time_used, gpu_time_used;
    start = clock();
    matmul_forward(out, inp, weight, bias, B, T, C, OC);
    mid = clock();
    cpu_time_used = ((double) (mid - start)) / CLOCKS_PER_SEC;

    matmul_forward_gpu(d_out, d_inp, d_weight, d_bias, B, T, C, OC);
    cudaDeviceSynchronize();
    end = clock();
    gpu_time_used = ((double) (end - mid)) / CLOCKS_PER_SEC;

    float* check;
    check = (float*)malloc(B*T*OC*sizeof(float));
    cudaMemcpy(check, d_out, B*T*OC*sizeof(float), cudaMemcpyDeviceToHost);

    printf("CPU time used: %f\n", cpu_time_used);
    printf("GPU time used: %f\n", gpu_time_used);
    int faster = (int)(cpu_time_used / gpu_time_used);
    printf("GPU is %d times faster than CPU\n", faster);

    // for(int i = 0;i<B*T*OC;i++){
    //     if(abs(out[i] - check[i] > 1e-5)){
    //         printf("Incorrect output try again!\n");
    //         return 1;
    //     }
    // }
    // printf("And Correct too!\n");

    free(inp);
    free(weight);
    free(bias);
    free(out);
    free(check);
    cudaFree(d_inp);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_out);





    return 0;
}