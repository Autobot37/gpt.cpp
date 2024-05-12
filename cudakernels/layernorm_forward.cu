#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

void layernorm_forward(float* out, float* mean, float* std_dev, float* inp, float* weight, float* bias, int B, int T, int C){
    float eps = 1e-5f;
    for(int b = 0;b<B;b++){
        for(int t = 0;t<T;t++){
            float* inp_p = inp + b*T*C + t*C;
            float* out_p = out + b*T*C + t*C;
                        
            float m = 0.0f;
            for(int i=0;i<C;i++){
                m += inp_p[i];
            }
            m = m/C;
            mean[b*T + t] = m;

            float v = 0.0f;
            for(int i = 0;i<C;i++){
                float diff = inp_p[i] - m;
                v += diff * diff;
            }
            v = v / C;
            float s = 1.0f/sqrtf(v + eps);
            std_dev[b*T + t] = s;

            for(int i = 0;i<C;i++){
                out_p[i] = ((inp_p[i] - m) * s) * weight[i] + bias[i];            
            }
        }    
    }
}

__global__ void layernorm_forward_kernel(float* out, float* mean, float* std_dev, float* inp, float* weight, float* bias, int B, int T, int C){
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    if(b>=B || t>=T){
        return;
    }
    float eps = 1e-5f;
    float* inp_p = inp + b*T*C + t*C;
    float* out_p = out + b*T*C + t*C;
                
    float m = 0.0f;
    for(int i=0;i<C;i++){
        m += inp_p[i];
    }
    m = m/C;
    mean[b*T + t] = m;

    float v = 0.0f;
    for(int i = 0;i<C;i++){
        float diff = inp_p[i] - m;
        v += diff * diff;
    }
    v = v / C;
    float s = 1.0f/sqrtf(v + eps);
    std_dev[b*T + t] = s;

    for(int i = 0;i<C;i++){
        out_p[i] = ((inp_p[i] - m) * s) * weight[i] + bias[i];            
    }
}

void layernorm_forward_gpu(float* out, float* mean, float* std_dev, float* inp, float* weight, float* bias, int B, int T, int C){
    dim3 threadsPerBlock(4, 256);
    dim3 numBlocks((B + threadsPerBlock.x - 1)/threadsPerBlock.x, (T + threadsPerBlock.y - 1)/threadsPerBlock.y);
    layernorm_forward_kernel<<<numBlocks, threadsPerBlock>>>(out, mean, std_dev, inp, weight, bias, B, T, C);
    cudaDeviceSynchronize();
}

void rand_init(float* arr, int size){
    for(int i = 0;i<size;i++){
        arr[i] = (float)rand() / RAND_MAX;
    }
}

int main(){

    int mul = 4;
    int B = 4*mul;
    int T = 128*mul;
    int C = 128*mul;

    float* inp = (float*)malloc(B*T*C*sizeof(float));
    float* out = (float*)malloc(B*T*C*sizeof(float));
    float* mean = (float*)malloc(B*T*sizeof(float));
    float* std_dev = (float*)malloc(B*T*sizeof(float));
    float* weight = (float*)malloc(C*sizeof(float));
    float* bias = (float*)malloc(C*sizeof(float));
    rand_init(inp, B*T*C);
    rand_init(weight, C);
    rand_init(bias, C);

    clock_t start, end;
    double time_used;
    start = clock();
    layernorm_forward(out, mean, std_dev, inp, weight, bias, B, T, C);
    end = clock();
    time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("CPU Time used: %f\n", time_used);

    //GPU
    float *d_inp, *d_out, *d_mean, *d_std_dev, *d_weight, *d_bias;
    cudaMalloc(&d_inp, B*T*C*sizeof(float));
    cudaMalloc(&d_out, B*T*C*sizeof(float));
    cudaMalloc(&d_mean, B*T*sizeof(float));
    cudaMalloc(&d_std_dev, B*T*sizeof(float));
    cudaMalloc(&d_weight, C*sizeof(float));
    cudaMalloc(&d_bias, C*sizeof(float));

    cudaMemcpy(d_inp, inp, B*T*C*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, C*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, C*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    start = clock();
    layernorm_forward_gpu(d_out, d_mean, d_std_dev, d_inp, d_weight, d_bias, B, T, C);
    end = clock();
    time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("GPU Time used: %f\n", time_used);

    float* check;
    check = (float*)malloc(B*T*C*sizeof(float));
    cudaMemcpy(check, d_out, B*T*C*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i=0;i<B*T*C;i++){
        if(abs(out[i] - check[i]) > 1e-3f){
            printf("Incorrect Output Try Again!\n");
            return 0;
        }
    }
    printf("And its Correct too! Yay!\n");

    return 0;
}