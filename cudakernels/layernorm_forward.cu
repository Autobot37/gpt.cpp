#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

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
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);  
    int idx = blockIdx.x *  warp.meta_group_size() + warp.meta_group_rank();
    int N = B * T;
    if(idx >= N){
        return;
    }
    float* x = inp + idx * C;

    float sum = 0.0f;
    for(int i = warp.thread_rank();i<C;i+=warp.size()){
        sum += x[i];
    }
    sum = cg::reduce(warp, sum, cg::plus<float>());
    float m = sum / C;
    if(warp.thread_rank == 0 && mean != nullptr){
        __stcs(mean + idx, m);
    }

    //rstd
    sum = 0.0f;
    for(int i = warp.thread_rank();i<C;i+=warp.size()){
        float diff = x[i] - m;
        sum += diff * diff;
    }
    sum = cg::reduce(warp, sum, cg::plus<float>());
    float s = rsqrtf(sum/C + 1e-5f);
    if(warp.thread_rank == 0 && std_dev != nullptr){
        __stcs(std_dev + idx, s);
    }
    float* o = out + idx * C;

    for(int c = warp.thread_rank();c<C;c+=warp.size()){
        float n = s * (__ldcs(x + c) - m);
        __stcs(o+c, n * weight[c] + bias[c]);
    }
}

void layernorm_forward_gpu(float* out, float* mean, float* std_dev, float* inp, float* weight, float* bias, int B, int T, int C){
    int N = B * T;
    int numThreads = 512;
    int grid_size = (N * 32 + numThreads - 1) / numThreads;
    layernorm_forward_kernel<<<grid_size, numThreads>>>(out, mean, std_dev, inp, weight, bias, B, T, C);
}

void rand_init(float* arr, int size){
    for(int i = 0;i<size;i++){
        arr[i] = (float)rand() / RAND_MAX;
    }
}

int main(){

    int B = 4;
    int T = 1024;
    int C = 2048;

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