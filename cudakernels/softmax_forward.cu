#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

void softmax_forward(float* out, float* inp, int B, int T, int V){
    for(int b=0;b<B;b++){
        for(int t=0;t<T;t++){
            float* inp_p = inp + b * T * V + t * V;
            float* out_p = out + b * T * V + t * V;
            float max_val = -1e-5f;
            for(int i=0;i<V;i++){
                if(inp_p[i] > max_val){
                    max_val = inp_p[i];
                }
            }
            float sum = 0.0f;
            for(int i=0;i<V;i++){
                out_p[i] = expf(inp_p[i] - max_val);
                sum += out_p[i];
            }
            for(int i=0;i<V;i++){
                out_p[i] = out_p[i] / sum;
            }
        }
    }
}

__global__ void softmax_forward_kernel(float* out, float* inp, int B, int T, int C){
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    float* x = inp + idx * C;
    float* o = out + idx * C;

    float max_val = -1e-5f;
    for(int i=tid;i<C;i+=block_size){
        max_val = fmaxf(max_val, x[i]);
    }
    shared[tid] = max_val;
    __syncthreads();

    for(int stride = block_size/2;stride>0;stride/=2){
        if(tid < stride){
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    float offset = shared[0];

    for(int i=tid;i<C;tid+=block_size){
        o[i] = expf(x[i] - offset);
    }
    __syncthreads();

    float sumval = 0.0f;
    for(int i=tid;i<C;i+=block_size){
        sumval += o[i];
    }
    shared[tid] = sumval;
    __syncthreads();

    for(int stride = block_size/2;stride>0;stride/=2){
        if(tid < stride){
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float sum = shared[0];

    for(int i=tid;i<C;i+=block_size){
        o[i] = o[i] / sum;
    }
}

void softmax_forward_gpu(float* out, float* inp, int B, int T, int V){
    int numThreads = 64;
    int numBlocks = B*T;
    float sharedMem = numThreads * sizeof(float);
    //softmax_forward_kernel<<<numBlocks, numThreads, sharedMem>>>(out, inp, B, T, V);
}


void rand_init(float* arr, int size){
    for(int i = 0;i<size;i++){
        arr[i] = (float)rand() / RAND_MAX;
    }
}
void print_mat(float* arr, int size){
    for(int i =0;i<size;i++){
	printf("%f",arr[i]);
    }
}

int main(){

    int mul = 16;
    int B = 1*mul;
    int T = 4*mul;
    int V = 4*mul;

    float* inp = (float*)malloc(B*T*V*sizeof(float));
    float* out = (float*)malloc(B*T*V*sizeof(float));
    rand_init(inp, B*T*V);
    
    clock_t start, end;
    start = clock();
    softmax_forward(out, inp, B, T, V);
    end = clock();
    printf("Time taken by CPU: %f\n", (double)(end-start)/CLOCKS_PER_SEC);

    //gpu
    float* d_inp;
    float* d_out;
    cudaMalloc(&d_inp, B*T*V*sizeof(float));
    cudaMalloc(&d_out, B*T*V*sizeof(float));

    cudaMemcpy(d_inp, inp, B*T*V*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    start = clock();
    softmax_forward_gpu(d_out, d_inp, B, T, V);
    cudaDeviceSynchronize();
    end = clock();
    printf("Time taken by GPU: %f\n", (double)(end-start)/CLOCKS_PER_SEC);

    float* check = (float*)malloc(B*T*V*sizeof(float));
    cudaMemcpy(check, d_out, B*T*V*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for(int i=0;i<B*T*V;i++){
        if(fabs(out[i] - check[i]) > 1e-5f){
            printf("Incorrect output Try Again!\n");
            return 0;
        }
    }
    printf("Correct output Yay!\n");
    
    return 0;
}
