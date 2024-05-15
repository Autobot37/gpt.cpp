#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>

void attention_forward(float* out, float* preatt, float* att, float* qkv, int B, int T, int C, int NH){
    int hs = C/NH;
    float scale = 1.0 / sqrtf(hs);
    #pragma omp parallel for collapse(3) schedule(dynamic)
    for(int b = 0;b<B;b++){
        for(int t = 0;t<T;t++){
            for(int h = 0;h<NH;h++){
                
                //q @ k 
                float* query = qkv + b * T * 3 * C + t * 3 * C + h * hs;
                float* preatt_p = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_p = att + b*NH*T*T + h*T*T + t*T;

                float maxval = 1e-5f;
                #pragma omp simd reduction(max:maxval)
                for(int t2=0;t2<=t;t2++){
                    float* key = qkv + b * T * 3 * C + t2 * 3 * C + h*hs + C;
                    float val = 0.0f;
                    #pragma omp simd reduction(+:val)
                    for(int i = 0;i<hs;i++){
                        val += query[i] * key[i];
                    }
                    val *= scale;
                    if(val>maxval){
                        maxval = val;
                    }
                    preatt_p[t2] = val;
                }
                //softmax
                float sum = 0.0f;
                #pragma omp simd reduction(+:sum)
                for(int t2=0;t2<=t;t2++){
                    float val = expf(preatt_p[t2] - maxval);
                    att_p[t2] = val;
                    sum += val;
                }
                float expinv = (sum==0.0f) ? 0.0f : 1.0f/sum;
                #pragma omp simd
                for(int t2=0;t2<T;t2++){
                    if(t2<=t){
                        att_p[t2] *= expinv;
                    }
                    else{
                        att_p[t2] = 0.0f;
                    }
                }   
                //accumulating
                float* out_p = out + b*T*C + t*C + h*hs;
                #pragma omp simd
                for(int t2=0;t2<hs;t2++){
                    float val = 0.0f;
                    #pragma omp simd reduction(+:val)
                    for(int i = 0;i<T;i++){
                        float value = qkv[b*T*3*C + i*3*C + 2*C + h*hs + t2];
                        val += att_p[i] * value;
                    }
                    out_p[t2] = val;
                }
            }
        }
    }
} 
//ok first parellizing across only B and T
__global__ void preatt_kernel(float* preatt,float* maxvals, float* qkv, int B, int T, int C, int NH){
    int bt = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int t2 = blockIdx.z * blockDim.z + threadIdx.z;
    int b = bt / T;
    int t = bt % T;
    if(b>=B || t>=T || h>=NH || t2>=T){
        return;
    }
    int hs = C/NH;
    float scale = 1.0 / sqrtf(hs);
    float* query = qkv + b * T * 3 * C + t * 3 * C + h * hs;
    float* preatt_p = preatt + b*NH*T*T + h*T*T + t*T;
    float* key = qkv + b * T * 3 * C + t2 * 3 * C + h*hs + C;
    float* maxval = maxvals + b*NH*T + t*T + h;

    if(t2>t){
        preatt_p[t2] = 0.0f;
    }
    //key @ query 
    float val = 0.0f;
    for(int i = 0;i<hs;i++){
        val += query[i] * key[i];
    }
    val *= scale;
    if(val>(*maxval)){
        *maxval = val;
    }
    preatt_p[t2] = val;
}

void preatt_gpu(float* preatt,float* maxval, float* qkv, int B, int T, int C, int NH){
    dim3 threads(8,8,8);
    dim3 blocks;
    blocks.x = (B*T+7)/8;
    blocks.y = (NH+7)/8;
    blocks.z = (T+7)/8;
    preatt_kernel<<<blocks, threads>>>(preatt, maxval, qkv, B, T, C, NH);
}


#define SHMEM_SIZE 1024
__device__ void warpReduce(volatile float* shmem_ptr, int t, int N) {
    if (t < N) {
        shmem_ptr[t] = fmaxf(shmem_ptr[t], shmem_ptr[t + 32]);
        shmem_ptr[t] = fmaxf(shmem_ptr[t], shmem_ptr[t + 16]);
        shmem_ptr[t] = fmaxf(shmem_ptr[t], shmem_ptr[t + 8]);
        shmem_ptr[t] = fmaxf(shmem_ptr[t], shmem_ptr[t + 4]);
        shmem_ptr[t] = fmaxf(shmem_ptr[t], shmem_ptr[t + 2]);
        shmem_ptr[t] = fmaxf(shmem_ptr[t], shmem_ptr[t + 1]);
    }
}

__device__ void max_reduction(float* result, float* arr, int N) {
    __shared__ float partial_max[SHMEM_SIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    if (tid < N) {
        partial_max[threadIdx.x] = fmaxf(arr[i], arr[i + blockDim.x]);
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            if (tid + s < N) {
                partial_max[threadIdx.x] = fmaxf(partial_max[threadIdx.x], partial_max[threadIdx.x + s]);
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        warpReduce(partial_max, threadIdx.x, N);
    }

    if (threadIdx.x == 0 && tid < N) {
        result[blockIdx.x] = partial_max[0];
    }
}

_device__ void warpReduceSum(volatile float* shmem_ptr, int t, int N) {
    if (t < N) {
        shmem_ptr[t] += (shmem_ptr[t + 32]);
        shmem_ptr[t] += (shmem_ptr[t + 16]);
        shmem_ptr[t] += (shmem_ptr[t + 8]);
        shmem_ptr[t] += (shmem_ptr[t + 4]);
        shmem_ptr[t] += (shmem_ptr[t + 2]);
        shmem_ptr[t] += (shmem_ptr[t + 1]);
    }
}

__device__ void sum_reduction(float* result, float* arr, int N) {
    __shared__ float partial_max[SHMEM_SIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    if (tid < N) {
        partial_max[threadIdx.x] = (expf(arr[i]) + expf(arr[i + blockDim.x]));
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            if (tid + s < N) {
                partial_max[threadIdx.x] = (expf(partial_max[threadIdx.x]) + expf(partial_max[threadIdx.x + s]));
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        warpReduceSum(partial_max, threadIdx.x, N);
    }

    if (threadIdx.x == 0 && tid < N) {
        result[blockIdx.x] = partial_max[0];
    }
}

__global__ void softmax_kernel(float* att, float* preatt, int B, int T, int C, int NH){
    int bt = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int t2 = blockIdx.z * blockDim.z + threadIdx.z;
    int b = bt / T;
    int t = bt % T;
    if(b>=B || t>=T || h>=NH || t2>=T){
        return;
    }
    float* preatt_p = preatt + b*NH*T*T + h*T*T + t*T;
    float* att_p = att + b*NH*T*T + h*T*T + t*T;
    //warpreduce for maxval from preatt_p
    float maxval = 0;
    int tb_size = 1024;
    int grid_size = (T+tb_size-1)/tb_size/2;
    float* maxvals;
    cudaMalloc(&maxvals, tb_size*sizeof(float));
    max_reduction<<<tb_size, grid_size>>>(tb_size, preatt_p, T);
    max_reduction<<<1, tb_size>>>(maxvals, maxvals, tb_size);
    maxval = maxvals[0];
    //sum reduction
    float sum = 0;
    int tb_size = 1024;
    int grid_size = (T+tb_size-1)/tb_size/2;
    float* maxvals;
    cudaMalloc(&maxvals, tb_size*sizeof(float));
    max_reduction<<<tb_size, grid_size>>>(tb_size, preatt_p, T);
    max_reduction<<<1, tb_size>>>(maxvals, maxvals, tb_size);
    float expinv = (sum==0.0f) ? 0.0f : 1.0f/sum;
    for(int t2=0;t2<T;t2++){
        if(t2<=t){
            att_p[t2] *= expinv;
        }
        else{
            att_p[t2] = 0.0f;
        }
    }   
}




__global__ void accumulate_kernel(){

}


void accumulate(){

}

__global__ void attention_forward_kernel(float* out, float* preatt, float* att, float* qkv, int B, int T, int C, int NH){
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.z * blockDim.z + threadIdx.z;
    if(b>=B || t>=T || h>=NH){
        return;
    }
    int hs = C/NH;
    float scale = 1.0 / sqrtf(hs);

    float* query = qkv + b * T * 3 * C + t * 3 * C + h * hs;
    float* preatt_p = preatt + b*NH*T*T + h*T*T + t*T;
    float* att_p = att + b*NH*T*T + h*T*T + t*T;
    //key @ query 
    float maxval = 1e-5f;
    for(int t2=0;t2<=t;t2++){
        float* key = qkv + b * T * 3 * C + t2 * 3 * C + h*hs + C;
        float val = 0.0f;
        for(int i = 0;i<hs;i++){
            val += query[i] * key[i];
        }
        val *= scale;
        if(val>maxval){
            maxval = val;
        }
        preatt_p[t2] = val;
    }
    //softmax
    float sum = 0.0f;
    for(int t2=0;t2<=t;t2++){
        float val = expf(preatt_p[t2] - maxval);
        att_p[t2] = val;
        sum += val;
    }
    float expinv = (sum==0.0f) ? 0.0f : 1.0f/sum;
    for(int t2=0;t2<T;t2++){
        if(t2<=t){
            att_p[t2] *= expinv;
        }
        else{
            att_p[t2] = 0.0f;
        }
    }   
    //accumulating
    float* out_p = out + b*T*C + t*C + h*hs;
    for(int t2=0;t2<hs;t2++){
        float val = 0.0f;
        for(int i = 0;i<T;i++){
            float value = qkv[b*T*3*C + i*3*C + 2*C + h*hs + t2];
            val += att_p[i] * value;
        }
        out_p[t2] = val;
    }
}
void attention_forward_gpu(float* out, float* preatt, float* att, float* qkv, int B, int T, int C, int NH){
    dim3 threads(8,8,8);
    dim3 blocks((B+7)/8, (T+7)/8, (NH+7)/8);
    attention_forward_kernel<<<blocks, threads>>>(out, preatt, att, qkv, B, T, C, NH);
}

void rand_init(float* arr, int size){
    for(int i = 0;i<size;i++){
        arr[i] = (float)rand() / RAND_MAX;
    }
}
void print_2d(float* arr, int r, int c){
    for(int i=0;i<r;i++){
        for(int j=0;j<c;j++){
            printf("%f ", arr[i*c+j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(){

    int mul = 4;
    int B = 1*mul;
    int T = 128*mul;
    int C = 128*mul;
    int OC = 128*mul;
    int NH = 4*mul;

    float *preatt, *att, *qkv, *out;
    preatt = (float*)malloc(B*NH*T*T*sizeof(float));
    att = (float*)malloc(B*NH*T*T*sizeof(float));
    qkv = (float*)malloc(B*T*3*C*sizeof(float));
    out = (float*)malloc(B*T*C*sizeof(float));
    rand_init(preatt, B*NH*T*T);
    rand_init(att, B*NH*T*T);
    rand_init(qkv, B*T*3*C);

    srand(time(NULL));
    clock_t start, end;
    double time_used;
    start = clock();
    attention_forward(out, preatt, att, qkv, B, T, C, NH);
    end = clock();
    time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("CPU Time: %f\n", time_used);

    //gpu
    float *d_preatt, *d_att, *d_qkv, *d_out;
    cudaMalloc(&d_preatt, B*NH*T*T*sizeof(float));
    cudaMalloc(&d_att, B*NH*T*T*sizeof(float));
    cudaMalloc(&d_qkv, B*T*3*C*sizeof(float));
    cudaMalloc(&d_out, B*T*C*sizeof(float));

    cudaMemcpy(d_preatt, preatt, B*NH*T*T*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_att, att, B*NH*T*T*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qkv, qkv, B*T*3*C*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);

    attention_forward_gpu(d_out, d_preatt, d_att, d_qkv, B, T, C, NH);
    
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);

    time_used = milliseconds / 1000.0;
    printf("GPU Time: %f\n", time_used);

    float* check_out;
    check_out = (float*)malloc(B*T*C*sizeof(float));
    cudaMemcpy(check_out, d_out, B*T*C*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i=0;i<B*T*C;i++){
        if(abs(out[i] - check_out[i]) > 1e-3f){
            printf("Incorrect output Try Again!\n");
            return 0;
        }
    }
    printf("Correct output Yay!\n");

    //preatt kernel
    
    float *d_maxval;
    cudaMalloc(&d_maxval, B*NH*T*sizeof(float));
    float* another_d_preatt;
    cudaMalloc(&another_d_preatt, B*NH*T*T*sizeof(float));

    cudaEvent_t start_gpu2, stop_gpu2;
    cudaEventCreate(&start_gpu2);
    cudaEventCreate(&stop_gpu2);
    cudaEventRecord(start_gpu2);

    preatt_gpu(another_d_preatt, d_maxval, d_qkv, B, T, C, NH);
    cudaDeviceSynchronize();
    cudaEventRecord(start_gpu2);
    cudaEventSynchronize(stop_gpu2);
    float milliseconds2 = 0;

    cudaEventElapsedTime(&milliseconds2, start_gpu2, stop_gpu2);
    printf("GPU Preatt Time: %f\n", milliseconds2/ 1000.0);

    float* preatt_check1;
    preatt_check1 = (float*)malloc(B*NH*T*T*sizeof(float));
    cudaMemcpy(preatt_check1, d_preatt, B*NH*T*T*sizeof(float), cudaMemcpyDeviceToHost);
    float* preatt_check2;
    preatt_check2 = (float*)malloc(B*NH*T*T*sizeof(float));
    cudaMemcpy(preatt_check2, another_d_preatt, B*NH*T*T*sizeof(float), cudaMemcpyDeviceToHost);

    for(int b=0;b<B;b++){
        for(int h=0;h<NH;h++){
            for(int t=0;t<T;t++){
                for(int t2=0;t2<=t;t2++){
                    if(abs(preatt_check1[b*NH*T*T + h*T*T + t*T + t2] - preatt_check2[b*NH*T*T + h*T*T + t*T + t2]) > 1e-3f){
                        printf("Incorrect preatt output Try Again!\n");
                        return 0;
                    }
                }
            }
        }
    }
    printf("Correct preatt output Yay!\n");

    //softmax
    //

    return 0;
}