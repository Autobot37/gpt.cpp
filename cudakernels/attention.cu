#include <cuda_runtime.h>
#include <iostream>
#include <cublas_v2.h>
#include <float.h>
#include <assert.h>
using namespace std;

void rand_init(float* x, int n){
    for(int i = 0;i<n;i++){
        x[i] = (float)rand() / RAND_MAX;
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
            cout << "Mismatches" << endl;
            for(int j = i;j<min(n, i+10);j++){
                cout << a[j] << " " << b[j] << endl;
            }
            return;
        }
    }
    cout << "Results match " << endl;
    for(int i = 0;i<4;i++){
        cout << a[i] << " " << b[i] << endl;
    }
}

void softmax(float* x, int N){
    float max = x[0];
    for(int i = 1;i<N;i++){
        if(x[i] > max){
            max = x[i];
        }
    }
    float sum = 0;
    for(int i = 0;i<N;i++){
        x[i] = exp(x[i] - max);
        sum += x[i];
    }
    for(int i = 0;i<N;i++){
        x[i] /= sum;
    }
}

void attention(float* out, float* att, float* qkv, float* key_cache, float* value_cache, int l, int pos, int C, int NH, int T){

    int head_size = C / NH;

    float* q = qkv;
    memcpy(key_cache + l * C * T + pos * C, qkv + C, C * sizeof(float));
    memcpy(value_cache + l * C * T + pos * C, qkv + 2*C, C * sizeof(float));

    float scale = 1.0 / sqrt(head_size);

    float* k = key_cache + l * C * T;
    float* v = value_cache + l * C * T;

    int h;
    #pragma omp parallel for private(h)
    for(h = 0;h<NH;h++){

        float* qh = q + h * head_size;
        float* atth = att + h * T;

        for(int t = 0;t<T;t++){
            float* kh = k + t * C + h * head_size;
            float score = 0.0f;
            for(int i = 0;i<head_size;i++){
                score += qh[i] * kh[i];
            }
            score *= scale;
            atth[t] = score;
        }
        for(int t=pos+1;t<T;t++){
            atth[t] = -FLT_MAX;
        }

        softmax(atth, T);

        float* outh = out + h * head_size;
        memset(outh, 0, head_size * sizeof(float));
        for(int t = 0;t<T;t++){
            float* vh = v + t * C + h * head_size;
            float score = atth[t];
            for(int i = 0;i<head_size;i++){
                outh[i] += score * vh[i];
            }
        }
    }
}

/*
Plan : 
qkv is shape of 3 * C
fill qkv_k into key_cache
fill qkv_v into value_cache
now the problem is caches are of size L * T * NH * HS
but we need caches to be of size L * NH * T * HS
so we directly copy permuted qkv to caches for every position for every layer
this way caches remain in shape L * NH * T * HS
simply do cublas gemm to get the attention scores.
*/

__global__
void fill_cache(float* key_cache, float* value_cache, float* arr, int l, int pos, int C, int NH, int T){
    //arr to cache
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < C){
        int head_size = C / NH;
        int nh = idx / head_size;
        int h = idx % head_size;
        float* key_cache_l = key_cache + l * T * NH * head_size;
        float* value_cache_l = value_cache + l * T * NH * head_size;
        //from-> NH HS 
        //to -> NH T HS
        int from = nh * head_size + h;
        int to = nh * T * head_size + pos * head_size + h;
        key_cache_l[to] = arr[from + C];
        value_cache_l[to] = arr[from + 2*C];
    }
}

__global__ 
void unfill_cache(float* cache, float* arr, int l, int pos, int C, int NH, int T){
    //cache to arr
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < C){
        int head_size = C / NH;
        int nh = idx / head_size;
        int h = idx % head_size;
        float* cache_l = cache + l * T * NH * head_size;
        //from-> NH T HS 
        //to -> NH HS
        int to = nh * head_size + h;
        int from = nh * T * head_size + pos * head_size + h;
        arr[to] = cache_l[from];
    }
}
__global__ 
void softmax_kernel2(float* x, int T, int NH, int pos, float scale){
    //requires launch <<<NH, 1024>>>
    int row = blockIdx.x;
    int idx = threadIdx.x;
    __shared__ float row_max[1024];
    __shared__ float row_sum[1024];
    row_max[threadIdx.x] = -FLT_MAX;
    row_sum[threadIdx.x] = 0;
    __syncthreads();
    

    for(int i = idx;i<pos;i+=blockDim.x){
        x[row * T + i] *= scale;
        float val = x[row * T + i];
        row_max[idx] = fmaxf(row_max[idx], val);
    }
    __syncthreads();
    if(idx==0){
        float maxval = -FLT_MAX;
        for(int i = 0;i<min(pos, blockDim.x);i++){
            maxval = fmaxf(maxval, row_max[i]);
        }
        row_max[0] = maxval;
    }
    __syncthreads();
    float maxval = row_max[0];
    
    for(int i = idx;i<pos;i+=blockDim.x){
        x[row * T + i] = expf(x[row * T + i] - maxval);
        row_sum[idx] += x[row * T + i];
    }
    __syncthreads();
    if(idx==0){
        float sum = 0;
        for(int i = 0;i<min(pos,blockDim.x);i++){
            sum += row_sum[i];
        }
        row_sum[0] = sum;
    }
    __syncthreads();
    float sum = row_sum[0];
    for(int i = idx;i<T;i+=blockDim.x){
        if(i<pos){
            x[row * T + i] /= sum;
        }
        else{
            x[row * T + i] = 0;
        }
    }
}

__device__ float warpReduceMax(float val){
    for(int offset = 16; offset > 0; offset /= 2){
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}
__device__ float warpReduceSum(float val){
    for(int offset = 16; offset > 0; offset /= 2){
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__
void softmax_kernel3(float* x, int T, int NH, int pos, float scale){
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = tid / 32;
    int laneId = tid % 32;

    int warpsPerBlock = blockDim.x / 32;

    extern __shared__ float shared[];
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    float* x_h = x + idx * T;
    float maxval = -FLT_MAX;
    for(int i= tid;i<pos;i+=blockDim.x){
        x_h[i] *= scale;
        maxval = fmaxf(maxval, x_h[i]);
    }
    maxval = warpReduceMax(maxval);
    if(laneId == 0){
        maxvals[warpId] = maxval;
    }
    __syncthreads();
    if(tid==0){
        float val = -FLT_MAX;
        for(int i = 0;i<warpsPerBlock;i++){
            val = fmaxf(val, maxvals[i]);
        }
        maxvals[0] = val;
    }
    __syncthreads();
    float offset = maxvals[0];

    for(int i = tid;i<pos;i+=blockDim.x){
        x_h[i] = expf(x_h[i] - offset);
    }
    //sum
    float sumval = 0.0f;
    for(int i = tid;i<pos;i+=blockDim.x){
        sumval += x_h[i];
    }
    sumval = warpReduceSum(sumval);
    if(laneId == 0){
        sumvals[warpId] = sumval;
    }
    __syncthreads();
    if(tid==0){
        float val = 0;
        for(int i = 0;i<warpsPerBlock;i++){
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    float sum = sumvals[0];
    for(int i = tid;i<T;i+=blockDim.x){
        x_h[i] = (i<pos) ? x_h[i] / sum : 0;
    }
}

void attention_gpu(float* out, float* att, float* qkv, float* key_cache, float* value_cache, int l, int pos, int C, int NH, int T){

    int head_size = C / NH;
    int numThreads = 1024;
    int blocks = (C + numThreads - 1) / numThreads;
    fill_cache<<<blocks, numThreads>>>(key_cache, value_cache, qkv, l, pos, C, NH, T);

    float* q = qkv;
    float* k = key_cache + l * C * T;
    float* v = value_cache + l * C * T;
    //performing attention
    //q [NH, 1, HS] @ K [NH, T, HS].T -> [NH, 1, T]
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0;
    float beta = 0.0;
    cublasStatus_t status = cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, T, 1, head_size, &alpha, k, head_size, head_size * T, q, head_size, head_size, &beta, att, T, T, NH);
    if(status!=CUBLAS_STATUS_SUCCESS){
        cout << "CUBLAS error" << endl;
    }
    float scale = 1.0 / sqrt(head_size);    
    size_t memory = 2 * 1024 * sizeof(float) / 32;
    softmax_kernel3<<<NH, 1024, memory>>>(att, T, NH, pos+1, scale);

    status = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, head_size, 1, T, &alpha, v, head_size, head_size * T, att, T, T, &beta, out, head_size, head_size, NH);
    if(status!=CUBLAS_STATUS_SUCCESS){
        cout << "CUBLAS error" << endl;
    }
    cublasDestroy(handle);
}
int main(){
    
    int C = 1024;
    int NH = 16;
    int head_size = C / NH;
    int T = 1200;
    int L = 8;


    float* out, *att, *qkv, *key_cache, *value_cache;
    float* d_out, *d_att, *d_qkv, *d_key_cache, *d_value_cache;

    if(true){
        out = (float*)malloc(NH * head_size * sizeof(float));
        att = (float*)malloc(NH * T * sizeof(float));
        qkv = (float*)malloc(3 * C * sizeof(float));
        key_cache = (float*)malloc(L * NH * T * head_size * sizeof(float));
        value_cache = (float*)malloc(L * NH * T * head_size * sizeof(float));

        rand_init(qkv, 3 * C);
        memset(key_cache, 0, L * NH * T * head_size * sizeof(float));
        memset(value_cache, 0, L * NH * T * head_size * sizeof(float));

        cudaMalloc(&d_out, NH * head_size * sizeof(float));
        cudaMalloc(&d_att, NH * T * sizeof(float));
        cudaMalloc(&d_qkv, 3 * C * sizeof(float));
        cudaMalloc(&d_key_cache, L * NH * T * head_size * sizeof(float));
        cudaMalloc(&d_value_cache, L * NH * T * head_size * sizeof(float));

        cudaMemcpy(d_qkv, qkv, 3 * C * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_key_cache, key_cache, L * NH * T * head_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_value_cache, value_cache, L * NH * T * head_size * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    
    int l = 7;
    int pos = 1023;

    attention(out, att, qkv, key_cache, value_cache, l, pos, C, NH, T);
    attention_gpu(d_out, d_att, d_qkv, d_key_cache, d_value_cache, l, pos, C, NH, T);


    float* check_att = (float*)malloc(NH * T * sizeof(float));
    cudaMemcpy(check_att, d_att, NH * T * sizeof(float), cudaMemcpyDeviceToHost);
    isequal(att, check_att, NH * T);

    float* check_out = (float*)malloc(NH * head_size * sizeof(float));
    cudaMemcpy(check_out, d_out, NH * head_size * sizeof(float), cudaMemcpyDeviceToHost);
    isequal(out, check_out, NH * head_size);

    //softmax check;
    // T = 50324;
    // float scale = 1.0 / sqrt(head_size);
    // float* in1, *in2;
    // cudaMalloc(&in1, NH * T * sizeof(float));
    // cudaMalloc(&in2, NH * T * sizeof(float));

    // cudaMemcpy(in1, in2, NH * T * sizeof(float), cudaMemcpyDeviceToDevice);

    // softmax_kernel2<<<NH, 1024>>>(in1, T, NH, 11, scale);
    // size_t memory = 2 * 1024 * sizeof(float) / 32;
    // softmax_kernel3<<<NH, 1024, memory>>>(in2, T, NH, 11, scale);

    // float* check1 = (float*)malloc(NH * T * sizeof(float));
    // float* check2 = (float*)malloc(NH * T * sizeof(float));
    // cudaMemcpy(check1, in1, NH * T * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(check2, in2, NH * T * sizeof(float), cudaMemcpyDeviceToHost);

    // isequal(check1, check2, NH * T);


    return 0;
}