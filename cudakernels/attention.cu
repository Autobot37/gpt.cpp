#include <cuda_runtime.h>
#include <iostream>
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
            cout << "Mismatch at index " << i << " CPU: " << a[i] << " GPU: " << b[i] << endl;
        }
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

void attention(float* out, float* att, float* qkv, float* key_cache, float* value_cache, int l, int pos, int C, int NH, int head_size, int T){

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

        for(int t = 0;t<=pos;t++){
            float* kh = k + t * C + h * head_size;
            float score = 0.0f;
            for(int i = 0;i<head_size;i++){
                score += qh[i] * kh[i];
            }
            score *= scale;
            atth[t] = score;
        }
        for(int t=pos+1;t<T;t++){
            atth[t] = -INFINITY;
        }

        softmax(atth, T);

        float* outh = out + h * head_size;
        memset(outh, 0, head_size * sizeof(float));
        for(int t = 0;t<=pos;t++){
            float* vh = v + t * C + h * head_size;
            float score = atth[t];
            for(int i = 0;i<head_size;i++){
                outh[i] += score * vh[i];
            }
        }
    }
}

__device__ void softmaxg(float* x, int N){
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

__global__
void attention_kernel(float* out, float* att, float* qkv, float* key_cache, float* value_cache, int l, int pos, int C, int NH, int head_size, int T){

    int h = threadIdx.x + blockIdx.x * blockDim.x;
    if(h >= NH){
        return;
    }

    float* q = qkv;
    float scale = 1.0 / sqrt(head_size);

    float* k = key_cache + l * C * T;
    float* v = value_cache + l * C * T;


    float* qh = q + h * head_size;
    float* atth = att + h * T;

    for(int t = 0;t<=pos;t++){
        float* kh = k + t * C + h * head_size;
        float score = 0.0f;
        for(int i = 0;i<head_size;i++){
            score += qh[i] * kh[i];
        }
        score *= scale;
        atth[t] = score;
    }
    for(int t=pos+1;t<T;t++){
        atth[t] = -INFINITY;
    }

    softmaxg(atth, T);

    float* outh = out + h * head_size;
    for(int t = 0;t<=pos;t++){
        float* vh = v + t * C + h * head_size;
        float score = atth[t];
        for(int i = 0;i<head_size;i++){
            outh[i] += score * vh[i];
        }
    }
}

void attention_gpu(float* out, float* att, float* qkv, float* key_cache, float* value_cache, int l, int pos, int C, int NH, int head_size, int T){
    int numThreads = 1024;
    int blocks = (NH + numThreads - 1) / numThreads;
    cudaMemcpy(key_cache + l * C * T + pos * C, qkv + C, C * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(value_cache + l * C * T + pos * C, qkv + 2*C, C * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemset(out, 0, C * sizeof(float));
    attention_kernel<<<blocks, numThreads>>>(out, att, qkv, key_cache, value_cache, l, pos, C, NH, head_size, T);
}

int main(){
    
    int l = 0;
    int pos = 0;
    int C = 768;
    int NH = 12;
    int head_size = 64;
    int T = 512;
    int L = 12;
    float* out = (float*)malloc(NH * head_size * sizeof(float));
    float* att = (float*)malloc(NH * T * sizeof(float));
    float* qkv = (float*)malloc(3 * C * sizeof(float));
    float* key_cache = (float*)malloc(L * C * T * sizeof(float));
    float* value_cache = (float*)malloc(L * C * T * sizeof(float));

    rand_init(qkv, 3 * C);
    rand_init(key_cache, L * C * T);
    rand_init(value_cache, L * C * T);

    attention(out, att, qkv, key_cache, value_cache, l, pos, C, NH, head_size, T);

    float* d_out, *d_att, *d_qkv, *d_key_cache, *d_value_cache;
    cudaMalloc(&d_out, NH * head_size * sizeof(float));
    cudaMalloc(&d_att, NH * T * sizeof(float));
    cudaMalloc(&d_qkv, 3 * C * sizeof(float));
    cudaMalloc(&d_key_cache, L * C * T * sizeof(float));
    cudaMalloc(&d_value_cache, L * C * T * sizeof(float));

    cudaMemcpy(d_qkv, qkv, 3 * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key_cache, key_cache, L * C * T * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value_cache, value_cache, L * C * T * sizeof(float), cudaMemcpyHostToDevice);

    attention_gpu(d_out, d_att, d_qkv, d_key_cache, d_value_cache, l, pos, C, NH, head_size, T);

    float* check = (float*)malloc(NH * head_size * sizeof(float));
    cudaMemcpy(check, d_out, NH * head_size * sizeof(float), cudaMemcpyDeviceToHost);

    isequal(out, check, NH * head_size);


    return 0;
}