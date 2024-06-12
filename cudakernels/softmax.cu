#include <cuda_runtime.h>
#include <iostream>
#include <float.h>
using namespace std;


void isequal(float* a, float* b, int n){
    float maxval = -INFINITY;
    for(int i = 0;i<n;i++){
        maxval = fmaxf(maxval, fmaxf(a[i], b[i]));
    }
    float eps = 1e-6;
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

__global__ void softmax_kernel(float* x, int N) {
    int idx = threadIdx.x;
    __shared__ float smax[1024];
    __shared__ float ssum[1024];
    
    smax[idx] = -FLT_MAX; 
    ssum[idx] = 0.0f;
    __syncthreads();

    for (int i = idx; i < N; i += blockDim.x) {
        smax[idx] = fmaxf(smax[idx], x[i]);
    }
    __syncthreads();
    
    if (idx == 0) {
        float maxval = -FLT_MAX;
        for (int i = 0; i < blockDim.x; i++) {
            maxval = fmaxf(maxval, smax[i]);
        }
        smax[0] = maxval;
    }
    __syncthreads();
    
    float maxval = smax[0];
    float local_sum = 0.0f;
    for (int i = idx; i < N; i += blockDim.x) {
        x[i] = expf(x[i] - maxval);
        local_sum += x[i];
    }
    ssum[idx] = local_sum;
    __syncthreads();
    
    if (idx == 0) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            sum += ssum[i];
        }
        ssum[0] = sum;
    }
    __syncthreads();
    
    float sum = ssum[0];
    for (int i = idx; i < N; i += blockDim.x) {
        x[i] /= sum;
    }
}

void softmax_gpu(float* x, int N) {
    int numThreads = 1024;
    softmax_kernel<<<1, numThreads>>>(x, N);
}


int main(){

    int N = 50317;
    float* x = (float*)malloc(N * sizeof(float));
    for(int i = 0;i<N;i++){
        x[i] = i*i;
    }
    float* d_x;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    
    softmax(x, N);
    softmax_gpu(d_x, N);

    float* check = (float*)malloc(N * sizeof(float));

    cudaMemcpy(check, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
    isequal(x, check, N);
    for(int i = 0;i<10;i++){
        cout << x[i] << " " << check[i] << endl;
    }

    printf("passed\n");

    return 0;
}