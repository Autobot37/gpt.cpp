#include <cuda_runtime.h>
#include <iostream>
using namespace std;

__global__
void add_kernel(float* out, float* in, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        out[index] = in[index] + 1;
    }
}

void add(float* out, float* in, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(out, in, n);
}

int main(){

    int n = 8;
    float* arr = (float*)malloc(n * sizeof(float));

    float* in,*out;
    cudaMalloc((void**)&in, n * sizeof(float));
    cudaMalloc((void**)&out, n * sizeof(float));

    for (int i = 0; i < n; i++) {
        arr[i] = i;
    }
    cudaMemcpy(in, arr, n * sizeof(float), cudaMemcpyHostToDevice);

    add(out, in, n);

    cudaMemcpy(arr, out, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    return 0;
}