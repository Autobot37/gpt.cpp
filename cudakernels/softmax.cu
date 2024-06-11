#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

void rand_init(float* x, int size){
    for(int i = 0;i<size;i++){
        x[i] = (float)rand()/(float)RAND_MAX;
    }
}

void print(float* x, int size){
    for(int i = 0;i<size;i++){
        printf("%f ", x[i]);
    }
    printf("\n");
}

void softmax_1d(float* x, int size){
    float max = x[0];
    for(int i = 1;i<size;i++){
        if(x[i]>max){
            max = x[i];
        }
    }
    float sum = 0.0f;
    for(int i = 0;i<size;i++){
        x[i] = expf(x[i]-max);
        sum += x[i];
    }
    for(int i = 0;i<size;i++){
        x[i] /= sum;
    }
}

__global__
void softmax_1d_kernel(float* x, int size){
    int idx = threadIdx.x;
    __shared__ float sdata[2048];
    sdata[idx] = 0.0f;
    sdata[idx + blockDim.x] = 0.0f;
    __syncthreads();

    for(int i = idx;i<size;i+=blockDim.x){
        sdata[idx] = fmaxf(sdata[idx], x[i]);
    }
    __syncthreads();
    if(idx==0){
        float val = 0.0f;
        for(int i = 0;i<blockDim.x;i++){
            val = fmaxf(val, sdata[i]);
        }
        sdata[0] = val;
    }
    float maxval = sdata[0];
    
    float* sum_sdata = sdata + blockDim.x;
    for(int i = idx;i<size;i+=blockDim.x){
        x[i] = expf(x[i]-maxval);
        sum_sdata[idx] += x[i];
    }
    __syncthreads();
    if(idx==0){
        float sum = 0.0f;
        for(int i = 0;i<blockDim.x;i++){
            sum += sum_sdata[i];
        }
        sdata[0] = sum;
    }
    __syncthreads();
    float sum = sdata[0];
    for(int i = idx;i<size;i+=blockDim.x){
        x[i] /= sum; 
    }    
}

void softmax_1d_gpu(float* x, int size){
    int numthreads = 1024;
    softmax_1d_kernel<<<1,numthreads>>>(x, size);
}

int main(){

    float* x, *d_x;
    int size = 50324;
    x = (float*)malloc(size*sizeof(float));
    rand_init(x, size);
    cudaMalloc(&d_x, size*sizeof(float));
    cudaMemcpy(d_x, x, size*sizeof(float), cudaMemcpyHostToDevice);


    clock_t start, end;
    start = clock();
    softmax_1d(x, size);
    end = clock();
    printf("CPU Time: %f\n", (float)(end-start)/CLOCKS_PER_SEC);

    cudaEvent_t start_gpu, end_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);
    cudaEventRecord(start_gpu);
    softmax_1d_gpu(d_x, size);
    cudaEventRecord(end_gpu);
    cudaEventSynchronize(end_gpu);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_gpu, end_gpu);
    printf("GPU Time: %f\n", milliseconds/1000);

    float* check = (float*)malloc(size*sizeof(float));
    cudaMemcpy(check, d_x, size*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0;i<size;i++){
        if(fabs(x[i]-check[i])>1e-6){
            printf("Mismatch at %d: %f %f\n", i, x[i], check[i]);
        }
    }
    
    return 0;
}