#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <cublas_v2.h>

__global__ void matmul_gpu_kernel(float* out, float* inp, float* weight, float* bias, int B, int T, int C, int OC){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int bt = bx * blockDim.x + tx;
    int oc = by * blockDim.y + ty;

    if(bt<B*T && oc<OC){
        float* inp_p = inp + bt * C;
        float* weight_p = weight + oc * C;

        float val = (bias != NULL) ? bias[oc] : 0.0f;
        for(int i = 0;i<C;i++){
            val += inp_p[i] * weight_p[i];
        }
        out[bt * OC + oc] = val;
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
            for(int o=0;o<OC;o++){
                float* weight_p = weight + o * C;
                float val = (bias != NULL) ? bias[o] : 0.0f;
                #pragma omp simd reduction(+:val) 
                for(int i = 0;i<C;i++){
                    val += inp_p[i] * weight_p[i];
                }
                out_p[o] = val;
            }
        }
    }
}

__global__ void add_bias(float* out, const float* bias, int B, int T, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < OC * B * T) {
        int oc = idx / (B * T);
        out[idx] += bias[oc];
    }
}

void matmul_forward2(float* d_out,
                     const float* d_inp, const float* d_weight, const float* d_bias,
                     int B, int T, int C, int OC) {
    int sqrt_block_size = 32;
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, B * T, C,
                             &alpha, d_weight, C, d_inp, C, &beta, d_out, OC);
    if (d_bias != nullptr) {
        int block_size = sqrt_block_size * sqrt_block_size;
        int grid_size = (OC * B * T + block_size - 1) / block_size;
        add_bias<<<grid_size, block_size>>>(d_out, d_bias, B, T, OC);
    }
    cublasDestroy(cublas_handle);
}


void rand_init(float* arr, int size){
    for(int i = 0;i<size;i++){
        arr[i] = (float)rand() / RAND_MAX;
    }
}
void print_3d(float* arr, int B, int T, int C){
    for(int b = 0;b<B;b++){
        for(int t = 0;t<T;t++){
            for(int c = 0;c<C;c++){
                printf("%f ", arr[b*T*C + t*C + c]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

int main(){

    int mul = 1;
    int B = 1*mul;
    int T = 4*mul;
    int C = 4*mul;
    int OC = 4*mul;

    float *inp, *weight, *bias, *out;
    float *d_inp, *d_weight, *d_bias, *d_out;
    inp = (float*)malloc(B*T*C*sizeof(float));
    weight = (float*)malloc(OC*C*sizeof(float));
    bias = (float*)malloc(OC*sizeof(float));
    out = (float*)malloc(B*T*OC*sizeof(float));
    rand_init(inp, B*T*C);
    rand_init(weight, OC*C);
    rand_init(bias, OC);
    float* check;
    check = (float*)malloc(B*T*OC*sizeof(float));

    cudaMalloc(&d_inp, B*T*C*sizeof(float));
    cudaMalloc(&d_weight, OC*C*sizeof(float));
    cudaMalloc(&d_bias, OC*sizeof(float));
    cudaMalloc(&d_out, B*T*OC*sizeof(float));

    cudaMemset(d_inp, 0, B*T*C*sizeof(float));
    cudaMemset(d_weight, 0, OC*C*sizeof(float));
    cudaMemset(d_bias, 0, OC*sizeof(float));
    cudaMemset(d_out, 0, B*T*OC*sizeof(float));

    cudaMemcpy(d_inp, inp, B*T*C*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, OC*C*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, OC*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    float* check_input;
    check_input = (float*)malloc(B*T*C*sizeof(float));
    cudaMemcpy(check_input, d_inp, B*T*C*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();


    for(int i = 0;i<B*T*C;i++){
        if(abs(inp[i] - check_input[i]) > 1e-3f){
            printf("Incorrect even input man Try again!\n");
            return 0;
        }
    }

    clock_t start, mid, end;
    double cpu_time_used, gpu_time_used;
    start = clock();
    matmul_forward(out, inp, weight, bias, B, T, C, OC);
    mid = clock();
    cpu_time_used = ((double) (mid - start)) / CLOCKS_PER_SEC;

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);

    matmul_forward_gpu(d_out, d_inp, d_weight, d_bias, B, T, C, OC);

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);

    cudaMemcpy(check, d_out, B*T*OC*sizeof(float), cudaMemcpyDeviceToHost);

    printf("CPU time used: %f\n", cpu_time_used);
    printf("GPU time: %f\n", milliseconds/1000.0f);

    for(int i = 0;i<B*T*OC;i++){
        if(abs(out[i] - check[i] > 1e-3f)){
            printf("Incorrect output Try again!\n");
            return 0;
        }
    }

    printf("And Correct too!\n");

    //
        //matmul_forward2
    float* another_out = (float*)malloc(B*T*OC*sizeof(float));
    float* d_another_out;
    cudaMalloc(&d_another_out, B*T*OC*sizeof(float));
    cudaEvent_t start_gpu2, stop_gpu2;
    cudaEventCreate(&start_gpu2);
    cudaEventCreate(&stop_gpu2);
    cudaEventRecord(start_gpu2);
    matmul_forward2(d_another_out, d_inp, d_weight, d_bias, B, T, C, OC);
    cudaEventRecord(stop_gpu2);
    cudaEventSynchronize(stop_gpu2);
    float milliseconds2 = 0;
    cudaEventElapsedTime(&milliseconds2, start_gpu2, stop_gpu2);
    printf("GPU time sgemm: %f\n", milliseconds2/1000.0f);

    cudaMemcpy(another_out, d_another_out, B*T*OC*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i=0;i<B*T*OC;i++){
        if((out[i] - another_out[i]) > 1e-5){
            printf("Incorrect try again\n");
            return 0;
        }
    }
    printf("Correct Sgemm\n");

    return 0;
}