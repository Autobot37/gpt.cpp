#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <x86intrin.h>
#include <string.h>
#include "mkl.h"
#include <sys/time.h> 

#define min(a,b) ((a) < (b) ? (a) : (b))

//inp(B,T,C) @  weight(3*C, C).T -> out(B,T,3*C)
void matmul_forward(float* out, float* inp, float* weight, float* bias, int B, int T, int C, int OC){
    #pragma omp parallel for collapse(2) schedule(static)    
    for(int b = 0;b<B;b++){
        for(int t = 0;t<T;t++){
            float* inp_p = inp + b * T * C + t * C;
            float* out_p = out + b * T * OC + t * OC;
            for(int t2=0;t2<OC;t2++){
                float* weight_p = weight + t2* C;
                float val = (bias != NULL) ? bias[t2] : 0.0f;
                #pragma omp simd reduction(+:val) 
                for(int i = 0;i<C;i++){
                    val += inp_p[i] * weight_p[i];
                }
                out_p[t2] = val;
            }
        }
    }
}

// Function to perform matrix multiplication using MKL
void matmul_forward_blas(float* out, float* inp, float* weight, float* bias, int B, int T, int C, int OC) {
    int M = B * T;
    int N = OC;
    int K = C;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, inp, K, weight, N, 0.0f, out, N);

    if (bias) {
        #pragma omp parallel for
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                out[i * N + j] += bias[j];
            }
        }
    }
}


void rand_init(float* arr, int size){
    for(int i = 0;i<size;i++){
        arr[i] = (float)rand() / RAND_MAX;
    }
}

int main(){

    int B = 32;
    int T = 1024;
    int C = 768;
    int OC = 3*C;

    float *inp, *weight, *bias, *out;
    inp = (float*)malloc(B*T*C*sizeof(float));
    weight = (float*)malloc(OC*C*sizeof(float));
    bias = (float*)malloc(OC*sizeof(float));
    out = (float*)malloc(B*T*OC*sizeof(float));
    rand_init(inp, B*T*C);
    rand_init(weight, OC*C);
    rand_init(bias, OC);

    struct timeval start, end;

    gettimeofday(&start, NULL);
    for (int i = 0; i < 4; i++) {
        matmul_forward(out, inp, weight, bias, B, T, C, OC);
    }
    gettimeofday(&end, NULL);
    double custom_duration = ((end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;

    gettimeofday(&start, NULL);
    for (int i = 0; i < 4; i++) {
        matmul_forward_blas(out, inp, weight, bias, B, T, C, OC);
    }
    gettimeofday(&end, NULL);
    double blas_duration = ((end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;

    printf("Custom implementation duration: %.6f seconds\n", custom_duration);
    printf("OpenBLAS implementation duration: %.6f seconds\n", blas_duration);
    
   return 0;
}