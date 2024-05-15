#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <x86intrin.h>
#include <string.h>

typedef int vec __attribute__ ((vector_size (32)));

#define min(a,b) ((a) < (b) ? (a) : (b))

vec* alloc(int n){
    vec* ptr = (vec*)aligned_alloc(32, n*sizeof(vec));
    memset(ptr, 0, n*sizeof(vec));
    return ptr;
}
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

void matmul_simd(float* out, float* _a, float* _b, int N){
    
    int nB = (N+8)/8;
    vec* a = alloc(N*nB);
    vec* b = alloc(N*nB);

    for(int i = 0;i<N;i++){
        for(int j = 0;j<N;j++){
            a[i*nB + j/8][j%8] = _a[i*N + j];
            b[i*nB + j/8][j%8] = _b[j*N + i];
        }
    }
    for(int i = 0;i<N;i++){
        for(int j = 0;j<N;j++){
            vec s = {};

            for(int k = 0;k<nB;k++){
                s += a[i*nB+k] * b[j*nB+k];
            }
            for(int k = 0;k<8;k++){
                out[i*N+j] += s[k];
            }
        }
    }
}

void matmul_normal(float* __restrict__ out, float* a, float* _b, int N){
    float* b = (float*)malloc(N*N*sizeof(float));
    for(int i=0;i<N;i++){
        for(int j = 0;j<N;j++){
            b[i*N+j] = _b[j*N+i];
        }
    }
    for(int i = 0;i<N;i++){
        for(int j = 0;j<N;j++){
            float val = 0.0f;
            for(int k = 0;k<N;k++){
                val += a[i*N+k] * b[j*N+k];
            }
            out[i*N+j] = val;
        }
    }
}

void rand_init(float* arr, int size){
    for(int i = 0;i<size;i++){
        arr[i] = (float)rand() / RAND_MAX;
    }
}


int main(){

    int mul = 4;
    int B = 4*mul;
    int T = 128*mul;
    int C = 128*mul;
    int OC = 128*2*mul;

    float *inp, *weight, *bias, *out;
    inp = (float*)malloc(B*T*C*sizeof(float));
    weight = (float*)malloc(OC*C*sizeof(float));
    bias = (float*)malloc(OC*sizeof(float));
    out = (float*)malloc(B*T*OC*sizeof(float));
    rand_init(inp, B*T*C);
    rand_init(weight, OC*C);
    rand_init(bias, OC);

    matmul_forward(out, inp, weight, bias, B, T, C, OC);
    //normal
    int N = 1920;
    float *a, *b, *c;
    a = (float*)malloc(N*N*sizeof(float));
    b = (float*)malloc(N*N*sizeof(float));
    c = (float*)malloc(N*N*sizeof(float));
    matmul_normal(c, a, b, N);
    //SIMD
    matmul_simd(c, a, b, N);    
    //blocked
    matmul_kernel(c, a, b, N);
    

    return 0;
}