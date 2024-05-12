#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#pragma GCC target("avx2")

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

void matmul_normal(float* out, float* a, float* _b, int N){
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

    clock_t start, mid;
    double cpu_time_used;
    start = clock();
    matmul_forward(out, inp, weight, bias, B, T, C, OC);
    mid = clock();
    cpu_time_used = ((double) (mid - start)) / CLOCKS_PER_SEC;
    printf("CPU time used: %f\n", cpu_time_used);

    //normal
    int N = 1024;
    float *a, *b, *c;
    a = (float*)malloc(N*N*sizeof(float));
    b = (float*)malloc(N*N*sizeof(float));
    c = (float*)malloc(N*N*sizeof(float));
    start = clock();
    matmul_normal(c, a, b, N);
    mid = clock();
    cpu_time_used = ((double) (mid - start)) / CLOCKS_PER_SEC;
    printf("CPU UNBATCHED time used: %f\n", cpu_time_used);

    free(inp);
    free(weight);
    free(bias);
    free(out);
    free(a);
    free(b);
    free(c);

    return 0;
}